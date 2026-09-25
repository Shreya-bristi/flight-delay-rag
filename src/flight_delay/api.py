"""
HTTP API for the flight-delay RAG assistant.

POST /ask is the single production answer path.
The optional /ask/stream endpoint reuses the same validated answer and emits SSE.
Prometheus metrics are exposed on /metrics.

"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .config import get_settings
from .metrics import airlabs_quota_remaining, index_chunks_total
from .pipeline import build_pipeline

STATE: dict = {}

# minimum evidence budget required for serving
MIN_EVIDENCE_TOKENS = 1000


def check_serving_settings(settings) -> None:
    """fail startup when serving configuration is unsafe or incomplete"""
    from .generation import evidence_token_budget

    problems = settings.test_double_problems()
    if settings.llm_provider != "echo" and not (settings.llm_base_url and settings.llm_model):
        problems.append(
            "no generator is configured: set LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL and the rest of "
            "the adopted Stage 2 winner's printed production settings (there is no default model)")
    probe = "What am I owed after my flight was cancelled?"
    budget = evidence_token_budget(settings, probe, None, "x" * 1400 * settings.max_validation_retries)
    if budget < MIN_EVIDENCE_TOKENS:
        problems.append(
            f"LLM_CONTEXT_WINDOW={settings.llm_context_window} leaves {budget} tokens for sources "
            f"after the system prompt and CONTEXT_ANSWER_RESERVE_TOKENS="
            f"{settings.context_answer_reserve_tokens} "
            f"(minimum {MIN_EVIDENCE_TOKENS}); raise the served context window and this setting")
    if problems:
        raise RuntimeError("refusing to serve: " + "; ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI):  
    """
    build shared application components once at startup
    """
    s = get_settings()
    check_serving_settings(s)
    # configure application logging
    app_log = logging.getLogger("flight_delay")
    if not app_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
        app_log.addHandler(handler)
    app_log.setLevel(s.log_level.upper())
    STATE["settings"] = s
    STATE["pipeline"] = build_pipeline(s)
    # load the reranker during startup so model-loading failures fail the rollout
    getattr(STATE["pipeline"].retriever.reranker, "model", None)
    store = STATE["pipeline"].retriever.store
    # ensure conversation tables exist independently of corpus indexing
    with contextlib.suppress(Exception):
        if hasattr(store, "ensure_chat_schema"):
            store.ensure_chat_schema()
    with contextlib.suppress(Exception):
        index_chunks_total.set(store.count())
    # Publish the current AirLabs quota before the first lookup
    with contextlib.suppress(Exception):
        quota = getattr(getattr(STATE["pipeline"], "tool", None), "quota", None)
        if quota is not None:
            airlabs_quota_remaining.set(quota.remaining())
    yield
    STATE.clear()


app = FastAPI(
    title="Flight Delay Compensation Assistant",
    version="0.1.0",
    description="Grounded Q&A over airline policies and aviation regulations, with live flight status.",
    lifespan=lifespan,
)


Regime = Literal["US", "EU", "UK"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # Conversation credentials returned by a previous response
    conversation_id: str | None = Field(None, max_length=64, pattern=r"^[A-Za-z0-9-]+$")
    conversation_token: str | None = Field(None, max_length=128, pattern=r"^[a-f0-9]+$")
    # optional manual jurisdiction scope
    jurisdiction: Regime | list[Regime] | None = None


class CitationOut(BaseModel):
    marker: str
    doc_title: str
    section_id: str
    source_url: str


class AskResponse(BaseModel):
    answer: str
    intent: str
    citations: list[CitationOut]
    flight: dict | None
    validation_failures: list[str]
    retry_count: int
    timings_ms: dict[str, float]
    conversation_id: str
    # Send with conversation_id to continue the conversation
    conversation_token: str
    # Pipeline outcome
    outcome: str = "answered"
    # True when answer is a question back to the passenger 
    clarification: bool = False
    # The route assumption the answer opens with, when the passenger could not say
    assumption: str | None = None
    # IATA code when the question is about a carrier this assistant does not cover
    unsupported_airline: str | None = None


# ---------------------------------------------------------------- rate limit
# Shared database rate limiting when available; bounded in-process fallback locally
_WINDOW = 60.0
_MAX_TRACKED_CLIENTS = 10_000
_BUCKET: dict[str, deque] = {}


def _rate_limited_in_process(key: str, limit: int) -> bool:
    now = time.time()
    hits = _BUCKET.setdefault(key, deque())
    while hits and now - hits[0] >= _WINDOW:
        hits.popleft()
    hits.append(now)
    if len(_BUCKET) > _MAX_TRACKED_CLIENTS:
        for k in [k for k, v in _BUCKET.items() if not v or now - v[-1] >= _WINDOW]:
            _BUCKET.pop(k, None)
    if len(_BUCKET) > _MAX_TRACKED_CLIENTS:
        
        idle_first = sorted(_BUCKET, key=lambda k: _BUCKET[k][-1] if _BUCKET[k] else 0.0)
        for k in idle_first[:len(_BUCKET) - _MAX_TRACKED_CLIENTS]:
            if k != key:
                _BUCKET.pop(k, None)
    return len(hits) > limit


def _rate_limited(key: str, settings, store=None) -> bool:
    if store is not None and hasattr(store, "rate_limit_hit"):
        try:
            return store.rate_limit_hit(key, settings.rate_limit_per_minute, int(_WINDOW))
        except Exception:
            pass   # database trouble must not turn every request into a 429
    return _rate_limited_in_process(key, settings.rate_limit_per_minute)


def client_key(request: Request, settings) -> str:
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


# ------------------------------------------------------- conversation tokens
#HMAC token prevents a conversation ID alone from granting access
_PROCESS_SECRET = secrets.token_hex(32)


def _secret(settings) -> bytes:
    return (settings.conversation_secret or _PROCESS_SECRET).encode()


def conversation_token(conversation_id: str, settings) -> str:
    return hmac.new(_secret(settings), conversation_id.encode(), hashlib.sha256).hexdigest()[:40]


def resolve_conversation(req: AskRequest, settings) -> str:
    """The conversation to use: a new one, or the requested one if its token matches."""
    if req.conversation_id is None:
        if req.conversation_token is not None:
            raise HTTPException(400, "conversation_token sent without conversation_id")
        return str(uuid.uuid4())
    expected = conversation_token(req.conversation_id, settings)
    if req.conversation_token is None or not hmac.compare_digest(expected, req.conversation_token):
        raise HTTPException(403, "unknown conversation or invalid conversation_token")
    return req.conversation_id


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """
    Readiness is distinct from liveness: the process can be alive but have an
    empty index, in which case it must not receive traffic. Kubernetes uses this
    distinction to avoid routing to a pod that would return garbage.
    """
    p = STATE.get("pipeline")
    if p is None:
        raise HTTPException(503, "pipeline not built")
    try:
        n = p.retriever.store.count()
    except Exception as e:
        raise HTTPException(503, f"store unreachable: {e}") from e
    # Refresh the chunk gauge because indexing runs in a separate process
    index_chunks_total.set(n)
    if n == 0:
        raise HTTPException(503, "index is empty - run scripts/index_corpus.py")
    identity = None
    if hasattr(p.retriever.store, "active_identity"):
        # reject an index built with incompatible parser, embedding, or chunk settings
        from .ingest import PARSER_VERSION
        from .store import identity_mismatches

        try:
            identity = p.retriever.store.active_identity()
        except Exception as e:
            raise HTTPException(503, f"store unreachable: {e}") from e
        problems = identity_mismatches(identity, STATE["settings"], PARSER_VERSION)
        if problems:
            raise HTTPException(503, "index incompatible with settings: " + "; ".join(problems))
    from .generation import SYSTEM_PROMPT_SHA256

    s = STATE["settings"]
    return {"status": "ready", "chunks": n,
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "llm": {"provider": s.llm_provider, "model": s.llm_model,
                    "input_assembly_budget_tokens": s.llm_context_window,
                    "context_answer_reserve_tokens": s.context_answer_reserve_tokens,
                    "max_completion_tokens": s.completion_cap_tokens,
                    "extra_body": s.llm_extra_body},
            "index": {k: identity.get(k) for k in ("corpus_digest", "parser_version",
                                                   "chunk_target_tokens", "embedding_model",
                                                   "built_at")} if identity else None}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _answer(req: AskRequest, request: Request) -> AskResponse:
    """The one production answer path, shared by /ask and the internal /ask/stream."""
    p = STATE["pipeline"]
    s = STATE["settings"]
    if _rate_limited(client_key(request, s), s, getattr(p, "store", None)):
        raise HTTPException(429, "rate limit exceeded")

    conv = resolve_conversation(req, s)
    flt = None
    if req.jurisdiction:
        regimes = [req.jurisdiction] if isinstance(req.jurisdiction, str) else list(req.jurisdiction)
        regimes = sorted(set(regimes))
        # Match the jurisdiction structure produced by route detection
        flt = {"jurisdiction_scope": regimes, "jurisdiction_governing": regimes}

    # RagPipeline.run: confidence gate -> generate -> validate -> one retry.
    a = p.run(req.question, conversation_id=conv, flt=flt)
    return AskResponse(
        answer=a.text,
        intent=a.intent,
        citations=[
            CitationOut(
                marker=c.marker, doc_title=c.doc_title,
                section_id=c.section_id, source_url=c.source_url,
            )
            for c in a.citations
        ],
        flight=(
            {
                "flight": a.live_data.flight_iata,
                "route": " to ".join(x for x in (a.live_data.dep_iata, a.live_data.arr_iata) if x),
                "status": a.live_data.status,
                "delay_minutes": a.live_data.worst_delay_min,
                "cause": "not reported by data source",
            }
            if a.live_data
            else None
        ),
        validation_failures=a.validation_failures,
        retry_count=a.retry_count,
        timings_ms={k: round(v, 1) for k, v in a.timings_ms.items()},
        conversation_id=conv,
        conversation_token=conversation_token(conv, s),
        outcome=a.outcome,
        clarification=a.clarification,
        assumption=a.assumption,
        unsupported_airline=a.unsupported_airline,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    """Production answer endpoint, and the one the UI uses."""
    return _answer(req, request)


@app.post("/ask/stream", include_in_schema=False)
def ask_stream(req: AskRequest, request: Request):
    """
    
    Optional internal SSE endpoint.

    Reuses the validated /ask response and emits it as SSE events.

    """
    import json

    if not STATE["settings"].enable_stream_endpoint:
        raise HTTPException(404, "Not Found")
    r = _answer(req, request)

    def gen():
        meta = {"intent": r.intent, "outcome": r.outcome, "sources": len(r.citations),
                "conversation_id": r.conversation_id, "conversation_token": r.conversation_token,
                "clarification": r.clarification, "assumption": r.assumption,
                "unsupported_airline": r.unsupported_airline}
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        yield f"event: token\ndata: {json.dumps({'t': r.answer})}\n\n"
        validation = {"ok": not r.validation_failures, "failures": r.validation_failures,
                      "retry_count": r.retry_count, "citations": [c.model_dump() for c in r.citations]}
        yield f"event: validation\ndata: {json.dumps(validation)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
def index_page():
    """
    Serve the lightweight built-in demo UI
    """
    return HTML_PAGE


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flight Delay Assistant</title>
<style>
 :root{--bg:#0f1216;--card:#171b21;--user:#23406f;--fg:#e6e9ef;--mut:#8b93a1;--acc:#4c8dff;--warn:#ffb020;--line:#232830}
 *{box-sizing:border-box}
 html,body{height:100%}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
 header h1{font-size:17px;margin:0;flex:1}
 header .sub{color:var(--mut);font-size:12px}
 #new{background:transparent;border:1px solid #2a3038;color:var(--fg);padding:6px 12px;border-radius:8px;cursor:pointer;font-size:13px}
 #thread{flex:1;overflow-y:auto;padding:20px}
 .inner{max-width:780px;margin:0 auto;display:flex;flex-direction:column;gap:12px}
 .msg{max-width:88%;padding:12px 14px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
 .me{align-self:flex-end;background:var(--user)}
 .bot{align-self:flex-start;background:var(--card);border:1px solid var(--line)}
 .ask{border-color:var(--warn)}
 .tag{display:block;font-size:12px;color:var(--warn);margin-bottom:6px;white-space:normal}
 .flight{border-left:3px solid var(--warn);padding-left:10px;margin-bottom:10px;font-size:13px;white-space:normal}
 .cite{font-size:13px;color:var(--mut);margin-top:10px;border-top:1px solid var(--line);padding-top:8px;white-space:normal}
 .cite a{color:var(--acc);text-decoration:none}
 .meta{font-size:11px;color:var(--mut);margin-top:6px;white-space:normal}
 .bad{color:#ff6b6b}
 .ex{color:var(--mut);font-size:13px;white-space:normal;margin-top:8px}
 .ex span{cursor:pointer;text-decoration:underline;display:block;margin:4px 0}
 footer{border-top:1px solid var(--line);padding:12px 20px 16px}
 .row{max-width:780px;margin:0 auto;display:flex;gap:8px}
 textarea{flex:1;resize:none;height:48px;padding:12px 14px;border-radius:10px;border:1px solid #2a3038;background:var(--card);color:var(--fg);font:inherit}
 textarea.waiting{border-color:var(--warn)}
 #go{padding:0 20px;border-radius:10px;border:0;background:var(--acc);color:#fff;font-weight:600;cursor:pointer}
 #go:disabled{opacity:.5;cursor:default}
 .disc{max-width:780px;margin:6px auto 0;color:var(--mut);font-size:11px}
</style></head><body>
<header><h1>Flight Delay Compensation Assistant</h1><span class="sub">Not legal advice</span><button id="new" title="Start a new conversation">New chat</button></header>
<div id="thread"><div class="inner" id="inner">
<div class="msg bot">Hi! Tell me what happened to your flight (delay, cancellation, denied boarding, missed connection) and I will explain your options from the regulations and the airline's own policies.<div class="ex">Try:
<span onclick="fill(this)">How much compensation for denied boarding?</span>
<span onclick="fill(this)">My Delta flight from London to New York was delayed 5 hours. What am I owed?</span>
<span onclick="fill(this)">My American flight from Dallas to Chicago was cancelled. Can I get a refund?</span>
</div></div>
</div></div>
<footer><div class="row">
  <textarea id="q" placeholder="Describe your situation..." autofocus></textarea>
  <button id="go">Send</button>
</div><div class="disc">Answers cite the regulation or airline document they come from. Flight data does not report the delay <em>cause</em>, which often decides entitlement.</div></footer>
<script>
// Every server value is inserted as TEXT (esc) or as a checked http(s) URL: an
// answer, a document title or an error detail must never become markup.
let CONV=null, TOKEN=null; // kept so a reply to a clarifying question joins its conversation
const DEBUG=new URLSearchParams(location.search).has('debug'); // ?debug=1 shows outcome + timings
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const safeUrl=u=>{try{const x=new URL(u);return (x.protocol==='https:'||x.protocol==='http:')?x.href:null}catch(e){return null}};
const inner=document.getElementById('inner'), thread=document.getElementById('thread'), box=document.getElementById('q');
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;inner.appendChild(d);thread.scrollTop=thread.scrollHeight;return d}
function fill(el){box.value=el.textContent.trim();ask()}
function waiting(on){box.classList.toggle('waiting',on);
 box.placeholder=on?'Reply to the question above (flight number, airports, or country)...':'Ask a follow-up or describe another situation...'}
async function ask(){
 const q=box.value.trim(); if(!q)return;
 const btn=document.getElementById('go');
 add('me',esc(q)); box.value=''; btn.disabled=true;
 const pending=add('bot','Thinking...');
 try{
  const body={question:q}; if(CONV){body.conversation_id=CONV;body.conversation_token=TOKEN}
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let d; try{d=await r.json()}catch(e){d={detail:'server error ('+r.status+')'}}
  if(!r.ok){const detail=typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||'error');
    pending.className='msg bot bad'; pending.innerHTML=esc(detail); return}
  CONV=d.conversation_id; TOKEN=d.conversation_token;
  let h='';
  if(d.clarification){h+='<span class="tag">I need one detail before I can answer. Reply below:</span>'}
  if(d.flight){h+='<div class="flight"><b>'+esc(d.flight.flight)+'</b> '+esc(d.flight.route||'')+' - status: '+esc(d.flight.status||'?')+
    ' - delay: '+esc(d.flight.delay_minutes!=null?d.flight.delay_minutes+' min':'none reported')+
    '<br>delay cause: <b>'+esc(d.flight.cause)+'</b></div>'}
  // [S1]-style markers link sentences to sources so the validator can check them;
  // the passenger gets the source list below instead. Strip ANY [Sn] marker, not only
  // the ones in d.citations: a marker the model invented, or one whose source was
  // dropped from the list, used to survive into the visible answer as "[S5]" noise.
  // Handles runs like "[S5][S3]" and the comma form "[S5, S6]" that generation.py
  // normalises on the way in - belt and braces, since one that arrived unnormalised
  // would be shown to the passenger. Eats the space before the marker so the
  // sentence still reads "...a full cash refund." and not "...refund ."
  // (HTML_PAGE is a plain Python string, so every backslash here is doubled in the
  //  source and reaches the browser singly.)
  let shown=String(d.answer||'').replace(/[ \\t]*\\[S\\d+(?:\\s*[,;]\\s*S\\d+)*\\](?:[ \\t]*\\[S\\d+(?:\\s*[,;]\\s*S\\d+)*\\])*/g,'');
  h+=esc(shown);
  if(d.citations.length){h+='<div class="cite"><b>Sources</b><br>'+
    d.citations.map(c=>{const u=safeUrl(c.source_url);
      return esc(c.doc_title)+' - '+esc(c.section_id)+
      (u?' <a href="'+esc(u)+'" target="_blank" rel="noopener noreferrer">link</a>':'')}).join('<br>')+'</div>'}
  // Outcome, timings and the "failed citation checks" notice are for whoever is
  // testing, not for a passenger (user, Session 37): shown only with ?debug=1.
  if(DEBUG){
   if(d.outcome==='validation_failed'){h+='<div class="meta bad">The generated answer failed citation checks and was not shown.</div>'}
   const tm=d.timings_ms||{}, sec=v=>(v/1000).toFixed(1)+' s';
   const retr=['embed','dense','sparse','fuse','governing','lanes','rerank'].reduce((s,k)=>s+(tm[k]||0),0);
   h+='<div class="meta">'+esc(d.outcome)+' - '+sec(tm.total||0)+
      (retr?' (search '+sec(retr)+(tm.generate?', model '+sec(tm.generate):'')+')':'')+
      (d.retry_count?' - retried '+esc(d.retry_count)+'x':'')+'</div>';
  }
  pending.className='msg bot'+(d.clarification?' ask':''); pending.innerHTML=h;
  waiting(!!d.clarification);
 }catch(e){pending.className='msg bot bad'; pending.innerHTML=esc(e)}
 finally{btn.disabled=false; box.focus(); thread.scrollTop=thread.scrollHeight}
}
document.getElementById('go').onclick=ask;
box.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});
document.getElementById('new').onclick=()=>{CONV=null;TOKEN=null;inner.querySelectorAll('.msg').forEach((m,i)=>{if(i>0)m.remove()});waiting(false);box.placeholder='Describe your situation...';box.focus()};
</script></body></html>
"""


def main():
    import uvicorn

    uvicorn.run(
        "flight_delay.api:app",
        host="0.0.0.0",  
        port=int(os.environ.get("PORT", 8000)),
        reload=bool(os.environ.get("RELOAD")),
    )


if __name__ == "__main__":
    main()

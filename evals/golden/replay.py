

from __future__ import annotations

# The reply used when a case does not script one. With airports on the case it
# names them; without, the passenger cannot help and production must assume.
NO_DETAILS_REPLY = "Sorry, I don't have my flight number or the airport details."


def default_reply(origin: str | None, dest: str | None) -> str:
    if origin and dest:
        return f"I don't have the flight number. I flew from {origin} to {dest}."
    return NO_DETAILS_REPLY


def fixture_fetcher(fixture: dict | None):
    """A flight lookup that serves one synthetic record instead of AirLabs."""
    from flight_delay.models import FlightStatus

    def fetch(flight_no: str):
        if not fixture or flight_no.upper() != fixture["flight_iata"].upper():
            return None
        return FlightStatus(
            flight_iata=fixture["flight_iata"], airline_iata=fixture.get("airline_iata"),
            status=fixture.get("status"), dep_iata=fixture.get("dep_iata"), dep_time=None,
            dep_estimated=None, dep_delayed=fixture.get("dep_delayed"),
            arr_iata=fixture.get("arr_iata"), arr_time=None, arr_estimated=None,
            arr_delayed=fixture.get("arr_delayed"), delayed=fixture.get("delayed"),
            from_cache=True, source="synthetic",   # labelled to the model as a synthetic record
        )

    return fetch


class _NoQuota:
    def remaining(self) -> int:
        return 10**9


class FixtureTool:
    """
    The AirLabs tool as every eval harness sees it: one case's synthetic flight
    record (golden `flight_fixture`), served by fixture_fetcher, and nothing else.
    Never touches the network or the quota file, so a case is deterministic and
    an eval run with AIRLABS_KEY set costs no live calls.

    Every lookup is recorded in `calls`, so a harness can assert what production
    asked for — an unsupported carrier's flight must never be looked up at all.
    """

    def __init__(self, fixture: dict | None):
        self.quota = _NoQuota()
        self.calls: list[str] = []
        self._fetch = fixture_fetcher(fixture)

    def get_flight(self, flight_no: str, allow_live: bool = False):  # noqa: ARG002
        self.calls.append(flight_no)
        return self._fetch(flight_no)


def replay(question: str, reply: str | None, fixture: dict | None = None):
    """
    Returns (first_plan, final_plan, reply_used). reply_used is None when
    production answered the question without asking.
    """
    from flight_delay.pipeline import plan_turn

    fetch = fixture_fetcher(fixture)
    first = plan_turn(question, [], fetch)
    if first.clarification is None:
        return first, first, None
    reply = reply or NO_DETAILS_REPLY
    final = plan_turn(reply, [("user", first.question), ("assistant", first.clarification)], fetch)
    return first, final, reply


class ConversationMemory:
    """Chat history for replaying a conversation without Postgres."""

    def __init__(self):
        self.messages: dict[str, list[tuple[str, str]]] = {}

    def history(self, conv_id, limit=6):
        return self.messages.get(conv_id, [])[-limit:]

    def save_message(self, conv_id, role, content, citations=None):  # noqa: ARG002
        self.messages.setdefault(conv_id, []).append((role, content))

    def load_state(self, conv_id):
        return getattr(self, "states", {}).get(conv_id)

    def save_state(self, conv_id, state):
        self.__dict__.setdefault("states", {})[conv_id] = state


def action_of(item: dict) -> str:
    """The golden record's expected behaviour: "answer", "clarify" or "abstain"."""
    return (item.get("expected_behavior") or {}).get("action", "answer")


def run_conversation(pipeline, item: dict, tool: FixtureTool | None = None):
    """
    Ask the golden question; if production asks a clarifying question first,
    send the case's scripted reply in the same conversation and return the
    final answer. Chat history is kept in memory, so an eval run leaves no rows
    in Postgres.

    A case whose expected action is "clarify" is the exception: asking IS the
    expected behaviour there (the missing detail decides which regime governs),
    so the first turn is what gets returned and the scripted reply is not sent.

    Flight lookups go to FixtureTool for the whole conversation: the case's
    synthetic `flight_fixture` (the record the golden builder replayed) or
    nothing. Never AirLabs, so hybrid cases get their flight data and a run
    with AIRLABS_KEY set spends no live calls. The pipeline's real tool and
    conversation store are restored afterwards. Pass `tool` to keep the
    FixtureTool afterwards and inspect which flights were looked up.

    Every turn runs with allow_followup=False. The follow-up templates
    (flight_delay.followups) answer a recognised second message from the
    previous answer without retrieving or generating, which is right in
    production and would be scoring the wrong thing here: a case whose scripted
    reply happened to read like "I want a refund instead" would be judged on a
    template, not on the pipeline. Both eval stages therefore measure the same
    retrieve-and-generate path they always did.
    """
    saved_store, saved_tool = pipeline.store, pipeline.tool
    pipeline.tool = tool if tool is not None else FixtureTool(item.get("flight_fixture"))
    try:
        if not item.get("reply") or action_of(item) == "clarify":
            return pipeline.run(item["question"], allow_retry=True, allow_followup=False)
        pipeline.store = ConversationMemory()
        cid = f"eval-{item['id']}"
        first = pipeline.run(item["question"], conversation_id=cid, allow_retry=True,
                             allow_followup=False)
        if not first.clarification:
            return first
        return pipeline.run(item["reply"], conversation_id=cid, allow_retry=True,
                            allow_followup=False)
    finally:
        pipeline.store, pipeline.tool = saved_store, saved_tool

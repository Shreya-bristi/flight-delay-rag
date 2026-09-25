"""
Deterministic follow-up replies, answered from the PREVIOUS turn's answer.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .generation import _MARKER_RE, _is_recommendation
from .generation import _answer_sentences as split_answer_sentences

# A follow-up is a short move in a conversation such as I want a refund instead", not a
# new situation. Anything longer is a fresh question and takes the full path.
FOLLOWUP_MAX_CHARS = 240
MAX_QUOTED_SENTENCES = 3
STORED_ANSWER_MAX_CHARS = 8000


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


# ---------------------------------------------------------------- topics
# What to look for among the previous answer's cited sentences
_REBOOK = _rx(r"re-?book|re-?rout|alternative flight|another flight|another carrier|"
              r"next available|earliest|onward|comparable air transportation|seat")
# NOT "reimburs": a hotel-reimbursement sentence is not about the ticket refund,
# and it crowded the real refund quotes out of the three slots.
_REFUND = _rx(r"refund|money back|unused|unflown|ticket value")
_HOTEL = _rx(r"hotel|accommodation|lodging|overnight")
_MEAL = _rx(r"meal|food|refreshment|drink")
_TRANSFER = _rx(r"transport|transfer|taxi|ground|between the airport")
_CARE = _rx(r"hotel|accommodation|meal|refreshment|care|assistance|overnight")
_CAUSE = _rx(r"extraordinary|control|controllab|weather|mechanical|maintenance|crew|"
             r"cause|reason|circumstance")
_AMOUNT = _rx(r"[€£$]|\bEUR\b|\bGBP\b|\bUSD\b|euro|pound|dollar|compensat|amount|"
              r"fixed (?:sum|payment)")
_UK = _rx(r"UK\s?261|United Kingdom|\bUK\b|CAA|Civil Aviation Authority")
_EU = _rx(r"EU\s?261|261/2004|European|\bEU\b|Commission")
_US = _rx(r"\bDOT\b|14 CFR|Part 2[0-9]{2}|United States|\bUS\b|federal|"
          r"Department of Transportation")
_CONNECTION = _rx(r"connect|final destination|onward|itinerary|leg|through")
_CLAIM = _rx(r"claim|complaint|customer (?:relations|service)|file|submit|contact|write")
_DEADLINE = _rx(r"business day|calendar day|within|deadline|prompt")
_CREDIT = _rx(r"credit|voucher|accept|in lieu|instead of")
_DOCS = _rx(r"receipt|document|record|evidence|proof|boarding pass|itinerary")
_DELAY = _rx(r"delay|hour|arrival|arrive|late|threshold")


# ------------------------------------------------------------ shared steps

_ASK_CAUSE = ("Ask {airline} to tell you the reason they have on record for the disruption, "
              "and to say whether they treat it as their own doing.")
_KEEP_RECEIPTS = ("Keep the disruption notice, both itineraries, your boarding passes and every "
                  "itemised receipt, so you can back up a claim later.")
_ASK_AGENT = ("Ask the {airline} agent for the earliest seat they can put you on and what the "
              "disruption is coded as.")
_ASK_BEFORE_SPENDING = ("Check with {airline} what they will cover before you pay for anything "
                        "yourself, and keep the receipt either way.")
_TELL_ROUTE = ("Tell me the airports at both ends of the flight and I will answer again for that "
               "exact route.")
# "Can you put me on the next flight?" is addressed to this assistant, which has no
# booking integration and must not let the passenger believe otherwise while they
# are standing at a gate.
_CANNOT_BOOK = ("I cannot book anything for you myself, but you can pick another flight in the "
                "{airline} app or website, at an airport kiosk, or with an agent at the airport.")


@dataclass(frozen=True)
class FollowUp:
    """One recognised follow-up move and the deterministic reply """

    id: str
    trigger: re.Pattern[str]
    topic: re.Pattern[str]
    lead: str          
    step: str          

def _f(id_: str, trigger: str, topic: re.Pattern[str], lead: str, step: str) -> FollowUp:
    return FollowUp(id_, _rx(trigger), topic, lead, step)


TEMPLATES: tuple[FollowUp, ...] = (
    # -- what the passenger wants to do next --------------------------------
    _f("still_travel",
       r"still (?:want|need|wish) to (?:travel|fly|go)|"
       r"i (?:want|need) to (?:still )?(?:travel|fly|get there)",
       _REBOOK,
       "You still want to travel, so the re-routing part of the answer is the part that counts.",
       "Ask {airline} for the earliest seat to your final destination, and hold on to your "
       "original booking reference. " + _CANNOT_BOOK),
    _f("next_flight",
       r"next available flight|put me on (?:the|another)|re-?book me|get me on|earliest flight",
       _REBOOK,
       "Here is what I already told you about being put on another flight:",
       "Ask {airline} to rebook you onto the next flight they have. " + _CANNOT_BOOK),
    _f("other_airline",
       r"(?:another|different|other|partner|alternative) (?:airline|carrier)|"
       r"fly with (?:someone|somebody) else|interline",
       _REBOOK,
       "Here is what I already told you about being moved to a different flight or carrier:",
       "Ask {airline} whether they will put you on another carrier, and ask them to confirm it "
       "in writing."),
    _f("want_refund",
       r"(?:want|prefer|rather have|like) (?:a |my |the )?(?:refund|money back)|refund instead|"
       r"just refund|cancel and refund",
       _REFUND,
       "Here is what I already told you about a refund:",
       "Ask {airline} for the refund in writing and keep the reference they give you."),
    _f("not_travel",
       r"(?:not|no longer|don'?t|do not) (?:want to |wish to )?(?:travel|fly|go)(?: any ?more)?|"
       r"choose not to travel|give up (?:on )?(?:the|this) trip",
       _REFUND,
       "You no longer want to travel, so the refund part of the answer is the part that counts.",
       "Tell {airline} you are declining the replacement before you accept anything else, and ask "
       "them to record that."),
    _f("offered_next_day",
       r"(?:offered|gave|giving|put) me (?:a )?(?:flight|seat)|flight tomorrow|next day|"
       r"the next morning|gets me there (?:the )?(?:next|following)",
       _REBOOK,
       "Here is what I already told you about the replacement they are offering:",
       "Ask {airline} whether anything earlier exists before you accept that one, and tell me "
       "which way you want to go."),

    # -- cause --------------------------------------------------------------
    _f("cause_unknown",
       r"don'?t know why|do not know why|no idea why|not sure why|"
       r"they (?:haven'?t|have not) said",
       _CAUSE,
       "Without the cause, the part of the answer that does not depend on it is the part to lean "
       "on:",
       _ASK_CAUSE),
    _f("cause_no_reason",
       r"(?:don'?t|won'?t|do not|will not|refuse)[^.]{0,20}(?:tell|say|give)[^.]{0,20}"
       r"(?:reason|why|cause)|no reason given",
       _CAUSE,
       "Here is what I already told you about the cause and what turns on it:",
       "Ask {airline} in writing for the recorded reason, and keep their reply."),
    _f("cause_weather",
       r"(?:said|was|because of|due to|blamed)[^.]{0,20}"
       r"(?:weather|storm|fog|snow|wind|hurricane)|\bweather\b",
       _CAUSE,
       "They are calling it weather, so here is what I already told you about causes of that kind:",
       "Ask {airline} to confirm that coding in writing, and keep your own receipts in case it "
       "changes."),
    _f("cause_mechanical",
       r"mechanical|maintenance|technical (?:issue|problem|fault)|"
       r"aircraft (?:issue|problem|fault)|broken (?:plane|aircraft)",
       _CAUSE,
       "They are calling it a mechanical problem, so here is what I already told you about causes of "
       "that kind:",
       "Ask {airline} whether they are treating this one as their own doing, and get that in "
       "writing."),
    _f("cause_crew",
       r"crew|staffing|pilot (?:issue|problem|shortage)|no (?:pilot|crew)",
       _CAUSE,
       "They are calling it a crew problem, so here is what I already told you about causes of that "
       "kind:",
       "Ask {airline} whether they are treating this one as their own doing, and get that in "
       "writing."),
    _f("outside_control",
       r"outside (?:its|their|the airline'?s) control|beyond (?:its|their) control|"
       r"extraordinary circumstance|not (?:their|its) fault|act of god",
       _CAUSE,
       "Here is what I already told you about a cause the airline says was not theirs:",
       "Ask {airline} to put that position in writing, because you will need it if you dispute "
       "it."),

    # -- care ---------------------------------------------------------------
    _f("hotel_reimbursement",
       r"(?:already |i )?(?:paid|booked|got)[^.]{0,30}hotel|hotel[^.]{0,20}myself|"
       r"reimburse[^.]{0,20}hotel|hotel[^.]{0,20}reimburs",
       _HOTEL,
       "You have already paid for the room, so here is what I already told you about accommodation:",
       "Send the itemised receipt to {airline} anyway and keep a copy, whatever they say on the "
       "phone."),
    _f("hotel_transport",
       r"transport(?:ation)?[^.]{0,20}hotel|hotel[^.]{0,20}transport|get to the hotel|"
       r"taxi|ride to the hotel|transfer",
       _TRANSFER,
       "Here is what I already told you about getting to and from the accommodation:",
       _ASK_BEFORE_SPENDING),
    _f("hotel",
       r"hotel|accommodation|somewhere to (?:sleep|stay)|put me up|overnight stay",
       _HOTEL,
       "Here is what I already told you about accommodation:",
       _ASK_BEFORE_SPENDING),
    _f("meal",
       r"meal|food|eat\b|drink|refreshment|voucher for (?:food|a meal)",
       _MEAL,
       "Here is what I already told you about meals and refreshments:",
       _ASK_BEFORE_SPENDING),

    # -- what has already been accepted --------------------------------------
    _f("credit_accepted",
       r"(?:accepted|took|taken|got|have)[^.]{0,25}(?:travel credit|credit|voucher|e-?credit)|"
       r"credit instead",
       _CREDIT,
       "What you accepted, and what you were told when you accepted it, decides this one. Here is "
       "what I already told you:",
       "Find the exact wording of what you agreed to, read it back to me, and ask {airline} what "
       "it replaced."),
    _f("replacement_accepted",
       r"(?:accepted|took|taken|flew on|boarded)[^.]{0,25}"
       r"(?:replacement|rebooked|new|alternative|later)[^.]{0,15}(?:flight|seat)|"
       r"i (?:already )?travell?ed",
       _AMOUNT,
       "You travelled on the replacement, so here is what I already told you about what that changes:",
       "Ask {airline} what taking the replacement changed, and say in your claim that you took "
       "it."),

    # -- money ---------------------------------------------------------------
    _f("how_much",
       r"how much|what (?:am i|are we) (?:owed|due)|what do (?:i|we) get|"
       r"am i (?:owed|due)|entitled to",
       _AMOUNT,
       "Here is what I already told you about what is payable:",
       "Tell me the arrival delay and the cause if you have them, because both change the answer."),
    _f("refund_deadline",
       r"how long[^.]{0,30}refund|when (?:do|will) (?:i|they)[^.]{0,20}(?:refund|money)|"
       r"refund[^.]{0,20}take|deadline[^.]{0,20}refund",
       _DEADLINE,
       "Here is what I already told you about the timing:",
       "Ask {airline} to date-stamp your refund request and keep their confirmation of it."),

    # -- route ---------------------------------------------------------------
    _f("from_uk",
       r"(?:matter|difference|change anything|relevant|count)[^.]{0,40}"
       r"(?:\bUK\b|United Kingdom|Britain|England|Scotland|Wales)|"
       r"what if[^.]{0,30}from (?:the )?(?:\bUK\b|United Kingdom|Britain)",
       _UK,
       "Where the flight departed from decides which rules I search. Here is what I already told you "
       "for that side:",
       _TELL_ROUTE),
    _f("from_eu",
       r"(?:matter|difference|change anything|relevant|count)[^.]{0,40}"
       r"(?:\bEU\b|Europe|European)|"
       r"what if[^.]{0,30}from (?:the )?(?:\bEU\b|Europe)",
       _EU,
       "Where the flight departed from decides which rules I search. Here is what I already told you "
       "for that side:",
       _TELL_ROUTE),
    _f("us_to_europe",
       r"(?:from|out of) (?:the )?(?:\bUS\b|\bU\.S\.|United States|America)[^.]{0,15}"
       r"to[^.]{0,15}(?:Europe|\bEU\b|\bUK\b)",
       _US,
       "The direction of travel changes which rules I search. Here is what I already told you for a "
       "departure from the United States:",
       _TELL_ROUTE),
    _f("arrival_delay",
       r"(?:arrive|arriving|arrival|get(?:ting)? (?:in|there)|land(?:ing)?)[^.]{0,25}"
       r"(?:late|hours?|behind)|hours? late",
       _DELAY,
       "How late you actually arrive is what the thresholds are measured against. Here is what I "
       "already told you:",
       "Tell me your real arrival time against the scheduled one and I will read it against those "
       "thresholds."),
    _f("missed_connection",
       r"miss(?:ed|ing)?[^.]{0,20}connect|connecting flight|next leg|onward flight",
       _CONNECTION,
       "Here is what I already told you about being taken through to your final destination:",
       "Ask {airline} to rebook the whole itinerary through to where you are actually going, not "
       "just the leg that broke."),

    # -- what to do about it --------------------------------------------------
    _f("ask_agent",
       r"(?:ask|say to|tell)[^.]{0,20}(?:the )?(?:gate )?agent|what (?:should|do) i (?:ask|say)|"
       r"at the (?:desk|gate|counter)",
       _CARE,
       "Ask the agent three things, in this order: the earliest seat they can give you, what the "
       "disruption is coded as, and what they will cover while you wait.",
       _ASK_AGENT),
    _f("documents",
       r"(?:what|which)[^.]{0,25}(?:document|receipt|paperwork|record|proof|evidence)|"
       r"(?:keep|save|hold on to)[^.]{0,20}(?:document|receipt|paperwork)",
       _DOCS,
       "Here is what I already told you about the evidence side:",
       _KEEP_RECEIPTS),
    _f("file_claim",
       r"how (?:do|can|would) i (?:file|make|submit|start|lodge|claim)|"
       r"file a (?:claim|complaint)|where do i (?:claim|complain)|make a claim",
       _CLAIM,
       "Here is what I already told you about making the claim:",
       "Start it through the refund or customer-relations channel {airline} runs, attach the "
       "receipts, and keep the confirmation."),

    # -- last resort: a bare "what should I do?" after an answer ---------------
    _f("what_now",
       r"^\W*(?:so )?what (?:should|do|can) (?:i|we) do(?: now| next)?\W*$|"
       r"^\W*what(?:'| a)?re my options\W*$|^\W*what now\W*$",
       _REBOOK,
       "Here are the moves the answer I gave you leaves open:",
       "Tell me whether you still want to travel or would rather have the money back, and I will "
       "take you through that one."),
)

# Used when the previous answer says nothing on the topic asked about. 
NO_MATERIAL = ("My last answer did not cover that, and I am not going to guess at it.")

SCAFFOLD_FORBIDDEN = re.compile(
    r"(\d|[€£$§%]|\bmust\b|\bshall\b|\bentitled\b|\brequired\b|\bcompensat|"
    r"\bliable\b)", re.I)


@dataclass
class FollowUpReply:
    """The composed reply"""

    template_id: str
    text: str
    markers: list[str] = field(default_factory=list)   # ["S1", "S4"], in order of use
    quoted: list[str] = field(default_factory=list)
    had_material: bool = False


def match(question: str) -> FollowUp | None:
    """
    The follow-up move this message is, or None.

    """
    text = (question or "").strip()
    if not text or len(text) > FOLLOWUP_MAX_CHARS:
        return None
    for spec in TEMPLATES:
        if spec.trigger.search(text):
            return spec
    return None


def reusable(sentence: str) -> bool:
    return bool(_MARKER_RE.search(sentence)) or _is_recommendation(sentence, set())


def relevant_sentences(previous_answer: str, topic: re.Pattern[str],
                       limit: int = MAX_QUOTED_SENTENCES) -> list[str]:
    """
    Reusable sentences of the previous answer that are about this topic
    """
    out: list[str] = []
    seen: set[str] = set()
    for sentence in split_answer_sentences(previous_answer or ""):
        s = sentence.strip()
        if not s or s in seen or not topic.search(s) or not reusable(s):
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def compose(spec: FollowUp, previous_answer: str, airline: str = "the airline") -> FollowUpReply:
    """Build the reply: the scaffold, naming the carrier, around verbatim cited quotes."""
    quotes = relevant_sentences(previous_answer, spec.topic)
    name = airline or "the airline"
    if quotes:
        body = "\n".join(f"- {q}" for q in quotes)
        text = f"{spec.lead.format(airline=name)}\n\n{body}\n\n{spec.step.format(airline=name)}"
    else:
        text = f"{NO_MATERIAL.format(airline=name)}\n\n{spec.step.format(airline=name)}"
    markers = list(dict.fromkeys(_MARKER_RE.findall("\n".join(quotes))))
    return FollowUpReply(spec.id, text, markers, quotes, bool(quotes))

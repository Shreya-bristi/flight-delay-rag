"""
Evidence selectors for the golden set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    doc: str
    section: str | None = None
    has: str | None = None


# ---- doc ids ---------------------------------------------------------------
EU, UK = "eu-261-2004", "uk-261"
P250, P259, P260 = "us-cfr-250-oversales", "us-cfr-259-protections", "us-cfr-260-refunds"
DOTQA = "us-dot-refunds-qa"
# The binding texts themselves, for primary_authority_coverage (Session 24). The DOT
# refunds Q&A is the department's guidance on Part 260 even though it is indexed with
# doc_type "regulation" (changing that would change retrieval, not just reporting).
BINDING_DOCS = frozenset({EU, UK, P250, P259, P260})
AA_COC, AA_CSP = "aa-conditions-of-carriage", "aa-customer-service-plan"
DL_DOM, DL_INTL, DL_POL = "dl-contract-domestic", "dl-contract-international", "dl-delay-cancel-policy"
UA_COC, UA_CC = "ua-contract-of-carriage", "ua-customer-commitment"
WN_COC, WN_CSP = "wn-contract-of-carriage", "wn-customer-service-plan"


def _articles(doc: str) -> dict[str, Ref]:
    return {
        "A2": Ref(doc, "article-2-definitions"),
        "A3": Ref(doc, "article-3-scope"),
        "A4": Ref(doc, "article-4-denied-boarding"),
        "A5": Ref(doc, "article-5-cancellation"),
        "A6": Ref(doc, "article-6-delay"),
        "A7": Ref(doc, "article-7-right-to-compensation"),
        "A8": Ref(doc, "article-8-right-to-reimbursement"),
        "A9": Ref(doc, "article-9-right-to-care"),
        "A10": Ref(doc, "article-10-upgrading"),
        "A14": Ref(doc, "article-14-obligation"),
        "A15": Ref(doc, "article-15-exclusion"),
        "A16": Ref(doc, "article-16-infringements"),
        "REC14": Ref(doc, "preamble", has="unexpected flight safety shortcomings"),
    }


_eu, _uk = _articles(EU), _articles(UK)

# ---- EU261 -----------------------------------------------------------------
EU_A2, EU_A3, EU_A4, EU_A5 = _eu["A2"], _eu["A3"], _eu["A4"], _eu["A5"]
EU_A6, EU_A7, EU_A8, EU_A9 = _eu["A6"], _eu["A7"], _eu["A8"], _eu["A9"]
EU_A10, EU_A14, EU_A15, EU_A16 = _eu["A10"], _eu["A14"], _eu["A15"], _eu["A16"]
EU_REC14 = _eu["REC14"]

# ---- UK261 -----------------------------------------------------------------
UK_A3, UK_A4, UK_A5 = _uk["A3"], _uk["A4"], _uk["A5"]
UK_A6, UK_A7, UK_A8, UK_A9 = _uk["A6"], _uk["A7"], _uk["A8"], _uk["A9"]
UK_A10, UK_A14, UK_A15, UK_A16 = _uk["A10"], _uk["A14"], _uk["A15"], _uk["A16"]
UK_REC14 = _uk["REC14"]

# ---- European Commission interpretative guidelines (C/2024/5687) -----------
# Session 22: the source for the CJEU case law the Regulation's text lacks.
# Section prefixes end in "-" where a shorter number would also match
# ("4-4-1-" must not select 4.4.10-4.4.14).
EUG = "eu-261-guidelines"
EUG_LONG_DELAY = Ref(EUG, "4-4-7-", has="with a delay of 3 hours or more are entitled to the same compensation")
EUG_CONNECTION = Ref(EUG, "4-4-8-", has="destination of the last flight taken by the passenger")
EUG_DELAY_AMOUNT = Ref(EUG, "4-4-10-", has="amounts to EUR 300")
EUG_REROUTE_EXEMPT = Ref(EUG, "4-4-12-", has="less than 7 days before the scheduled departure")
EUG_EXTRAORDINARY = Ref(EUG, "5-1-", has="must simultaneously prove")
EUG_CREW_ABSENCE = Ref(EUG, "5-2-2-", has="Unexpected absence of crew members")
EUG_MEASURE_ARRIVAL = Ref(EUG, "3-3-3-")

# ---- UK Civil Aviation Authority guidance ----------------------------------
CAA = "uk-caa-guidance"
CAA_DELAY_COMP = Ref(CAA, "compensation", has="more than three hours late at your destination airport")
CAA_DELAY_AMOUNTS = Ref(CAA, "how-much-compensation", has="between three and four hours late")
CAA_CARE = Ref(CAA, "what-care-should-i-receive")
CAA_CANCEL_7_14 = Ref(CAA, "seven-to-14-days-notice")
CAA_CANCEL_7_14_EXC = Ref(CAA, "circumstances-when-the-above-table")
CAA_CANCEL_LT7 = Ref(CAA, "less-than-seven-days-notice")
CAA_CANCEL_COMP = Ref(CAA, "compensation-2", has="less than 14 days")

# ---- 14 CFR Part 260 (refunds) ---------------------------------------------
P260_DEF_DELAY = Ref(P260, "260-2", has="six or more hours for international itineraries after the original scheduled arrival")
P260_COVERED = Ref(P260, "260-2", has="Covered flight means")
P260_PROMPT = Ref(P260, "260-2", has="Prompt refund means")
P260_BAG_DEF = Ref(P260, "260-2", has="Significantly delayed checked bag means")
P260_REFUND = Ref(P260, "260-6", has="holds a nonrefundable ticket on a scheduled flight")
P260_ALT = Ref(P260, "260-6", has="Alternative to refund")
P260_ANCILLARY = Ref(P260, "260-4", has="prompt and automatic refund to a consumer for any fees")
P260_BAG = Ref(P260, "260-5", has="fee charged for transporting a lost bag")
P260_ACCEPT = Ref(P260, "260-7")
P260_10 = Ref(P260, "260-10")

# ---- 14 CFR Part 250 (oversales) -------------------------------------------
P250_APPL = Ref(P250, "250-2-applicability")
P250_AMT_DOM = Ref(P250, "250-5", has="shall pay compensation in interstate air transportation")
P250_AMT_INTL = Ref(P250, "250-5", has="more than one hour but less than four hours after the")
P250_EXC = Ref(P250, "250-6", has="substitution of equipment of lesser capacity")
P250_PAY = Ref(P250, "250-8", has="within 24 hours after the time the denied boarding occurs")
P250_NOTICE_EXC = Ref(P250, "250-9", has="you have not fully complied with the airline")
P250_NOTICE_DOM = Ref(P250, "250-9", has="1 to 2 hour arrival delay")
P250_NOTICE_INTL = Ref(P250, "250-9", has="1 to 4 hour arrival delay")
P250_METHOD = Ref(P250, "250-9", has="may insist on the cash/check payment")
P250_ANCILLARY = Ref(P250, "250-6", has="refund all unused ancillary fees")
P250_TERRITORIES = Ref(P250, "250-9", has="including the territories and possessions")

# ---- 14 CFR Part 259 (enhanced protections) --------------------------------
P259_APPL = Ref(P259, "259-2")
P259_DEF_TARMAC = Ref(P259, "259-3", has="Tarmac delay means")
P259_TARMAC_DEPLANE = Ref(P259, "259-4", has="before the tarmac delay exceeds three hours")
P259_TARMAC_FOOD = Ref(P259, "259-4", has="adequate food and potable water no later than two hours")
P259_TARMAC_UPDATES = Ref(P259, "259-4", has="notify the passengers on board the aircraft during a tarmac")
P259_CSP_ADHERE = Ref(P259, "259-5", has="adhere to the plan")
P259_CSP_CONTENTS = Ref(P259, "259-5", has="Meeting customers")
P259_CSP_NOTIFY = Ref(P259, "259-5", has="Notifying consumers of known delays")
P259_COMPLAINT = Ref(P259, "259-7", has="within 30 days of receiving it")

# ---- DOT refunds Q&A -------------------------------------------------------
DOTQA_REFUND = Ref(DOTQA, "am-i-entitled-to-a-refund-of-the-ticket-price", has="regardless of the reason")
DOTQA_SIGNIFICANT = Ref(DOTQA, "am-i-entitled-to-a-refund-of-the-ticket-price", has="Late arrival")
DOTQA_TRAVELED = Ref(DOTQA, "am-i-entitled-to-a-refund-of-the-ticket-price", has="Consumer Traveled")
DOTQA_ANCILLARY = Ref(DOTQA, "am-i-entitled-to-a-refund-of-fees-related-to-ancillary")
DOTQA_BAG = Ref(DOTQA, "am-i-entitled-to-a-refund-of-fees-related-to-checked", has="within 12 hours")
DOTQA_REBOOK = Ref(DOTQA, "what-if-the-airline-offers-to-rebook")
DOTQA_VOUCHER = Ref(DOTQA, "what-if-the-airline-offers-me-travel-credits")
DOTQA_AUTO = Ref(DOTQA, "what-should-i-do")

# ---- American --------------------------------------------------------------
AA_COC_SCHED = Ref(AA_COC, "our-responsibilities-when-there-are-schedule")
AA_COC_FM = Ref(AA_COC, "events-beyond-our-control")
AA_COC_VDB = Ref(AA_COC, "voluntary-denied-boarding")
AA_COC_IDB = Ref(AA_COC, "involuntary-denied-boarding")
AA_COC_DBC = Ref(AA_COC, "compensation-for-involuntary-denied-boarding")
AA_COC_REBOOK = Ref(AA_COC, "rebooking-your-delayed")
AA_COC_BYUS = Ref(AA_COC, "delays-caused-by-us")
AA_COC_WX = Ref(AA_COC, "delays-beyond-our-control")
AA_COC_NONREF = Ref(AA_COC, "non-refundable-tickets")
AA_COC_INVREF = Ref(AA_COC, "involuntary-refunds")
AA_CSP_BYUS = Ref(AA_CSP, "delays-and-cancellations-caused-by-us")
AA_CSP_WX = Ref(AA_CSP, "delays-beyond-our-control")
AA_CSP_DIV = Ref(AA_CSP, "diversions")
AA_CSP_TARMAC = Ref(AA_CSP, "essential-customer-needs")
AA_CSP_CC_REFUND = Ref(AA_CSP, "refunds-to-a-credit-card")

# ---- United ----------------------------------------------------------------
UA_CC_ASSIST = Ref(UA_CC, "providing-you-assistance", has="meal voucher")
UA_CC_REFUND = Ref(UA_CC, "provide-prompt-refunds")
UA_CC_REFUND_MORE = Ref(UA_CC, "flights-where-a-customer-is-downgraded")
UA_CC_CREDITS = Ref(UA_CC, "provide-flight-credits")
UA_CC_TARMAC = Ref(UA_CC, "meet-customers-essential-needs")
UA_CC_OVERSALE = Ref(UA_CC, "treat-customers-fairly", has="oversale")
UA_COC_NONUS = Ref(UA_COC, "rule-24", has="Non-U.S.A. Origin Flights")
UA_COC_FM = Ref(UA_COC, "rule-24", has="Any shortage of labor, fuel, or facilities")
UA_COC_IRROPS = Ref(UA_COC, "rule-24", has="Irregular Operations caused by UA")
UA_COC_LODGING = Ref(UA_COC, "rule-24", has="Lodging - UA will provide")
UA_COC_ANCILLARY = Ref(UA_COC, "rule-24", has="within 90 days")
UA_COC_R25_ORIGIN = Ref(UA_COC, "rule-25", has="Canadian Flight Origin")
UA_COC_DBC = Ref(UA_COC, "rule-25", has="interstate transportation")
UA_COC_DBC_EXC = Ref(UA_COC, "rule-25", has="not later than 60 minutes")
UA_COC_NOSHOW = Ref(UA_COC, has="Failure to Occupy Space")
UA_COC_DISPUTE = Ref(UA_COC, has="notify United of any dispute")

# ---- Delta -----------------------------------------------------------------
DL_POL_TYPES = Ref(DL_POL, "am-i-experiencing")
DL_POL_DELAY = Ref(DL_POL, "managing-a-delay")
DL_POL_CANCEL = Ref(DL_POL, "managing-a-canceled")
DL_POL_REFUND = Ref(DL_POL, "requesting-a-refund")
DL_POL_OTHER = Ref(DL_POL, "other-refund")
DL_POL_RESTRICT = Ref(DL_POL, "refund-restrictions")
DL_POL_REIMB = Ref(DL_POL, "requesting-reimbursement")
DL_POL_ACCOM = Ref(DL_POL, "requesting-accommodations")
DL_DOM_R19_REFUND = Ref(DL_DOM, has="arrive 180 minutes or more after")
DL_DOM_R19_AMENITIES = Ref(DL_DOM, has="interrupted for more than 4 hours after the scheduled")
DL_DOM_R20_DBC = Ref(DL_DOM, has="but no more than $1,075.00")
DL_INTL_R20_REFUND = Ref(DL_INTL, has="arrive 360 minutes or more after")
DL_INTL_R20_AMENITIES = Ref(DL_INTL, has="interrupted for more than 4 hours after the scheduled")
DL_INTL_R21_DBC = Ref(DL_INTL, has="but no more than $1,075.00")

# ---- Southwest -------------------------------------------------------------
WN_COC_DEF = Ref(WN_COC, "1-introduction", has="Significantly Delayed or Changed Flight means")
WN_COC_CANCEL = Ref(WN_COC, "9-flight-changes-cancellations-delays-and-diversions", has="In the event Southwest cancels a flight")
WN_COC_EXCLUSIVE = Ref(WN_COC, "9-flight-changes-cancellations-delays-and-diversions", has="exclusive remedy")
WN_COC_DBC = Ref(WN_COC, "9-flight-changes-cancellations-delays-and-diversions", has="Compensation shall be at least two hundred percent")
WN_COC_DBC_FORM = Ref(WN_COC, "9-flight-changes-cancellations-delays-and-diversions", has="insist on receiving compensation by")
WN_CSP_NOTIFY = Ref(WN_CSP, "2-notifying")
WN_CSP_REFUND = Ref(WN_CSP, "5-when-a-refund")
WN_CSP_TARMAC = Ref(WN_CSP, "9-meeting")
WN_CSP_BUMP = Ref(WN_CSP, "10-handling")
WN_CSP_CONTROL = Ref(WN_CSP, "14-", has="within our control or Southwest-initiated cancellations")
WN_CSP_VOUCHER = Ref(WN_CSP, "14-", has="LUV Voucher (of at least $75)")

# Scope pages the regulators publish beside the regulation: an answer may cite
# either to say which law applies (golden/required.py, Session 24).
CAA_SCOPE = Ref(CAA, "does-uk-law-apply")
EUG_SCOPE = Ref(EUG, "2-1-1-")

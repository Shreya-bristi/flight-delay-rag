"""
Required evidence per answerable case (Session 23, revised Session 24 after the
user's review): the PREMISES an answer cannot be right without, not particular
section ids.

THREE NUMBERS, THREE QUESTIONS (evals/retrieval_eval.py)
    required_premise_complete    every premise below proved, by any accepted source
    primary_authority_coverage   ... and how often the binding regulation itself was
                                 there, rather than only guidance repeating it
    evidence_recall              how much of all the useful gold text was retrieved

This list is a maintainer draft reviewed once by the user (2026-09-19); the legal
judgement of what an answer needs is the user's, not the harness's.

live-05 is expected to fail: its only source for "territories count as the US" is
Part 250's denied-boarding notice, which the topic filter removes from a
cancellation question. It stays until a neutral source is added (PROGRESS.md).
"""

from __future__ import annotations

from . import refs as R
from .refs import Ref


def premise(*refs: Ref) -> tuple[Ref, ...]:
    """One premise, proved by ANY of these sources."""
    return refs


# Equivalent statements of the same rule, used across cases.
US_SIGNIFICANT_DELAY = premise(R.P260_DEF_DELAY, R.DOTQA_SIGNIFICANT)
US_REFUND = premise(R.P260_REFUND, R.DOTQA_REFUND)
UK_SCOPE = premise(R.UK_A3, R.CAA_SCOPE)
EU_SCOPE = premise(R.EU_A3, R.EUG_SCOPE)
UK_DELAY_TRIGGER = premise(R.UK_A6, R.CAA_DELAY_COMP)
UK_DELAY_AMOUNT = premise(R.UK_A7, R.CAA_DELAY_AMOUNTS)
UK_CANCEL_COMPENSATION = premise(R.UK_A5, R.CAA_CANCEL_COMP, R.CAA_CANCEL_7_14, R.CAA_CANCEL_LT7)
EU_DELAY_AMOUNT = premise(R.EU_A7, R.EUG_DELAY_AMOUNT)

REQUIRED: dict[str, list[tuple[Ref, ...]]] = {
    # ---- US domestic
    "dom-01": [US_SIGNIFICANT_DELAY, premise(R.AA_CSP_BYUS)],
    "dom-04": [US_SIGNIFICANT_DELAY, premise(R.WN_CSP_VOUCHER)],
    "dom-07": [US_SIGNIFICANT_DELAY, premise(R.DL_DOM_R19_AMENITIES)],
    "dom-13": [US_SIGNIFICANT_DELAY],
    "dom-15": [US_REFUND, premise(R.AA_CSP_BYUS)],
    "dom-17": [US_REFUND, premise(R.P260_ACCEPT)],
    # "Do I have to accept the credit?" - the affirmative-acceptance rule is the answer.
    "dom-21": [US_REFUND, premise(R.P260_ALT, R.DOTQA_VOUCHER), premise(R.P260_ACCEPT),
               premise(R.UA_CC_CREDITS)],
    "dom-24": [premise(R.P260_PROMPT)],
    "dom-25": [premise(R.P250_AMT_DOM), premise(R.P250_PAY)],
    "dom-29": [premise(R.P250_EXC)],
    "dom-31": [premise(R.P250_NOTICE_EXC)],
    "dom-34": [premise(R.P259_TARMAC_FOOD)],
    "dom-36": [US_SIGNIFICANT_DELAY, premise(R.DL_DOM_R19_REFUND)],
    "dom-37": [premise(R.AA_CSP_DIV)],
    "dom-44": [premise(R.P259_COMPLAINT)],
    # ---- UK -> US
    "uk-us-01": [UK_SCOPE, UK_DELAY_TRIGGER, UK_DELAY_AMOUNT],
    # Art 6 carries both the care thresholds and 6(4), the extraordinary-circumstances exemption.
    "uk-us-03": [premise(R.UK_A6), premise(R.UK_A9, R.CAA_CARE)],
    "uk-us-06": [UK_CANCEL_COMPENSATION, premise(R.UK_A8)],
    "uk-us-09": [UK_SCOPE, UK_DELAY_TRIGGER, UK_DELAY_AMOUNT],
    # Scope is the question itself (award tickets, Art 3(3)), so guidance does not stand in.
    "uk-us-14": [premise(R.UK_A3), UK_CANCEL_COMPENSATION, premise(R.UK_A8)],
    # ---- EU -> US
    "eu-us-01": [EU_SCOPE, premise(R.EUG_LONG_DELAY), EU_DELAY_AMOUNT],
    "eu-us-05": [premise(R.EU_A5, R.EUG_REROUTE_EXEMPT), premise(R.EU_A7)],
    "eu-us-07": [premise(R.EU_A4), premise(R.EU_A7)],
    "eu-us-12": [EU_SCOPE],
    "eu-us-13": [EU_SCOPE, premise(R.EUG_CONNECTION), EU_DELAY_AMOUNT],
    # ---- US -> Europe
    "us-eur-01": [EU_SCOPE, US_SIGNIFICANT_DELAY],
    "us-eur-02": [UK_SCOPE, US_SIGNIFICANT_DELAY],
    "us-eur-04": [premise(R.P250_AMT_INTL)],
    "us-eur-06": [premise(R.P259_TARMAC_DEPLANE)],
    "us-eur-09": [EU_SCOPE, premise(R.EUG_DELAY_AMOUNT)],
    # ---- intra-Europe
    "intra-04": [EU_SCOPE, premise(R.EU_A5, R.EUG_REROUTE_EXEMPT), premise(R.EU_A7),
                 premise(R.EU_A8)],
    "intra-07": [premise(R.P260_COVERED), UK_SCOPE, UK_CANCEL_COMPENSATION, premise(R.UK_A8)],
    # ---- law vs airline
    "law-02": [premise(R.AA_COC_BYUS), premise(R.P259_CSP_ADHERE)],
    "law-03": [premise(R.EU_A15), premise(R.EUG_EXTRAORDINARY, R.EUG_CREW_ABSENCE)],
    "law-07": [premise(R.P260_REFUND, R.DOTQA_TRAVELED, R.P260_DEF_DELAY, R.DOTQA_SIGNIFICANT),
               premise(R.EU_A7, R.EUG_LONG_DELAY), UK_DELAY_AMOUNT],
    # ---- live flight data
    "live-01": [UK_SCOPE, UK_DELAY_TRIGGER, UK_DELAY_AMOUNT],
    "live-02": [EU_SCOPE, US_SIGNIFICANT_DELAY],
    "live-03": [US_SIGNIFICANT_DELAY],
    "live-04": [premise(R.WN_CSP_VOUCHER)],
    "live-05": [premise(R.P250_TERRITORIES), US_REFUND],
    "live-06": [EU_SCOPE, US_REFUND],
    # ---- other
    "other-01": [premise(R.P260_COVERED), EU_SCOPE],
    "unsup-01": [UK_SCOPE, UK_CANCEL_COMPENSATION, premise(R.UK_A8)],
}

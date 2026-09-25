"""
Ambiguous direction, law-vs-airline, live-data (hybrid), edge and trap cases.

Ambiguous route (amb-01..03, amb-06): the question does not settle where the
flight departed - or, for amb-06, which "Manchester" it left from - and that
missing detail decides which regime governs, so there is no correct answer to
give yet. These cases are `clarify=True, answerable=False` and carry
`expected_behavior.action == "clarify"`: expected_output is production's own
clarifying question, not a drafted answer that guesses or covers both
directions, and `expected_clarification` holds that question.
`expected_facts`/`forbidden_claims` describe what a correct FINAL answer would
need once the passenger replies. `reply` is still scripted where one exists, so
the builder can check the route the reply settles on against the route tag.


"""

from __future__ import annotations

from . import refs as R
from .spec import case

TRAP = dict(route="not_applicable", disruption="None", control="N/A", category="trap", answerable=False)

CASES = [
    # ======================================================================
    # Ambiguous direction -> production asks, doesn't get a resolving reply
    # ======================================================================
    case(
        "amb-01",
        "My United Paris flight was cancelled. What am I owed?",
        [R.EU_A3, R.EU_A5, R.EU_A7, R.P260_REFUND, R.UA_CC_ASSIST],
        "",
        route="unknown_direction", carrier="United", disruption="Cancellation", control="Unknown",
        category="ambiguous_direction", answerable=False, clarify=True,
        expected_facts=[
            "The departure airport is unknown, and it decides whether EU261 or only US rules apply, so "
            "the assistant must ask before answering.",
            "The compensation amount (€600 or none) is entirely different depending on direction, so "
            "guessing would risk telling the passenger the wrong entitlement.",
        ],
        forbidden_claims=[
            "The assistant assumes a direction and gives a definitive compensation amount without "
            "asking.",
            "The assistant claims EU261 compensation applies without knowing the departure airport.",
        ],
    ),
    case(
        "amb-02",
        "My Delta London flight is delayed 5 hours. Do I get compensation?",
        [R.UK_A3, R.UK_A6, R.UK_A7, R.P260_DEF_DELAY, R.DL_INTL_R20_AMENITIES],
        "",
        route="unknown_direction", carrier="Delta", disruption="Delay", control="Unknown",
        category="ambiguous_direction", answerable=False, clarify=True,
        expected_facts=[
            "Whether UK261 applies depends on which airport is the departure, which the question doesn't "
            "state.",
            "The assistant should ask which airport the passenger is departing from before giving a "
            "compensation figure.",
        ],
        forbidden_claims=[
            "The assistant states a specific compensation amount without knowing the direction of "
            "travel.",
        ],
    ),
    case(
        "amb-03",
        "American denied me boarding on the Dublin–Chicago flight. What compensation do I get?",
        [R.EU_A4, R.EU_A7, R.P250_APPL, R.P250_AMT_INTL, R.AA_COC_DBC],
        "",
        route="unknown_direction", carrier="American", disruption="Denied Boarding", control="N/A",
        category="ambiguous_direction", answerable=False, clarify=True,
        notes="Session 22: an outside audit read the question as 'Dublin -> Chicago' and called this "
              "answerable. The question has an en dash, and a hyphen/dash is not a direction (user's "
              "rule, CLAUDE.md), so it stays a clarify case.",
        expected_facts=[
            "Compensation depends entirely on whether the flight departed Dublin (EU261) or Chicago (the "
            "US oversales rule), which isn't stated.",
            "The two regimes set very different amounts, so the assistant must ask which airport was the "
            "departure before answering.",
        ],
        forbidden_claims=[
            "The assistant picks one direction and states its compensation amount as the answer.",
        ],
    ),
    case(
        "amb-06",
        "My Delta flight from Manchester to Atlanta was cancelled at the airport. What am I owed?",
        [R.UK_A3, R.UK_A5, R.UK_A7, R.UK_A8, R.UK_A9, R.P260_REFUND, R.DL_POL_CANCEL],
        "",
        route="UK_to_US", carrier="Delta", disruption="Cancellation", control="Unknown", origin="MAN",
        dest="ATL", answerable=False, clarify=True,
        reply="The Manchester in England. I don't have the flight number.",
        notes="Manchester is ambiguous (UK or New Hampshire): production asks which one, and the reply "
              "settles it. Replaces the old 'Manchester to Newark' regression (uk-us-09 was reworded to "
              "'Manchester, England' so it no longer exercises the question). A clarify case, not a "
              "clarify-then-answer one: which Manchester decides UK261 (GBP 520 + care) vs US DOT alone "
              "(no cash compensation), so no answer is correct before the passenger says which. The "
              "drafted final answer, kept here for whoever reviews the reply-settled turn: 'Your flight "
              "departed the UK, so UK261 applies alongside US DOT rules. For this cancellation you get a "
              "choice of a refund within seven days or re-routing to Atlanta at the earliest "
              "opportunity, care while you wait (meals, calls, and a hotel if the new flight is the next "
              "day), and 520 pounds compensation for this over-3,500 km route, unless Delta proves "
              "extraordinary circumstances. US rules also entitle you to a full refund if you decline "
              "rebooking or a credit. Claim the compensation from Delta in writing.'",
        expected_facts=[
            "\"Manchester\" is ambiguous (Manchester, UK or Manchester, New Hampshire), and which one it "
            "is decides the governing regime, so the assistant must ask before answering.",
            "Once the reply settles Manchester as the UK city, UK261 governs and gives a "
            "refund/re-routing choice and care; compensation of up to £520 for this over-3,500km route "
            "depends on the re-routing offered after a same-day cancellation and on extraordinary "
            "circumstances.",
            "US DOT's refund right also applies if the passenger declines rebooking or credit.",
        ],
        forbidden_claims=[
            "The assistant picks one Manchester and states an entitlement without asking.",
            "The assistant answers as if Manchester, New Hampshire were the departure.",
            "No compensation is owed because the cancellation happened at the airport, not in advance.",
        ],
    ),

    # ======================================================================
    # Law vs airline promise
    # ======================================================================
    case(
        "law-02",
        "American's Conditions of Carriage say that for a 3-hour delay their 'sole obligation' is a refund. "
        "Does that mean they won't give me a hotel when the delay is their fault?",
        [R.AA_COC_SCHED, R.AA_COC_BYUS, R.AA_CSP_BYUS, R.P259_CSP_ADHERE],
        """
        No - read together, American still owes you more than a refund when the disruption is its fault.
        The 'sole obligation is a refund' line applies to the base contract remedy, but American's
        Conditions of Carriage separately promise a hotel if you won't board before midnight on a
        disruption it caused, and its Customer Service Plan adds a food credit and transport too. Federal
        rules require American to actually follow its published Customer Service Plan. Confirm the delay
        is coded as American's fault, then ask for the hotel voucher.
        """,
        route="US_to_US", carrier="American", disruption="Delay", control="Controllable",
        category="law_vs_airline",
        expected_facts=[
            "American's 'sole obligation is a refund' clause does not override its separate hotel "
            "commitment for disruptions American caused.",
            "Federal rules require an airline to follow the terms of its own published Customer Service "
            "Plan.",
            "The hotel commitment applies only if the disruption is confirmed as American's fault.",
        ],
        forbidden_claims=[
            "The 'sole obligation is a refund' clause means American owes nothing else, even when it "
            "caused the delay.",
        ],
    ),
    case(
        "law-03",
        "United's contract lists 'shortage of labor' as a force majeure event with no liability. My United "
        "flight from Paris to Newark was cancelled for a crew shortage. Does that cancel my EU261 rights?",
        [R.EU_A15, R.EU_A5, R.EU_A8, R.EU_A9, R.EUG_EXTRAORDINARY, R.EUG_CREW_ABSENCE, R.UA_COC_FM,
         R.UA_COC_NONUS],
        """
        No - a contract clause can't take away your EU261 rights. Even though United's contract lists
        crew shortages as 'force majeure' with no liability, EU261 only excuses compensation if United
        proves the cancellation was a genuine extraordinary circumstance it couldn't have avoided, and an
        ordinary crew shortage isn't automatically one. You still get a refund or re-routing choice and
        care while you wait. Compensation can be up to 600 euros for this route, but it also depends on
        when United told you and what re-routing it offered. Ask United for the actual reason in writing
        and claim accordingly.
        """,
        route="EU_to_US", carrier="United", disruption="Cancellation", control="Controllable", origin="CDG",
        dest="EWR", category="law_vs_airline",
        expected_facts=[
            "EU261 rights cannot be waived or limited by a clause in the airline's own contract of "
            "carriage.",
            "United's 'force majeure' label for a crew shortage does not by itself meet the legal "
            "extraordinary-circumstances test EU261 requires.",
            "The passenger keeps the refund/re-routing choice and care regardless of the contract "
            "clause; compensation of up to €600 also depends on the notice and re-routing offered.",
        ],
        forbidden_claims=[
            "United's contract clause is legally sufficient to cancel the passenger's EU261 rights.",
            "€600 is owed regardless of when the passenger was told and what re-routing was offered.",
        ],
    ),
    case(
        "law-07",
        "How is compensation for a long flight delay different in the US versus the EU and UK?",
        [R.P260_REFUND, R.DOTQA_TRAVELED, R.EU_A3, R.EU_A6, R.EU_A7, R.EUG_LONG_DELAY, R.UK_A3,
         R.UK_A6, R.UK_A7, R.CAA_DELAY_AMOUNTS],
        """
        They're quite different. The US has no fixed cash payout for ordinary delays - a significant
        delay gives you a refund only if you decide not to fly. The EU and UK set fixed compensation by
        distance once you arrive 3 or more hours late: 250, 400 or 600 euros under EU261 and 220, 350 or
        520 pounds under UK261 (on the longest flights, half that for an arrival 3 to 4 hours late), on
        top of meals, and a hotel for an overnight delay. Which rules cover you depends on the route
        and, for a flight arriving in the EU or UK from elsewhere, on whether the airline is an EU or UK
        carrier.
        """,
        route="not_applicable", carrier="Unspecified", disruption="Delay", control="N/A",
        category="law_vs_airline",
        expected_facts=[
            "US rules set no fixed cash compensation for delays; only a refund if the passenger chooses "
            "not to travel.",
            "UK261 and EU261 both set fixed compensation (by distance) for arrival delays of 3+ hours, "
            "unlike the US: €250/€400/€600 under EU261, £220/£350/£520 under UK261.",
            "Coverage depends on the route and, for flights into the EU or UK, on the operating "
            "airline.",
        ],
        forbidden_claims=[
            "US rules provide the same fixed cash compensation as the EU or UK.",
            "EU261 compensation is roughly €220-€520.",
            "Which rules apply never depends on the airline.",
        ],
    ),

    # ======================================================================
    # Hybrid: answer depends on live flight data (synthetic fixtures)
    # ======================================================================
    case(
        "live-01",
        "Flight UA 9901 is showing a big delay. What am I owed?",
        [R.UK_A3, R.UK_A6, R.UK_A7, R.UK_A8, R.UK_A9, R.CAA_DELAY_AMOUNTS, R.UA_CC_ASSIST,
         R.P260_DEF_DELAY],
        """
        Your flight data shows UA9901 about 5 hours late leaving Heathrow for Chicago and about 4 hours
        50 minutes late arriving. Because it departed the UK, UK261 applies: you're owed meals and
        communication during the wait and the option of a refund instead of flying. If you still arrive
        more than 4 hours late, compensation is 520 pounds; if the delay shrinks to between 3 and 4
        hours, it's 260 pounds - unless United proves extraordinary circumstances. US rules don't add
        anything here since their international delay-refund threshold is 6 hours. Ask United to
        confirm the cause, since that affects the compensation.
        """,
        route="UK_to_US", carrier="United", disruption="Delay", control="Unknown", origin="LHR", dest="ORD",
        category="hybrid_live_data",
        fixture={"flight_iata": "UA9901", "airline_iata": "UA", "status": "active", "dep_iata": "LHR",
                 "arr_iata": "ORD", "dep_delayed": 300, "arr_delayed": 290, "synthetic": True},
        expected_facts=[
            "The flight record shows a departure delay of about 5 hours from London Heathrow to Chicago, "
            "so UK261 applies alongside US rules.",
            "At 5 hours the passenger can choose a refund instead of flying; compensation is £520 if "
            "the actual arrival is more than 4 hours late (£260 for 3-4 hours), unless United proves "
            "extraordinary circumstances.",
            "The US refund threshold for an international delay is 6 hours, which this doesn't reach.",
        ],
        forbidden_claims=[
            "US DOT rules provide compensation for this delay.",
            "£520 is owed for any arrival 3+ hours late.",
        ],
    ),
    case(
        "live-02",
        "Is DL 9902 delayed, and do I get anything?",
        [R.EU_A3, R.P260_DEF_DELAY, R.DL_INTL_R20_AMENITIES, R.DL_POL_REIMB],
        """
        Yes, DL9902 shows about a 4.5-hour delay from New York to Paris. Since it departs the US, not the
        EU, EU261 doesn't apply here - it would cover your return flight. Under US rules there's no fixed
        delay compensation, and the refund threshold for an international flight is 6 hours, which this
        doesn't reach. Ask Delta about its own meal or hotel provisions if the delay turns out to be
        within its control.
        """,
        route="US_to_EU", carrier="Delta", disruption="Delay", control="Unknown", origin="JFK", dest="CDG",
        category="hybrid_live_data",
        fixture={"flight_iata": "DL9902", "airline_iata": "DL", "status": "scheduled", "dep_iata": "JFK",
                 "arr_iata": "CDG", "dep_delayed": 270, "arr_delayed": 260, "synthetic": True},
        expected_facts=[
            "The flight record shows a departure delay of about 4.5 hours from New York to Paris, a US "
            "departure, so EU261 does not apply.",
            "US rules set no fixed delay compensation, and the international refund threshold is 6 "
            "hours, which 4.5 hours doesn't reach.",
        ],
        forbidden_claims=[
            "EU261 compensation applies to this US-departing flight.",
        ],
    ),
    case(
        "live-03",
        "I'm on AA 9903. What are my options?",
        [R.P260_DEF_DELAY, R.P260_REFUND, R.AA_CSP_BYUS, R.AA_CSP_WX],
        """
        AA9903 is showing about a 3-hour-20-minute delay, which meets the threshold for a 'significantly
        delayed' domestic flight. If you'd rather not fly, you can decline the rebooking and get a full
        refund of your fare and fees. If American caused the delay, ask about a food credit and, if you're
        stuck overnight, a hotel voucher - those depend on confirming the cause is on American, not
        weather.
        """,
        route="US_to_US", carrier="American", disruption="Delay", control="Unknown", origin="ORD", dest="DFW",
        category="hybrid_live_data",
        fixture={"flight_iata": "AA9903", "airline_iata": "AA", "status": "scheduled", "dep_iata": "ORD",
                 "arr_iata": "DFW", "dep_delayed": 200, "arr_delayed": 195, "synthetic": True},
        expected_facts=[
            "The flight record shows roughly a 3-hour-20-minute delay on a US domestic flight, meeting "
            "the 3-hour significantly-delayed threshold.",
            "A refund is available if the passenger declines to fly; if American caused the delay, meal "
            "and hotel help may apply.",
        ],
        forbidden_claims=[
            "No refund option exists because the ticket is non-refundable.",
        ],
    ),
    case(
        "live-04",
        "WN 9904 is running late. Can I get a voucher?",
        [R.WN_CSP_CONTROL, R.WN_CSP_VOUCHER, R.P260_DEF_DELAY, R.WN_COC_CANCEL],
        """
        WN9904 is running about 3 hours late, just over the significant-delay threshold. Whether you get a
        LUV Voucher (at least $75) or just a rebooking depends on the cause: if the delay is within
        Southwest's control, like a mechanical issue, you likely qualify - file at
        Southwest.com/DelayForm within a year. If it's weather or air traffic control related, Southwest
        will only rebook you at no cost.
        """,
        route="US_to_US", carrier="Southwest", disruption="Delay", control="Unknown", origin="DEN", dest="PHX",
        category="hybrid_live_data",
        fixture={"flight_iata": "WN9904", "airline_iata": "WN", "status": "scheduled", "dep_iata": "DEN",
                 "arr_iata": "PHX", "dep_delayed": 190, "arr_delayed": 185, "synthetic": True},
        expected_facts=[
            "The flight record shows an arrival delay just over 3 hours on a US domestic flight.",
            "A voucher (LUV Voucher, at least $75) applies only if the delay is within Southwest's "
            "control; if it's weather or ATC-related, only rebooking applies.",
        ],
        forbidden_claims=[
            "The LUV Voucher is guaranteed regardless of the cause of the delay.",
        ],
    ),
    case(
        "live-05",
        "My flight UA 9905 was cancelled. Can I get my money back?",
        [R.P250_TERRITORIES, R.P260_COVERED, R.P260_REFUND, R.DOTQA_REFUND, R.UA_CC_REFUND, R.UA_CC_ASSIST],
        """
        Yes. Guam and Saipan both count as part of the United States under federal aviation rules, so this
        cancelled flight is treated the same as any domestic US cancellation. You can decline United's
        rebooking and get a full refund of your fare and fees, even on a non-refundable ticket. If you'd
        rather fly, United will rebook you on its next available flight at no cost.
        """,
        route="US_to_US", carrier="United", disruption="Cancellation", control="Unknown", origin="GUM",
        dest="SPN", category="hybrid_live_data", edge="us_territory",
        fixture={"flight_iata": "UA9905", "airline_iata": "UA", "status": "cancelled", "dep_iata": "GUM",
                 "arr_iata": "SPN", "synthetic": True},
        notes="Before 2026-09-14 airport_regions.json classed GUM and SPN as OTHER, so this flight got no US "
              "DOT rules at all (OTHER -> OTHER). The territory rule is cited from 14 CFR 250.9's passenger "
              "notice; Parts 259/260 do not define 'United States' in the corpus.",
        expected_facts=[
            "Guam and Saipan are both US territories, so this flight is treated as domestic under US "
            "rules even though it's outside the mainland.",
            "A cancellation entitles the passenger to a full refund if rebooking is declined.",
        ],
        forbidden_claims=[
            "US rules don't apply because Guam and Saipan aren't in the mainland United States.",
        ],
    ),
    case(
        "live-06",
        "My United Paris flight UA 9906 was cancelled. Do I get the €600 EU261 compensation?",
        [R.EU_A3, R.P260_REFUND, R.UA_CC_REFUND, R.UA_CC_ASSIST],
        """
        No 600 euros here - your flight data shows UA9906 actually departs Newark for Paris, not the
        other way around, and EU261 only applies to EU departures. Since this is a US-departing flight,
        US rules apply instead: you can decline the rebooking and get a full refund of your fare and
        fees, but there's no fixed cash compensation. If you'd rather travel, United will rebook you on
        its next available flight.
        """,
        route="US_to_EU", carrier="United", disruption="Cancellation", control="Unknown", origin="EWR",
        dest="CDG", category="hybrid_live_data", edge="inapplicable_regime",
        fixture={"flight_iata": "UA9906", "airline_iata": "UA", "status": "cancelled", "dep_iata": "EWR",
                 "arr_iata": "CDG", "synthetic": True},
        notes="'My Paris flight' in the text would ask which direction; the flight record (EWR -> CDG) "
              "overrides the text and settles it without a question.",
        expected_facts=[
            "The flight record shows this flight departs Newark, not Paris, so EU261 does not apply even "
            "though the passenger called it 'my Paris flight'.",
            "US rules apply instead: a full refund if rebooking is declined, but no fixed cash "
            "compensation.",
        ],
        forbidden_claims=[
            "The €600 EU261 compensation applies because the destination is Paris.",
        ],
    ),

    # ======================================================================
    # Edge: OTHER region and unsupported airlines
    # ======================================================================
    case(
        "other-01",
        "My Delta flight from Cancun to Amsterdam was cancelled. Do the US DOT refund rules or EU261 apply?",
        [R.P260_COVERED, R.EU_A3, R.DL_POL_CANCEL],
        """
        Neither, based on what's available. This flight never touches the US, so US DOT refund rules
        don't apply, and since Delta departed Mexico (not the EU) and isn't a European carrier, EU261
        doesn't cover it either. Delta's own policy is what's left: it will try to rebook you, and if
        you're unhappy with the new flight, you can cancel it for a refund of the unflown portion or take
        a 5-year eCredit. Mexico's own passenger-rights rules aren't covered here.
        """,
        route="OTHER_to_EU", carrier="Delta", disruption="Cancellation", control="N/A", origin="CUN",
        dest="AMS", category="route_applicability", edge="other_region",
        notes="False-US trap: 'not EU/UK' must never be read as US. Cancun is OTHER, so no US DOT, and a "
              "US carrier's flight INTO the EU gets no EU261.",
        expected_facts=[
            "This flight touches neither the US nor an EU/UK departure by a European carrier, so neither "
            "US DOT nor EU261 applies.",
            "A flight departing outside the US, EU, and UK cannot be assumed to fall under US rules just "
            "because it isn't European.",
        ],
        forbidden_claims=[
            "US DOT rules apply because the flight isn't within the EU.",
            "EU261 applies because the destination is in the EU.",
        ],
    ),
    case(
        "unsup-01",
        "My British Airways flight BA117 from London Heathrow to New York JFK was cancelled. What am I owed?",
        [R.UK_A3, R.UK_A5, R.UK_A7, R.UK_A8, R.UK_A9, R.CAA_CANCEL_COMP, R.P260_COVERED, R.P260_REFUND],
        """
        I can only check live flight status for American, Delta, United, and Southwest, so I can't look
        up BA117 directly - but the law still applies based on your route. Since your flight departed
        London for New York, UK261 and the US refund rule both apply regardless of the airline. For this
        cancellation, you're entitled to a refund or re-routing choice and care while you wait.
        Compensation can be 520 pounds for this route (over 3,500 km), but only if the cancellation
        qualifies: when you were told, the timing of any replacement flight, and whether extraordinary
        circumstances caused it all matter.
        """,
        route="UK_to_US", carrier="British Airways", disruption="Cancellation", control="Unknown",
        origin="LHR", dest="JFK", category="route_applicability",
        notes="Unsupported airline with a known route: production answers from government rules only, "
              "opens with its notice, and must not retrieve any AA/DL/UA/WN document or call AirLabs.",
        expected_facts=[
            "British Airways is not one of the four airlines this assistant tracks live flight data for, "
            "but the government rules (UK261 and US DOT) still apply based on the route.",
            "UK261 applies because the flight departed the UK; the US refund rule applies because it "
            "flies to the US.",
            "Compensation of up to £520 for this over-3,500km cancellation depends on the notice given, "
            "the timing of any re-routing, and extraordinary circumstances.",
        ],
        forbidden_claims=[
            "The assistant answers using an American, Delta, United, or Southwest policy document for "
            "this British Airways flight.",
        ],
    ),
    case(
        "unsup-02",
        "Is BA117 delayed today? Do I get anything if it is?",
        [],
        "",
        carrier="British Airways", **TRAP,
        notes="Unsupported airline, no route: production declines (expected_output is its reply) and does "
              "not look the flight up.",
        expected_facts=[
            "British Airways is not a carrier this assistant can look up live flight status for.",
            "No route is given, so there's nothing else to answer from government rules either.",
        ],
        forbidden_claims=[
            "The assistant fabricates a flight status or delay time for BA117.",
        ],
    ),

    # ======================================================================
    # Traps: the corpus cannot answer -> abstain and point somewhere useful
    # ======================================================================
    case(
        "trap-04",
        "How much compensation do Canada's Air Passenger Protection Regulations give for a 6-hour delay on my "
        "United flight from Toronto?",
        [],
        """
        The provided sources do not cover Canada's Air Passenger Protection Regulations, so I can't tell you the
        delay compensation amounts. This assistant covers US DOT rules, EU261 and UK261. Ask United, or check
        the Canadian Transportation Agency's guidance on APPR.
        """,
        carrier="United", **TRAP,
        expected_facts=[
            "Canada's Air Passenger Protection Regulations are not in this assistant's corpus (US DOT, "
            "EU261, UK261 only).",
        ],
        forbidden_claims=[
            "The assistant invents a compensation amount under Canadian law.",
        ],
    ),
    case(
        "trap-08",
        "How much compensation will JetBlue pay me for a 5-hour delay?",
        [],
        "",
        carrier="JetBlue", **TRAP,
        notes="Unsupported airline named in text, route unknown: production declines (expected_output is "
              "its reply). Before 2026-09-14 this was answered from all four supported airlines' documents.",
        expected_facts=[
            "JetBlue is not one of the four airlines this assistant supports.",
        ],
        forbidden_claims=[
            "The assistant states a compensation figure for JetBlue.",
        ],
    ),
]

"""US domestic cases (US -> US): US DOT rules only, plus each carrier's own documents."""

from __future__ import annotations

from . import refs as R
from .spec import case

DOM = dict(route="US_to_US")

CASES = [
    # ======================================================================
    # Delays — carrier-controlled cause stated
    # ======================================================================
    case(
        "dom-01",
        "My American Airlines flight from DFW to Chicago O'Hare is delayed 4 hours because of a crew "
        "scheduling problem. What am I owed?",
        [R.P260_DEF_DELAY, R.P260_REFUND, R.DOTQA_TRAVELED, R.AA_CSP_BYUS, R.AA_COC_BYUS, R.AA_COC_SCHED],
        """
        This is a US domestic flight, so US DOT rules apply, and there's no fixed cash payout for a
        delay. A federal refund depends on your arrival: if the delay means you're now scheduled to
        arrive 3 hours or more late, you can decline to travel and get a full refund of the fare and
        any fees. If you do fly, American's own policy is what applies: because the cause is a crew
        scheduling issue on American's side, you're eligible for a meal credit after the qualifying
        wait and, if you're stuck overnight, a hotel voucher and transport. Ask the gate agent to
        confirm the delay is coded as American's fault before requesting these.
        """,
        carrier="American", disruption="Delay", control="Controllable", origin="DFW", dest="ORD", **DOM,
        expected_facts=[
            "US DOT sets no fixed cash compensation for a domestic delay.",
            "The federal refund right turns on a scheduled arrival 3+ hours late (a significant delay), "
            "and applies only if the passenger declines to fly.",
            "Because American caused the delay, its Customer Service Plan offers a meal credit after 3+ "
            "hours and a hotel voucher for an overnight delay.",
        ],
        forbidden_claims=[
            "American is legally required to pay cash compensation for the delay.",
            "The passenger is entitled to a refund even though they intend to take the rebooked flight.",
            "A 4-hour wait at departure by itself guarantees a federal refund, whatever the arrival time.",
        ],
    ),
    case(
        "dom-04",
        "My Southwest flight from Dallas Love Field to Houston Hobby was delayed 3 hours because of a "
        "mechanical problem. Is there any compensation?",
        [R.WN_CSP_CONTROL, R.WN_CSP_VOUCHER, R.WN_COC_DEF, R.P260_DEF_DELAY, R.DOTQA_TRAVELED],
        """
        This is a domestic flight, so there's no federal cash compensation for a delay - a refund only
        applies if you decide not to fly. Because the mechanical issue is within Southwest's control, it
        will rebook you at no cost and give you a meal voucher on request. If you ultimately arrive at
        your final destination 3 or more hours late, you may also qualify for a LUV Voucher of at least
        $75; request it at Southwest.com/DelayForm within a year of the flight.
        """,
        carrier="Southwest", disruption="Delay", control="Controllable", origin="DAL", dest="HOU", **DOM,
        expected_facts=[
            "No federal cash compensation applies to delays; a refund only applies if the passenger "
            "chooses not to fly.",
            "Southwest rebooks at no cost and offers a meal voucher when the delay is within its control.",
            "A LUV Voucher of at least $75 may apply if the passenger arrives at the final destination "
            "3+ hours late, requested within a year.",
        ],
        forbidden_claims=[
            "Southwest must pay federal compensation for the delay.",
            "The LUV Voucher is guaranteed regardless of whether Southwest caused the delay.",
            "A 3-hour departure delay alone qualifies for the LUV Voucher, whatever the arrival time.",
        ],
    ),

    # ======================================================================
    # Delays — weather / ATC stated
    # ======================================================================
    case(
        "dom-07",
        "Snowstorm in Minneapolis and my Delta flight to Detroit is delayed until tomorrow. Does Delta owe "
        "me a hotel?",
        [R.DL_POL_REIMB, R.DL_DOM_R19_AMENITIES, R.P260_DEF_DELAY, R.P260_REFUND],
        """
        Usually no. Because this delay is caused by snow, Delta treats it as outside its control, and its
        policy does not cover hotel costs for weather delays - and US rules don't require one either.
        Since the delay stretches overnight, well past 3 hours, you can instead decline the rebooked
        flight and get a full refund of your fare and fees. If it turns out the real cause is within
        Delta's control, ask about reimbursement, but don't count on it here.
        """,
        carrier="Delta", disruption="Delay", control="Uncontrollable", origin="MSP", dest="DTW", **DOM,
        expected_facts=[
            "Weather delays don't create a US legal right to a hotel.",
            "Delta's contract excludes amenities for force-majeure delays, including weather.",
            "Because the delay is overnight (well over 3 hours), a full refund is available if the "
            "passenger declines to travel.",
        ],
        forbidden_claims=[
            "Delta is legally required to provide a hotel for a weather delay.",
            "Delta will reimburse hotel costs for this weather delay.",
        ],
    ),

    # ======================================================================
    # Delays — cause unknown (answer must be conditional)
    # ======================================================================
    case(
        "dom-13",
        "My Delta flight from Atlanta to Tampa is delayed 90 minutes. Can I cancel and get my money back?",
        [R.P260_DEF_DELAY, R.DOTQA_TRAVELED, R.DL_POL_TYPES, R.DL_POL_DELAY, R.DL_POL_RESTRICT],
        """
        Not yet. For a domestic flight the federal refund right starts when you're scheduled to arrive
        3 hours or more late, and a 90-minute delay doesn't reach that, so Delta isn't required to
        refund a non-refundable ticket for it. Keep watching the flight status - if the change means
        you'll arrive 3+ hours late, you can then decline to travel and take a refund.
        """,
        carrier="Delta", disruption="Delay", control="Unknown", origin="ATL", dest="TPA", **DOM,
        notes="Delta's footnote gives a >120-minute refund right for tickets issued before 28 Oct 2024 "
              "for travel on/after that date; 90 minutes is under that threshold either way.",
        edge="below_threshold",
        expected_facts=[
            "A US refund right for delay starts only once the flight is significantly delayed, meaning "
            "a scheduled arrival 3+ hours late for a domestic itinerary.",
            "A 90-minute delay does not meet that threshold, so no refund is currently owed.",
        ],
        forbidden_claims=[
            "A 90-minute delay already qualifies for a refund.",
            "Delta owes a refund regardless of how long the delay is.",
        ],
    ),

    # ======================================================================
    # Cancellations
    # ======================================================================
    case(
        "dom-15",
        "American cancelled my evening flight from Chicago O'Hare to Dallas because of a mechanical issue "
        "and the next flight is tomorrow morning. I don't live in Chicago. What does AA owe me?",
        [R.AA_CSP_BYUS, R.AA_COC_REBOOK, R.AA_COC_BYUS, R.P260_REFUND, R.DOTQA_REFUND],
        """
        Since American cancelled the flight for a mechanical reason - within its control - you're covered
        two ways. First, US rules entitle you to a full refund if you'd rather not travel at all. Second,
        if you do want to fly, American will rebook you on its own or a partner airline's next available
        flight at no cost, and because the cause is on American, it should also give you a hotel voucher,
        transport, and a food credit for the overnight wait. Ask the gate agent for these before booking
        anything yourself.
        """,
        carrier="American", disruption="Cancellation", control="Controllable", origin="ORD", dest="DFW", **DOM,
        expected_facts=[
            "A cancellation gives a full refund if the passenger declines rebooking and any credit.",
            "American rebooks on its own or a partner airline's next flight at no cost.",
            "Because the cause is within American's control, a hotel voucher, transport, and a food "
            "credit apply for the overnight wait.",
        ],
        forbidden_claims=[
            "American only owes a refund and nothing else.",
            "The passenger must book and pay for their own hotel with no reimbursement.",
        ],
    ),
    case(
        "dom-17",
        "Delta cancelled my Seattle to Minneapolis flight because of a maintenance issue and I don't want "
        "the new flight they booked. Can I get cash back instead of an eCredit?",
        [R.DL_POL_REFUND, R.DL_POL_CANCEL, R.P260_REFUND, R.P260_ACCEPT, R.P260_10],
        """
        Yes, cash back is your right here. Because Delta cancelled the flight, you can decline the
        rebooked flight and any eCredit, and Delta must give a full refund - including taxes and fees -
        to your original payment method, with no processing fee. Delta can't treat you as having accepted
        an eCredit unless you say so. In My Trips or with an agent, cancel the rebooked flight and choose
        refund, not credit.
        """,
        carrier="Delta", disruption="Cancellation", control="Controllable", origin="SEA", dest="MSP", **DOM,
        expected_facts=[
            "A cancellation gives the right to a full cash refund if the passenger declines rebooking and "
            "any credit.",
            "Delta cannot treat silence as acceptance of an eCredit; the passenger must actively agree.",
            "The refund must return to the original payment method with no processing fee.",
        ],
        forbidden_claims=[
            "Delta may issue an eCredit instead of cash without the passenger's agreement.",
            "No refund is available because the ticket was non-refundable.",
        ],
    ),
    case(
        "dom-21",
        "United cancelled my flight from Chicago to Newark for weather and offered me a travel credit. Do "
        "I have to accept it?",
        [R.P260_REFUND, R.P260_ALT, R.P260_ACCEPT, R.DOTQA_VOUCHER, R.UA_CC_CREDITS],
        """
        No, you don't have to accept the credit. Because United cancelled the flight, you can turn down
        the credit and take a full cash refund instead, including taxes and fees - the weather cause
        doesn't change this. United's own credits are described as valid for only a year, which is
        shorter than the five-year minimum federal rule requires when a credit substitutes for an owed
        refund. Tell United in writing that you want the refund, not the credit.
        """,
        carrier="United", disruption="Cancellation", control="Uncontrollable", origin="ORD", dest="EWR",
        category="law_vs_airline", **DOM,
        notes="Conflict between UA Customer Commitment (1-year credits) and 14 CFR 260.6(c) (5-year minimum "
              "for credits offered as an explicit alternative to a required refund). Reviewer: confirm this "
              "reading and whether the bot should flag it this directly.",
        expected_facts=[
            "For a cancelled flight, the passenger can decline any credit and United must issue a full "
            "refund instead.",
            "A credit substituted for a required refund must remain valid for at least five years under "
            "federal rules.",
            "United's own materials describe its credits as valid for only one year, shorter than that "
            "regulatory minimum.",
        ],
        forbidden_claims=[
            "The passenger must accept the travel credit United offered.",
            "A one-year credit satisfies the refund requirement.",
        ],
    ),
    case(
        "dom-24",
        "I paid cash for my Southwest ticket and the flight was cancelled. When will I get my refund?",
        [R.P260_PROMPT, R.P260_10, R.WN_CSP_REFUND],
        """
        For a cash purchase, federal rules require Southwest to issue the refund within 20 calendar days
        of your request, in your original form of payment, with no processing fee. If you haven't
        requested the refund yet, do so now - you're owed it as long as you didn't accept rebooking or a
        credit. Follow up if 20 days pass with nothing.
        """,
        route="unspecified", carrier="Southwest", disruption="Refund Procedure", control="N/A",
        category="procedure",
        notes="The DOT Q&A page in the corpus says '20 business days (for cash purchases)' in one answer and "
              "'20 calendar days' in another; 14 CFR 260.2 says calendar days. Expected answer follows the "
              "regulation.",
        expected_facts=[
            "US refund rules require processing within 20 calendar days for non-credit-card payments "
            "like cash.",
            "The refund must go to the original payment method with no processing fee.",
        ],
        forbidden_claims=[
            "Southwest can take longer than 20 calendar days without justification.",
            "A processing fee may be deducted from the refund.",
        ],
    ),

    # ======================================================================
    # Denied boarding (oversales)
    # ======================================================================
    case(
        "dom-25",
        "I got bumped from an oversold United flight from Houston to Denver and they rebooked me on a flight "
        "arriving 90 minutes after my original. What am I owed?",
        [R.P250_AMT_DOM, R.P250_PAY, R.UA_COC_DBC, R.UA_CC_OVERSALE],
        """
        Since you were involuntarily bumped and your new flight arrives just 90 minutes late, federal
        rules entitle you to at least 200% of your one-way fare, up to $1,075 - United's own contract
        matches that amount. It should be paid in cash or check the same day, or within 24 hours if your
        new flight leaves before that. Ask for the payment before you leave the airport.
        """,
        carrier="United", disruption="Denied Boarding", control="N/A", origin="IAH", dest="DEN", **DOM,
        expected_facts=[
            "Involuntary denied boarding with the replacement arriving 1-2 hours late entitles the "
            "passenger to at least 200% of the one-way fare, capped at $1,075.",
            "Payment must be made in cash or check on the day, or within 24 hours if the new flight "
            "departs first.",
        ],
        forbidden_claims=[
            "No compensation is owed because the passenger was rebooked.",
            "The compensation cap does not apply and any amount may be paid.",
        ],
    ),
    case(
        "dom-29",
        "United swapped to a smaller plane for operational reasons and I lost my seat on my Chicago to "
        "Denver flight. Do I get denied boarding compensation?",
        [R.P250_EXC, R.P250_NOTICE_EXC, R.UA_COC_DBC_EXC],
        """
        Probably not. Federal rules and United's own contract both exclude denied-boarding compensation
        when a smaller aircraft is substituted for operational or safety reasons. Ask United to confirm
        that reason in writing - if the real cause turns out to be an ordinary oversale instead, the usual
        compensation rules would apply. Either way, ask for the next available flight.
        """,
        carrier="United", disruption="Denied Boarding", control="N/A", origin="ORD", dest="DEN", **DOM,
        expected_facts=[
            "Denied boarding compensation does not apply when a smaller aircraft is substituted for "
            "operational or safety reasons.",
            "If the real reason were an ordinary oversale rather than an equipment swap, compensation "
            "would apply instead.",
        ],
        forbidden_claims=[
            "Compensation is automatically owed whenever a passenger loses their seat.",
            "An equipment swap made for safety reasons still requires compensation.",
        ],
    ),
    case(
        "dom-31",
        "I got to the gate late and American gave my seat away on an oversold flight. Am I owed bump "
        "compensation?",
        [R.P250_NOTICE_EXC, R.AA_COC_IDB],
        """
        Since you arrived at the gate late, you're likely not entitled to compensation - both federal
        rules and American's own conditions exclude passengers who didn't meet the airline's check-in and
        gate deadlines. Check what American's cutoff was for your flight; if you actually made it on time
        and were still bumped, ask for the written denied-boarding statement and compensation.
        """,
        route="unspecified", carrier="American", disruption="Denied Boarding", control="Passenger-caused",
        expected_facts=[
            "Denied boarding compensation does not apply if the passenger failed to meet the airline's "
            "check-in or gate deadlines.",
            "If the passenger did meet those deadlines, standard compensation rules would apply.",
        ],
        forbidden_claims=[
            "The passenger is entitled to bump compensation regardless of arriving late.",
        ],
    ),

    # ======================================================================
    # Tarmac delays
    # ======================================================================
    case(
        "dom-34",
        "Our Southwest flight sat on the ground at Chicago Midway for 2 hours after landing and we couldn't "
        "get off. What were they required to provide?",
        [R.P259_DEF_TARMAC, R.P259_TARMAC_FOOD, R.P259_TARMAC_UPDATES],
        """
        At the two-hour mark, federal rules already required Southwest to provide food and drinking water,
        unless the captain judged it unsafe, plus working lavatories, medical help if needed, and a status
        update once you passed 30 minutes on the ground. If any of that was missing, write down the times
        and file a complaint with Southwest.
        """,
        carrier="Southwest", disruption="Tarmac Delay", control="N/A", dest="MDW", **DOM,
        notes="Session 23: Southwest CSP section 9 was removed from the gold refs. It only says Southwest "
              "adopted a Tarmac Contingency Plan and lists causes; the food, water, lavatory and 30-minute "
              "obligations the answer states are 14 CFR 259.4's.",
        expected_facts=[
            "Airlines must provide food and water no later than two hours into a tarmac delay, unless "
            "safety prevents it.",
            "Working lavatories and medical attention must be available, plus a status update after 30 "
            "minutes.",
        ],
        forbidden_claims=[
            "No obligations apply until the tarmac delay exceeds three hours.",
        ],
    ),

    # ======================================================================
    # Missed connections, diversions, schedule changes, downgrades
    # ======================================================================
    case(
        "dom-36",
        "Delta's delay made me miss my connection in Detroit on my Boston to Minneapolis trip, and I'm going "
        "to arrive 5 hours late. Can I get a refund for the rest of the trip?",
        [R.DL_DOM_R19_REFUND, R.DL_POL_REFUND, R.P260_DEF_DELAY, R.P260_REFUND],
        """
        Because the delay pushes your arrival 5 hours late, this counts as significantly delayed under
        federal rules - if you'd rather not continue, Delta must refund the unflown portion, including
        fees like paid bags. Delta's own policy backs this up for any missed-connection delay of 3 hours
        or more. If you'd rather keep going, Delta will fly you on its next available flight in your
        original class instead.
        """,
        carrier="Delta", disruption="Missed Connection", control="Unknown", origin="BOS", dest="MSP", **DOM,
        expected_facts=[
            "A 5-hour late arrival makes the itinerary significantly delayed under federal rules, giving "
            "a refund right if the passenger stops traveling.",
            "Delta will cancel and refund the remaining ticket, including bag fees, for a missed-"
            "connection delay of 180+ minutes if requested.",
        ],
        forbidden_claims=[
            "No refund is available because part of the trip was already flown.",
        ],
    ),
    case(
        "dom-37",
        "My American flight to Phoenix diverted to Tucson and we won't be leaving again until after midnight. "
        "What does American provide?",
        [R.AA_CSP_DIV, R.AA_COC_BYUS, R.DOTQA_TRAVELED],
        """
        There's no federal requirement for a hotel after a diversion, but American's own commitment covers
        you here: since you won't board for Phoenix before 11:59 p.m., American should give you a hotel
        voucher, transport to and from it, and a food credit since you'll be held in Tucson more than 3
        hours. If it can't provide vouchers directly, it will reimburse reasonable costs instead. Ask the
        agent for these before booking anything yourself.
        """,
        carrier="American", disruption="Diversion", control="N/A", dest="PHX", **DOM,
        notes="AA Conditions of Carriage conditions the hotel on 'disruption is our fault or you're diverted' "
              "and 'away from your city of residence'; the CSP diversion section has no cause condition. "
              "Session 22: an outside audit wanted the food credit made conditional on cause; not adopted, "
              "because the CSP Diversions section grants it after 3 hours in the diversion city whatever "
              "the cause. The trigger is not boarding before 11:59 p.m., not arrival.",
        expected_facts=[
            "US rules don't require hotels or meals after a diversion; that comes from the airline's own "
            "commitments.",
            "American's policy provides a hotel voucher, transport, and a food credit when a diversion "
            "keeps a passenger from boarding before midnight.",
        ],
        forbidden_claims=[
            "American has no obligation at all after a diversion.",
            "The passenger must pay for their own hotel with no reimbursement option.",
        ],
    ),
    case(
        "dom-44",
        "Delta never responded to my written complaint about a cancelled flight. How long do they have?",
        [R.P259_COMPLAINT],
        """
        Federal rules give Delta 30 days to acknowledge your written complaint and 60 days to send a
        substantive response. If 60 days pass with nothing further, follow up in writing, referencing
        your original complaint and its date.
        """,
        route="unspecified", carrier="Delta", disruption="Complaint", control="N/A", category="procedure",
        expected_facts=[
            "Airlines must acknowledge a written complaint within 30 days and give a substantive "
            "response within 60 days.",
        ],
        forbidden_claims=[
            "There is no deadline for an airline to respond to a written complaint.",
        ],
    ),
]

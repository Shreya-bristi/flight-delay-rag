"""
Cases that touch Europe: UK -> US, EU -> US, US -> UK/EU, and intra-Europe.

The origin airport decides (CLAUDE.md applicability matrix): EU261/UK261 bind
these US carriers only on departure from EU/UK territory, and US DOT rules apply
only when the itinerary touches the US.

Distance bands used below (great-circle, Art 7): every US <-> Europe route here
is over 3,500 km; London-Paris/Amsterdam/Dublin are under 1,500 km;
London <-> Athens is between 1,500 and 3,500 km.
"""

from __future__ import annotations

from . import refs as R
from .spec import case

UK_US = dict(route="UK_to_US")
EU_US = dict(route="EU_to_US")

CASES = [
    # ======================================================================
    # UK -> US  (UK261 + US DOT)
    # ======================================================================
    case(
        "uk-us-01",
        "My United flight from LHR to EWR was delayed 4 hours due to a mechanical issue. What am I owed?",
        [R.UK_A3, R.UK_A6, R.UK_A7, R.UK_A9, R.CAA_DELAY_AMOUNTS, R.UK_REC14, R.UA_CC_ASSIST,
         R.UA_COC_NONUS, R.P260_DEF_DELAY],
        """
        Because your flight left London, UK261 applies alongside US rules. Compensation depends on how
        late you reach Newark: this route is over 3,500 km, so it's 520 pounds if you arrive more than
        4 hours late, or 260 pounds if you arrive between 3 and 4 hours late - unless United can prove
        the mechanical issue was a genuine extraordinary circumstance, which it has to justify, not just
        claim. At a 4-hour wait you're also owed meals and two free calls or emails. US DOT rules don't
        add anything here since their international delay-refund threshold is 6 hours. Claim from United
        in writing and keep your receipts.
        """,
        carrier="United", disruption="Delay", control="Controllable", origin="LHR", dest="EWR", **UK_US,
        notes="The user's reference example. Session 22: the UK CAA guidance (now in the corpus) pays £260 "
              "for a 3-4 hour arrival delay over 3,500 km and £520 above 4 hours; 'delayed 4 hours' does not "
              "say which delay, so the answer gives both bands. Case law (Huzar v Jet2) that routine "
              "technical faults are not extraordinary is still not in the corpus.",
        expected_facts=[
            "UK261 applies because the flight departed the UK; US DOT also applies because it flies to "
            "the US.",
            "Compensation turns on the arrival delay at the final destination: on this over-3,500km route "
            "£260 for 3-4 hours late, £520 for more than 4 hours, unless United proves extraordinary "
            "circumstances.",
            "US DOT's delay-refund threshold is 6 hours for international flights, so it isn't triggered "
            "here.",
        ],
        forbidden_claims=[
            "US DOT rules provide a fixed cash payment for this delay.",
            "A mechanical issue automatically counts as an extraordinary circumstance excusing "
            "compensation.",
            "£520 is owed for any delay of 3 hours or more, whatever the arrival delay.",
        ],
    ),
    case(
        "uk-us-03",
        "Fog at Heathrow delayed my Delta flight to Atlanta by 5 hours. Delta says weather means no "
        "compensation. Is that right?",
        [R.UK_A6, R.UK_A9, R.UK_A8, R.UK_REC14, R.DL_POL_REIMB, R.P260_DEF_DELAY],
        """
        Delta is only partly right. Fog can remove UK261 compensation, but only if Delta proves it was an
        extraordinary circumstance that caused the delay and that the delay couldn't have been avoided
        even with all reasonable measures - it isn't automatic, and you can ask for that proof. Either
        way, weather doesn't remove Delta's duty of care: at 5 hours you're still owed meals, two free
        calls or emails, and the option of a refund instead of flying. Ask Delta for those now and keep
        any receipts.
        """,
        carrier="Delta", disruption="Delay", control="Uncontrollable", origin="LHR", dest="ATL",
        category="law_vs_airline", **UK_US,
        expected_facts=[
            "UK261 lets the airline avoid compensation only if it proves the delay was an extraordinary "
            "circumstance, such as certain weather - it isn't automatic.",
            "Even if compensation is excused, UK261 still requires meals, communication, and a "
            "refund/rebooking option for the 5-hour wait.",
        ],
        forbidden_claims=[
            "Weather automatically excuses the airline from all obligations, including care.",
            "Delta doesn't need to prove anything to deny compensation.",
            "Delta must prove the fog made the flight impossible to operate safely.",
        ],
    ),
    case(
        "uk-us-06",
        "Delta emailed me three weeks before departure that my Heathrow to JFK flight is cancelled. Do I get "
        "UK261 compensation?",
        [R.UK_A5, R.UK_A8, R.P260_REFUND, R.DL_POL_REFUND],
        """
        No compensation here - UK261 excuses it when you're told at least two weeks ahead, and three
        weeks' notice clears that bar. You still get to choose between a full refund within seven days or
        re-routing to JFK at the next opportunity. Decide which you'd rather have and tell Delta.
        """,
        carrier="Delta", disruption="Cancellation", control="N/A", origin="LHR", dest="JFK", **UK_US,
        edge="exception_clause",
        expected_facts=[
            "UK261 compensation for a cancellation is not owed if the passenger was notified at least "
            "two weeks before departure.",
            "The passenger still keeps the right to a refund or re-routing.",
        ],
        forbidden_claims=[
            "Advance notice removes the refund or re-routing choice too.",
        ],
    ),
    case(
        "uk-us-09",
        "I booked Manchester, England to Newark to Denver on one United ticket. The first flight was late, I missed "
        "the connection and reached Denver 5 hours late. Does UK261 cover me?",
        [R.UK_A3, R.UK_A6, R.UK_A7, R.UA_CC_ASSIST, R.P260_DEF_DELAY],
        """
        Yes - because your whole trip was booked as one ticket starting in Manchester, UK261 treats the
        journey as departing the UK even though the missed connection happened in Newark. Since you
        reached Denver 5 hours late, you're entitled to 520 pounds compensation (this counts as an
        over-3,500 km route) unless United proves the delay was an extraordinary circumstance. Claim it
        in writing with your booking reference showing the single ticket.
        """,
        carrier="United", disruption="Missed Connection", control="Unknown", origin="MAN", dest="DEN",
        category="route_applicability", **UK_US,
        edge="multi_leg",
        expected_facts=[
            "A multi-leg journey booked as one ticket is treated as departing from the first leg's "
            "origin under UK261.",
            "A 3+ hour delay to the final destination on that basis triggers UK261 compensation (£520 "
            "here, over 3,500 km) unless United proves extraordinary circumstances.",
        ],
        forbidden_claims=[
            "UK261 doesn't apply because the missed connection happened at a US airport, not in the UK.",
        ],
    ),
    case(
        "uk-us-14",
        "I booked my Delta flight from Heathrow to Boston with SkyMiles and it was cancelled. Does UK261 still "
        "apply to an award ticket?",
        [R.UK_A3, R.UK_A5, R.UK_A8, R.CAA_CANCEL_COMP, R.P260_10],
        """
        Yes, UK261 still applies - it excludes only free tickets or ones with fares not open to the
        public, and a SkyMiles award ticket doesn't fall into that. Since your flight departed the UK,
        you're entitled to a refund or re-routing and care while you wait. Compensation of up to 520
        pounds depends on when Delta told you about the cancellation, the timing of any replacement
        flight, and whether extraordinary circumstances caused it. Ask Delta to refund the miles and
        fees, or re-route you, and ask when and why the flight was cancelled.
        """,
        carrier="Delta", disruption="Cancellation", control="Unknown", origin="LHR", dest="BOS",
        category="route_applicability", **UK_US,
        edge="exception_clause",
        expected_facts=[
            "UK261 covers frequent-flyer/award tickets, excluding only free or non-public reduced fares.",
            "Cancellation rights include a refund or re-routing choice, care, and compensation unless an "
            "exception applies.",
        ],
        forbidden_claims=[
            "Award or mileage tickets are excluded from UK261 protection.",
            "£520 compensation is owed for the cancellation regardless of notice or rerouting.",
        ],
    ),

    # ======================================================================
    # EU -> US  (EU261 + US DOT)
    # ======================================================================
    case(
        "eu-us-01",
        "My Delta flight from Paris CDG to New York JFK arrived 5 hours late because of a technical fault. Am "
        "I owed EU261 compensation?",
        [R.EU_A3, R.EU_A6, R.EU_A7, R.EU_A9, R.EUG_LONG_DELAY, R.EUG_DELAY_AMOUNT, R.EU_REC14,
         R.DL_INTL_R20_AMENITIES, R.P260_DEF_DELAY],
        """
        Very likely yes. Because your flight left Paris, EU261 applies, and under EU law a delay of 3 or
        more hours to your final destination is treated the same as a cancellation for compensation
        purposes - 600 euros on a route this long - unless Delta proves the technical fault was a genuine
        extraordinary circumstance it couldn't have avoided. Ask Delta exactly what the fault was. You're
        also owed meals and communication during the wait. Claim the 600 euros from Delta in writing.
        """,
        carrier="Delta", disruption="Delay", control="Controllable", origin="CDG", dest="JFK",
        **EU_US,
        expected_facts=[
            "EU261 applies because the flight departed an EU airport.",
            "A 3+ hour delay to final destination is treated as entitling passengers to Article 7 "
            "compensation under EU case law, €600 on this over-3,500km route, unless Delta proves "
            "extraordinary circumstances.",
            "The 3+ hour delay-compensation right comes from CJEU case law (Sturgeon), stated in the "
            "Commission's interpretative guidelines; the full €600 applies because the arrival delay "
            "is 4 hours or more.",
        ],
        forbidden_claims=[
            "US DOT rules provide the €600 compensation.",
            "A technical fault automatically qualifies as an extraordinary circumstance.",
        ],
    ),
    case(
        "eu-us-05",
        "United cancelled my Munich to Newark flight the day before and put me on a flight arriving 3 hours "
        "later than my original. How much EU261 compensation do I get?",
        [R.EU_A5, R.EU_A7],
        """
        Compensation is due here - notice of less than a week only excuses it if your replacement arrives
        under 2 hours late, and yours is 3 hours late, so that exemption doesn't apply. The full amount
        for this route is 600 euros, but because your new flight arrives within 4 hours of the original
        time, United can reduce it by half, to 300 euros. It still must be paid unless United proves
        extraordinary circumstances. Claim at least 300 euros in writing.
        """,
        carrier="United", disruption="Cancellation", control="Unknown", origin="MUC", dest="EWR", **EU_US,
        expected_facts=[
            "Short-notice cancellation only avoids compensation if the replacement arrives less than 2 "
            "hours after the original schedule; a 3-hour-later arrival does not qualify for that "
            "exemption.",
            "The base compensation for this over-3,500km route is €600, reducible to €300 because the "
            "replacement arrives within 4 hours of the original time.",
        ],
        forbidden_claims=[
            "No compensation is owed because United gave a day's notice.",
            "The full €600 must be paid with no possible reduction.",
        ],
    ),
    case(
        "eu-us-07",
        "I was denied boarding on an oversold American flight from Madrid to Miami and they offered me a "
        "voucher. What am I entitled to?",
        [R.EU_A4, R.EU_A7, R.EU_A8, R.EU_A9, R.AA_COC_DBC, R.P250_APPL],
        """
        EU261 covers this, not the US oversales rule, since you departed Madrid. You're owed compensation
        immediately: 600 euros for a route this long, or 300 euros if American gets you to Miami within
        4 hours of your original arrival time. A voucher is only valid if you agree to it in writing - you
        can insist on cash, bank transfer, or check instead. Decline the voucher unless you actually
        prefer it, and ask for the cash payment.
        """,
        carrier="American", disruption="Denied Boarding", control="N/A", origin="MAD", dest="MIA",
        category="law_vs_airline", **EU_US,
        expected_facts=[
            "EU261, not the US oversales rule, governs since the flight departed the EU.",
            "Denied boarding compensation is owed immediately: €600 for this distance, or €300 if "
            "rebooked within 4 hours of the original arrival.",
            "A voucher is only acceptable if the passenger signs to agree to it; cash, bank transfer, or "
            "check can be insisted on instead.",
        ],
        forbidden_claims=[
            "The passenger must accept the voucher American offered.",
            "US oversales rules set the compensation amount here.",
        ],
    ),
    case(
        "eu-us-12",
        "Does EU261 apply to my Delta flight from Amsterdam to Minneapolis if I bought the ticket in the US?",
        [R.EU_A3],
        """
        Yes - EU261 looks at where the flight departs, not where you bought the ticket, so your Amsterdam
        departure is covered regardless. Keep in mind the return flight, Minneapolis to Amsterdam, would
        only get EU261 if the operating airline were a European carrier, and Delta isn't. If this
        Amsterdam flight is delayed, cancelled, or overbooked, you can rely on EU261.
        """,
        carrier="Delta", disruption="None", control="N/A", origin="AMS", dest="MSP",
        category="route_applicability", **EU_US,
        edge="exception_clause",
        expected_facts=[
            "EU261 applies based on the departure airport, not where the ticket was purchased.",
            "The return leg (US to EU) would not get EU261 unless the operating carrier is a European "
            "(community) carrier.",
        ],
        forbidden_claims=[
            "EU261 doesn't apply because the ticket was bought in the US.",
        ],
    ),
    case(
        "eu-us-13",
        "My Delta itinerary was Nice to New York JFK, then JFK to Raleigh-Durham, on one booking. The Nice "
        "flight was delayed, I missed my connection and reached Raleigh 6 hours late. Does EU261 apply?",
        [R.EU_A2, R.EU_A3, R.EU_A7, R.EUG_CONNECTION, R.EUG_LONG_DELAY, R.DL_INTL_R20_REFUND],
        """
        Yes. Because your itinerary started in Nice on one booking, EU261 covers the whole trip, and what
        matters is your final stop, Raleigh-Durham, not where the connection was missed. Arriving there
        6 hours late entitles you to 600 euros compensation for this distance, unless Delta proves
        extraordinary circumstances. Delta separately offers to refund the unused portion if a change
        makes you miss connections. Claim the 600 euros in writing, citing your single booking reference.
        """,
        carrier="Delta", disruption="Missed Connection", control="Unknown", origin="NCE", dest="RDU",
        category="route_applicability", **EU_US,
        edge="multi_leg",
        expected_facts=[
            "EU261 applies because the trip began at an EU airport, and for connecting flights on one "
            "booking the 'final destination' is the last flight's destination.",
            "Arriving 3+ hours late at the final destination triggers compensation under CJEU case law "
            "(Commission guidelines); at 6 hours late it is the full €600 for this over-3,500km "
            "itinerary.",
        ],
        forbidden_claims=[
            "EU261 doesn't apply because the missed connection happened in the US.",
        ],
    ),

    # ======================================================================
    # US -> UK / US -> EU  (US DOT only; EU261/UK261 do NOT apply)
    # ======================================================================
    case(
        "us-eur-01",
        "My Delta flight from JFK to Rome was 4 hours late. I heard EU261 pays €600 — can I claim it?",
        [R.EU_A3, R.EU_A2, R.P260_DEF_DELAY, R.DL_INTL_R20_AMENITIES, R.DL_POL_REIMB],
        """
        No, not on this flight. EU261 only covers flights into the EU when the airline itself is
        European, and Delta isn't, so a US-departing flight to Rome falls outside it - it would apply to
        your return flight home instead. Under US rules there's no fixed cash payout for delays, and the
        refund threshold for an international flight is 6 hours, which your 4-hour delay doesn't reach.
        Ask Delta about its own hotel or meal provisions for delays it caused.
        """,
        route="US_to_EU", carrier="Delta", disruption="Delay", control="Unknown", origin="JFK", dest="FCO",
        category="route_applicability",
        edge="inapplicable_regime",
        expected_facts=[
            "EU261 only covers a US-departing flight into the EU if the operating airline is a European "
            "(community) carrier.",
            "Delta is a US carrier, so EU261 does not apply to this outbound flight; it would apply on "
            "the return leg from Rome.",
            "US DOT rules govern instead, with a 6-hour threshold for an international delay refund, "
            "which 4 hours doesn't meet.",
        ],
        forbidden_claims=[
            "The passenger can claim €600 under EU261 for this outbound delay.",
        ],
    ),
    case(
        "us-eur-02",
        "United flight from Newark to London Heathrow delayed 5 hours. Can I claim UK261?",
        [R.UK_A3, R.P260_DEF_DELAY, R.UA_CC_ASSIST, R.UA_CC_REFUND, R.UA_COC_LODGING],
        """
        No. UK261 only covers a flight into the UK if the airline is British or European, and United is a
        US carrier, so it doesn't apply here - it would cover your flight home from London. Under US
        rules there's no fixed delay compensation, and the refund threshold for an international flight
        is 6 hours; your 5-hour delay is under that. Ask United about its own meal or hotel provisions if
        the delay is within its control.
        """,
        route="US_to_UK", carrier="United", disruption="Delay", control="Unknown", origin="EWR", dest="LHR",
        category="route_applicability",
        edge="inapplicable_regime",
        expected_facts=[
            "UK261 covers a US-departing flight into the UK only if the airline is a UK or EU carrier; "
            "United is neither.",
            "US DOT's international delay-refund threshold is 6 hours, which a 5-hour delay does not "
            "meet.",
        ],
        forbidden_claims=[
            "UK261 applies to this outbound flight because it's landing in the UK.",
        ],
    ),
    case(
        "us-eur-04",
        "Delta bumped me from an oversold Atlanta to Amsterdam flight and the replacement arrives 3 hours "
        "later. Compensation?",
        [R.P250_AMT_INTL, R.P250_NOTICE_INTL, R.DL_INTL_R21_DBC, R.EU_A3],
        """
        US oversales rules cover this, not EU261, since Delta departed a US airport and isn't a European
        carrier. Because your replacement flight arrives 3 hours later - within the 1-to-4-hour band -
        you're entitled to at least 200% of your one-way fare, up to $1,075, paid in cash or check the
        same day. Ask for that payment before you leave the airport; you can decline a travel credit and
        insist on cash.
        """,
        route="US_to_EU", carrier="Delta", disruption="Denied Boarding", control="N/A", origin="ATL",
        dest="AMS",
        expected_facts=[
            "The US oversales rule applies to a US-departing flight, not EU261, since Delta isn't an EU "
            "carrier.",
            "An alternate flight arriving 1-4 hours late entitles the passenger to at least 200% of the "
            "one-way fare, capped at $1,075.",
        ],
        forbidden_claims=[
            "EU261 governs this compensation instead of the US oversales rule.",
            "No compensation is owed because the delay is only 3 hours.",
        ],
    ),
    case(
        "us-eur-06",
        "Our United flight from Newark to Frankfurt has been on the tarmac at Newark for 3.5 hours. Don't they "
        "have to let us off at 3 hours?",
        [R.P259_TARMAC_DEPLANE, R.P259_TARMAC_FOOD, R.UA_CC_TARMAC],
        """
        Not on this flight - the 3-hour deplaning rule is for domestic flights only; international ones
        get 4 hours before the airline must offer a chance to get off, unless safety or air traffic
        control prevents it. You should already have had food and water, since that requirement kicks in
        at 2 hours regardless. If you haven't, ask the crew now.
        """,
        route="US_to_EU", carrier="United", disruption="Tarmac Delay", control="N/A", origin="EWR", dest="FRA",
        expected_facts=[
            "The 3-hour deplaning limit applies only to domestic flights; international flights get 4 "
            "hours.",
            "Food, water, working lavatories, and medical attention must still be provided starting at 2 "
            "hours.",
        ],
        forbidden_claims=[
            "This international flight must let passengers off after 3 hours.",
        ],
    ),
    case(
        "us-eur-09",
        "I have a United round trip Chicago–Paris–Chicago. My outbound flight was 5 hours late. If my return "
        "from Paris is also delayed, does EU261 apply to either flight?",
        [R.EU_A3, R.EU_A2, R.EU_A6, R.EU_A7, R.EUG_LONG_DELAY, R.EUG_DELAY_AMOUNT],
        """
        Only the return leg. Your outbound Chicago-to-Paris delay isn't covered by EU261 because United
        is a US carrier - only US rules apply there, with no fixed cash payout. Your return flight from
        Paris is different: EU261 covers it regardless of airline. If it arrives in Chicago at least 3
        but less than 4 hours late, compensation is 300 euros; at 4 hours or more it's 600 euros, unless
        United proves extraordinary circumstances. Keep your return boarding pass and delay details in
        case you need to claim.
        """,
        route="US_EU_round_trip", carrier="United", disruption="Delay", control="Unknown",
        category="route_applicability",
        edge="multi_leg",
        expected_facts=[
            "EU261 doesn't cover the outbound (US-to-EU) leg because United isn't a European carrier.",
            "EU261 does cover the return (EU-to-US) leg regardless of airline, since it departs the EU.",
            "On the return leg, compensation depends on the arrival delay: €300 for 3 to under 4 hours, "
            "€600 at 4 hours or more (CJEU case law, per the Commission guidelines).",
        ],
        forbidden_claims=[
            "EU261 covers the outbound Chicago-to-Paris delay.",
            "Any 3+ hour delay on the return leg earns the full €600.",
        ],
    ),

    # ======================================================================
    # Intra-Europe  (UK -> EU: UK261 only; EU -> UK: EU261 only; no US DOT)
    # ======================================================================
    case(
        "intra-04",
        "Delta cancelled my Athens to London Heathrow flight on the day. What does EU law give me?",
        [R.EU_A3, R.EU_A5, R.EU_A7, R.EU_A8, R.EU_A9, R.EUG_REROUTE_EXEMPT],
        """
        EU261 covers this - your flight left Athens, an EU airport - and US rules don't apply since the
        trip never touches the US. You get a refund or re-routing choice and care while you wait. Because
        you were told on the day, no compensation is due if Delta re-routes you to leave no more than 1
        hour early and arrive less than 2 hours late. Otherwise it's 400 euros for this route (or 200
        euros if the re-routing gets you there within 3 hours of your original time), unless Delta proves
        extraordinary circumstances. Choose refund or re-routing, then claim the compensation in writing.
        """,
        route="EU_to_UK", carrier="Delta", disruption="Cancellation", control="Unknown", origin="ATH",
        dest="LHR", notes="Hypothetical sector for a US carrier (see intra-01).",
        expected_facts=[
            "EU261 applies because the flight departed Greece, an EU country; US DOT does not apply "
            "since the trip never touches the US.",
            "With same-day notice, no compensation is due if the re-routing departs no more than 1 hour "
            "early and arrives less than 2 hours late.",
            "Otherwise compensation for this 1,500-3,500 km route is €400, reducible to €200 if the "
            "re-routing arrives within 3 hours of the original arrival, unless Delta proves "
            "extraordinary circumstances.",
        ],
        forbidden_claims=[
            "US DOT rules apply to this flight.",
            "€400 is owed whatever re-routing Delta offers.",
        ],
    ),
    case(
        "intra-07",
        "My Delta flight from London to Amsterdam was cancelled. Can I get a refund under US DOT rules?",
        [R.P260_COVERED, R.P260_REFUND, R.UK_A3, R.UK_A5, R.UK_A8, R.CAA_CANCEL_COMP],
        """
        No, not under US rules - those only cover flights touching the US, and London to Amsterdam
        doesn't. You're covered by UK261 instead, since the flight left the UK: you get a refund within
        seven days or re-routing, and care while you wait. Compensation can be up to 220 pounds for this
        distance, but whether it's owed depends on when you were told, the timing of any replacement
        flight, and whether extraordinary circumstances caused the cancellation. Ask Delta for the UK261
        refund or re-routing, and ask when and why the flight was cancelled.
        """,
        route="UK_to_EU", carrier="Delta", disruption="Cancellation", control="Unknown", origin="LHR",
        dest="AMS", category="route_applicability",
        notes="Hypothetical sector for a US carrier (see intra-01).",
        edge="inapplicable_regime",
        expected_facts=[
            "US DOT refund rules only cover flights to, from, or within the US, so they don't apply to a "
            "London-Amsterdam flight.",
            "UK261 applies instead because the flight departed the UK.",
            "Compensation (up to £220 here) depends on notice, re-routing timing and the cause.",
        ],
        forbidden_claims=[
            "US DOT refund rules apply to this flight.",
            "£220 is owed for any cancellation of this flight.",
        ],
    ),
]

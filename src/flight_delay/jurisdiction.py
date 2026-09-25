"""
Determine which passenger-rights regimes apply to a flight question.

All four supported airlines are US carriers, but EU261 or UK261 may also apply
when the flight departs from the EU or UK.

For these carriers:

    UK -> US   UK261 + US DOT
    EU -> US   EU261 + US DOT
    US -> UK   US DOT only
    US -> EU   US DOT only
    US -> US   US DOT only
    UK -> EU   UK261 only
    EU -> UK   EU261 only

Regions are US, EU, UK and OTHER (genuinely outside all three). Airport codes
are classified by data/airport_regions.json and nothing else. OTHER never
stands in for the US: US DOT governs only when a known or assumed endpoint is a
US airport, so DXB -> CDG carries no US rules.

The module returns:

    governing  Regimes the known or assumed route actually supports. They
               receive guaranteed context.

    scope      Regimes worth retrieving, even if they do not govern.

For example, a US -> Paris flight may include EU261 in `scope` so Article 3 can
explain why it does not apply, but EU261 is not added to `governing`. Likewise
an UNKNOWN endpoint is not treated as the US, but may be one: Paris -> ? has
EU261 governing and US DOT in `scope` only, so refund rules are not lost from
retrieval before the passenger says where the flight was going.

If a passenger asks about their own disrupted trip and the departure is not
known ("my Paris flight was cancelled", "my Delta flight was delayed 5 hours",
"my Manchester to Newark flight"), `needs_clarification` is set and the pipeline
asks for what is missing: the flight number (looked up in AirLabs), the
departure/arrival airports, which Manchester, or which country an unrecognised
city is in. If the reply still leaves it open, `assume_route()` picks the route
to answer on and the sentence the answer opens with ("Assuming you're flying
from Manchester, UK.").

`resolve_regimes()` is the matrix itself, as a pure function of the two
regions. `detect_jurisdictions()` uses it, and so does the golden-set builder,
so the eval set and the running system cannot disagree about the law.

The legal applicability rules themselves remain in the source documents; this
module only decides which jurisdictions should be retrieved or prioritized.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Airport code -> region. The single source of truth is
# data/airport_regions.json (AirLabs /airports, classified US / EU / UK /
# OTHER; EEA states and Switzerland are EU, since EU261 applies there).
# Loaded once at import; a missing file stops the process rather than letting
# every airport silently fall back to OTHER.
# --------------------------------------------------------------------------

AIRPORT_REGIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "airport_regions.json"


def _load_airport_regions(path: Path = AIRPORT_REGIONS_PATH) -> dict[str, str]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["airports"]
    except FileNotFoundError:
        raise RuntimeError(
            f"Airport region table not found at {path}. It decides which passenger-rights "
            "regime applies; ship data/airport_regions.json with the app."
        ) from None


_AIRPORT_REGIONS = _load_airport_regions()

# Codes recognised when a passenger TYPES them ("ATL to CDG"): these carriers'
# main US and European airports. Not a region table (regions come from the
# JSON above); a whitelist, because nearly every uppercase three-letter word is
# some airport's code: CFR (Caen), TSA (Taipei), ATC, AND, NOT... Live AirLabs
# codes are not limited to this list.
TEXT_AIRPORT_CODES = frozenset(["CDG", "ORY", "NCE", "LYS", "MRS", "TLS", "BOD", "FRA", "MUC", "BER", "DUS", "HAM", "STR", "CGN", "NUE", "AMS", "BRU", "CRL", "LUX", "DUB", "SNN", "ORK", "MAD", "BCN", "AGP", "PMI", "VLC", "SVQ", "ALC", "IBZ", "LPA", "TFS", "TFN", "LIS", "OPO", "FAO", "FNC", "PDL", "FCO", "MXP", "LIN", "VCE", "NAP", "BLQ", "FLR", "PSA", "CTA", "PMO", "BRI", "ATH", "SKG", "HER", "RHO", "JMK", "JTR", "CFU", "VIE", "SZG", "INN", "ZRH", "GVA", "BSL", "CPH", "BLL", "ARN", "GOT", "NYO", "OSL", "BGO", "TRD", "SVG", "HEL", "KEF", "WAW", "KRK", "GDN", "WRO", "KTW", "PRG", "BUD", "OTP", "CLJ", "SOF", "VAR", "BOJ", "ZAG", "SPU", "DBV", "LJU", "BTS", "TLL", "RIX", "VNO", "KUN", "MLA", "LCA", "PFO", "LHR", "LGW", "LCY", "LTN", "STN", "SEN", "MAN", "BHX", "EDI", "GLA", "ABZ", "NCL", "LPL", "BRS", "CWL", "BFS", "BHD", "INV", "EMA", "LBA", "SOU", "EXT", "NQY", "ATL", "ORD", "MDW", "DFW", "DAL", "DEN", "LAX", "SFO", "SJC", "OAK", "SAN", "SEA", "PDX", "LAS", "PHX", "TUS", "SLC", "JFK", "LGA", "EWR", "BOS", "IAD", "DCA", "BWI", "PHL", "PIT", "CLT", "RDU", "BNA", "MIA", "FLL", "MCO", "TPA", "RSW", "JAX", "IAH", "HOU", "AUS", "SAT", "MSY", "MSP", "DTW", "CLE", "CMH", "CVG", "IND", "STL", "MCI", "MKE", "OMA", "ABQ", "HNL", "OGG", "ANC", "SMF", "BUR", "ONT", "SNA", "MHT", "BHM", "BDL", "PVD", "ALB", "BUF", "SYR", "SJU", "BQN", "STT", "STX", "GUM", "SPN"])

# --------------------------------------------------------------------------
# Place names, as passengers write them. Cities and countries are listed apart
# because a reply of "Romania" or "UK" names a COUNTRY: it can settle which
# Manchester was meant, or which country an unrecognised city is in.
# --------------------------------------------------------------------------

_EU_CITIES = (
    "paris|nice|lyon|marseille|toulouse|bordeaux|"
    "frankfurt|munich|münchen|berlin|dusseldorf|düsseldorf|hamburg|stuttgart|cologne|"
    "amsterdam|schiphol|brussels|luxembourg|dublin|shannon|cork|"
    "madrid|barcelona|malaga|málaga|mallorca|majorca|valencia|seville|alicante|ibiza|"
    "canary islands|tenerife|lisbon|porto|faro|madeira|azores|"
    "rome|milan|venice|naples|bologna|florence|pisa|catania|palermo|"
    "athens|thessaloniki|crete|rhodes|mykonos|santorini|corfu|"
    "vienna|salzburg|innsbruck|zurich|zürich|geneva|basel|"
    "copenhagen|stockholm|gothenburg|oslo|bergen|helsinki|reykjavik|reykjavík|"
    "warsaw|krakow|kraków|gdansk|gdańsk|wroclaw|prague|budapest|bucharest|sofia|"
    "zagreb|split|dubrovnik|ljubljana|bratislava|tallinn|riga|vilnius|"
    "malta|cyprus|larnaca|paphos"
)
_EU_COUNTRIES = (
    "france|french|germany|german|netherlands|holland|dutch|belgium|belgian|ireland|irish|"
    "spain|spanish|portugal|portuguese|italy|italian|greece|greek|austria|austrian|"
    "switzerland|swiss|denmark|danish|sweden|swedish|norway|norwegian|finland|finnish|"
    "iceland|icelandic|poland|polish|czech republic|czechia|czech|hungary|hungarian|"
    "romania|romanian|bulgaria|bulgarian|croatia|croatian|slovenia|slovakia|"
    "estonia|latvia|lithuania|maltese|"
    "european union|eu ?261|ec ?261|eu|europe|european"
)
_UK_CITIES = (
    "london|heathrow|gatwick|stansted|luton|"
    "edinburgh|glasgow|aberdeen|liverpool|bristol|cardiff|belfast|inverness"
)
_UK_COUNTRIES = "england|scotland|wales|northern ireland|united kingdom|great britain|britain|british|uk ?261|uk"
# US cities, states and territories. San Juan / Puerto Rico are US: 14 CFR 250
# counts "the territories and possessions" as within the United States.
_US_CITIES = (
    "new york|newark|laguardia|boston|washington|dulles|baltimore|philadelphia|pittsburgh|"
    "atlanta|charlotte|raleigh|durham|nashville|miami|fort lauderdale|orlando|tampa|jacksonville|"
    "chicago|o'hare|o’hare|midway|detroit|cleveland|columbus|cincinnati|indianapolis|"
    "minneapolis|st\\.? louis|kansas city|milwaukee|omaha|"
    "dallas|love field|fort worth|houston|hobby|austin|san antonio|new orleans|"
    "denver|salt lake city|phoenix|tucson|albuquerque|las vegas|"
    "los angeles|san francisco|oakland|san jose|san diego|sacramento|burbank|"
    "seattle|portland|honolulu|anchorage|hartford|providence|buffalo|san juan"
)
_US_COUNTRIES = (
    "united states|america|"
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|"
    "hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|"
    "massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|"
    "new hampshire|new jersey|new mexico|north carolina|north dakota|ohio|oklahoma|"
    "oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|"
    "vermont|virginia|west virginia|wisconsin|wyoming|puerto rico"
)
# Outside the US, EU and UK: no regime in this corpus governs these.
_OTHER_CITIES = (
    "toronto|vancouver|montreal|calgary|cancun|cancún|mexico city|"
    "tokyo|seoul|sydney|tel aviv|istanbul|dubai|delhi|mumbai|sao paulo|são paulo|bogota|bogotá"
)
_OTHER_COUNTRIES = (
    "canada|mexico|japan|korea|china|india|australia|brazil|argentina|colombia|"
    "turkey|israel|serbia|albania|ukraine|russia|moldova|montenegro|bosnia|north macedonia|"
    "morocco|egypt"
)


def _words_re(alternation: str, flags=re.I) -> re.Pattern:
    return re.compile(r"\b(" + alternation + r")\b", flags)


# (pattern, region, is_country)
_PLACE_TABLE = (
    (_words_re(_EU_CITIES), "EU", False),
    (_words_re(_EU_COUNTRIES), "EU", True),
    (_words_re(_UK_CITIES), "UK", False),
    (_words_re(_UK_COUNTRIES), "UK", True),
    (_words_re(_US_CITIES), "US", False),
    (_words_re(_US_COUNTRIES), "US", True),
    (_words_re(_OTHER_CITIES), "OTHER", False),
    (_words_re(_OTHER_COUNTRIES), "OTHER", True),
    # Case-sensitive: "US" is a country, "us" is a pronoun.
    (re.compile(r"\b(US|USA|U\.S\.(?:A\.)?|NH)(?![\w.])"), "US", True),
)

# Place names that are also everyday words: they count only when capitalised.
_COMMON_WORD_PLACES = frozenset({
    "nice", "split", "cork", "faro", "hobby", "midway", "buffalo", "turkey", "china",
    "georgia", "phoenix", "reading", "washington", "columbus", "portland", "providence",
})

# UK cities that share a name with places outside the UK. Named on their own,
# the passenger is asked which one they mean.
AMBIGUOUS_PLACES = {
    "manchester": ("Manchester, UK (MAN)", "Manchester, New Hampshire, US (MHT)"),
    "birmingham": ("Birmingham, UK (BHX)", "Birmingham, Alabama, US (BHM)"),
    "newcastle": ("Newcastle, UK (NCL)", "Newcastle, Australia (NTL)"),
}
_AMBIGUOUS_RE = _words_re("|".join(AMBIGUOUS_PLACES))

# Mentions that name a legal regime or a nationality, not a departure point.
_REGIME_WORDS = frozenset({
    "eu", "uk", "us", "usa", "u.s.", "eu261", "eu 261", "ec261", "ec 261", "uk261", "uk 261",
    "europe", "european", "european union", "british", "french", "german", "dutch", "irish",
    "spanish", "portuguese", "italian", "greek", "austrian", "swiss", "danish", "swedish",
    "norwegian", "finnish", "icelandic", "polish", "hungarian", "romanian", "bulgarian",
    "croatian", "belgian", "maltese", "america",
})

# Finds things that look like 3-letter airport codes. Only TEXT_AIRPORT_CODES
# that are also in the region table count, so "ATC" or "TSA" is not an airport.
_IATA_TOKEN_RE = re.compile(r"\b([A-Z]{3})\b")

#the location after this word is probably the origin
_DEPART_WORDS = frozenset({"from", "out", "departing", "leaving", "originating", "depart", "departed", "departs"})
#These indicate a destination
_ARRIVE_WORDS = frozenset({"to", "into", "arriving", "arrive", "arrived", "arrives", "bound", "toward", "towards"})
#This combines the departure and arrival expressions into a regular expression
_DIRECTION_RE = re.compile(
    r"\b(from|out of|departing|leaving|originating|departs?|departed|"
    r"to|into|arriving|arrives?|arrived|bound for|towards?)\b",
    re.I,
)
# Handles a route written like "Paris to New York": a place immediately followed by "to" is the origin.
# A bare hyphen or dash is NOT read as direction: "my Dublin–Chicago flight" is
# how passengers name a route in either direction (or a round trip).
_TRAILING_TO_RE = re.compile(r"^\s*(?:→|to\b)", re.I)

# Words allowed between a direction marker and the place it describes
# ("from an EU flight", "to the UK"). Anything else means the marker belongs to
# a different phrase: in "a flight from Delta One to economy on my Paris to
# Atlanta flight", the "to" before "economy" says nothing about Paris.
_GAP_WORDS = frozenset({"the", "a", "an", "my", "our"})

# A capitalised phrase of up to three words: a proper noun or an airport code.
_CAP = r"([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,2})"
# Slots where a capitalised phrase is almost certainly a place. Used to spot
# places that are in none of the tables above ("from Nantes", "my Timisoara flight").
_PLACE_SLOT_RES = (
    re.compile(r"\b(?i:from|out of|departing|leaving|originating (?:in|from|at)|to|into|"
               r"bound for|arriving (?:in|at)|landed in|landing in)\s+(?:the\s+)?" + _CAP),
    re.compile(r"\b(?i:my|our|the|a)\s+" + _CAP + r"\s+(?i:flight|trip|connection)\b"),
    re.compile(_CAP + r"\s+(?i:to)\s+(?=[A-Z])"),
    re.compile(_CAP + r"\s*[–—-]\s*(?=[A-Z])"),
    re.compile(r"[–—-]\s*" + _CAP),
)

# Capitalised words that are not places.
_NOT_A_PLACE = frozenset(["American", "Delta", "United", "Southwest", "JetBlue", "Alaska", "Spirit", "Frontier", "Airlines", "Air", "Lines", "AA", "DL", "UA", "WN", "DOT", "FAA", "TSA", "ATC", "CAA", "I", "I'm", "I've", "I'd", "My", "Our", "The", "This", "That", "A", "An", "It", "It's", "Terminal", "Gate", "Economy", "Premium", "First", "Business", "Main", "Basic", "Comfort", "Plus", "Class", "Cabin", "Select", "One", "Flight", "Credit", "Credits", "Voucher", "LUV", "SkyMiles", "AAdvantage", "MileagePlus", "Rapid", "Rewards", "Eurostar", "Wi", "Fi", "Rule", "Article", "Section", "Part", "Regulation", "Contract", "Conditions", "Carriage", "Customer", "Service", "Plan", "Commitment", "Can", "Could", "Do", "Does", "Did", "Is", "Are", "Was", "Were", "Will", "Would", "Should", "What", "When", "Where", "Which", "Who", "Why", "How", "If", "Am", "Aren't", "Don't", "Isn't", "Compensation", "Flying", "Flew", "Fly", "Travelling", "Traveling", "Going", "Heading", "Returning", "Connecting", "Leaving", "Departing", "Snow", "Fog", "Snowstorm", "Storm", "Weather", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

# The passenger is asking about THEIR OWN trip, not about the rules in general.
_PERSONAL_TRIP_RE = re.compile(
    r"\b(my|our|mine|i|i'm|i’m|i've|i’ve|i'd|we|we're|we’re|we've|we’ve|me)\b", re.I
)
# ...and about something that went wrong with it.
_DISRUPTION_RE = re.compile(
    # Common misspellings count: "my delta flight in cacelled" skipped the question
    # and was answered on an unstated US assumption (Session 17).
    r"\b(delay\w*|de(?:al|l)y\w*|late|cancel\w*|ca(?:nc|n|c)e?l+(?:ed|ing|ation|ations)?|"
    r"bumped|denied|overbook\w*|oversold|miss(?:ed|ing)|"
    r"misconnect\w*|divert\w*|diversion|tarmac|stranded|stuck|downgrad\w*|rebook\w*|"
    r"schedule change|refund\w*|compensat\w*|voucher|hotel|meals?)\b",
    re.I,
)
_DOMESTIC_RE = re.compile(r"\bdomestic\b", re.I)

REGIME_NAMES = {"US": "US DOT", "EU": "EU261", "UK": "UK261"}
_ORDER = ("US", "EU", "UK")
_KNOWN = ("US", "EU", "UK", "OTHER")   # a settled region; OTHER = outside US/EU/UK
_EUROPE = ("EU", "UK")


@dataclass(frozen=True)
class Place:
    """One place named in the text. region: US | EU | UK | OTHER | AMBIGUOUS | UNKNOWN."""

    text: str
    region: str
    start: int
    end: int
    country: bool = False      # a country/state/regime word, not an airport city
    role: str | None = None    # "origin" | "destination" | None


@dataclass(frozen=True)
class RouteJurisdiction:
    """What the route means for which passenger-rights regimes are in play."""
    """"
    scope: Which legal regimes might be relevant enough that their documents can be retrieved?
    US to Paris could give EU in scope because EU261 may be useful for explaining why it does not apply.

    governing: Which regime actually applies and should receive guaranteed context? Us to Paris
    EU might be inscope, but Us rules are governing

    direction_known: Whether the code confidently figured out which location is departure versus arrival.

    touches_us: A known or assumed endpoint is a US airport. Never inferred from
    "not EU/UK" (that is OTHER) or from an unknown endpoint.

    needs_clarification: The passenger asked about their own disrupted trip and the departure
    region is unknown, or a non-US departure's destination is unknown (it decides whether US
    DOT governs). The pipeline asks for the flight number and whatever else is missing,
    rather than guess.
    """

    scope: tuple[str, ...]           # retrievable: regimes the route touches or may touch
    governing: tuple[str, ...]       # actually applies: guaranteed a context slot
    origin_region: str | None        # "US" | "EU" | "UK" | "OTHER" | None if unknown
    destination_region: str | None
    direction_known: bool
    touches_us: bool                 # a known/assumed US endpoint; False for intra-Europe
    needs_clarification: bool = False
    origin_place: tuple[str, str] | None = None        # (text, region) as named
    destination_place: tuple[str, str] | None = None
    endpoints: tuple[tuple[str, str], ...] = ()        # named with no direction, in order
    ambiguous_places: tuple[str, ...] = ()             # "Manchester": UK or elsewhere?
    unknown_places: tuple[str, ...] = ()               # not in any place table
    domestic: bool = False                             # "my domestic flight"
    # Airport codes from flight data that data/airport_regions.json does not list.
    # Unknown, not OTHER: they are treated as unknown endpoints (user decision, 2026-09-16).
    unknown_airport_codes: tuple[str, ...] = ()


def region_of(code: str | None) -> str | None:
    """
    Region of an airport code, from data/airport_regions.json: "US", "EU", "UK"
    or "OTHER" (a listed airport outside all three).

    None when no code is supplied OR the table does not list the code. The table
    names every airport it knows, OTHER ones included, so a missing code is
    genuinely unknown: it is an unknown endpoint (US DOT retrievable, never
    governing on its account, and never assumed to be US), not a known OTHER
    airport where no regime applies. (User decision 2026-09-16; before, a
    missing code was OTHER.)
    """
    if not code:
        return None
    return _AIRPORT_REGIONS.get(code.strip().upper())


def is_known_airport(code: str | None) -> bool:
    return bool(code) and code.strip().upper() in _AIRPORT_REGIONS


_region_of = region_of  # kept for existing callers


def resolve_regimes(
    origin_region: str | None,
    destination_region: str | None,
    mentioned: tuple[str, ...] | set[str] = (),
    unplaced: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """
    The applicability matrix, as a pure function: (scope, governing, touches_us).

    Regions are "US" | "EU" | "UK" | "OTHER", or None when unknown.

    `unplaced` holds the regions of places named as an end of the flight with
    no direction ("sat on the tarmac at Chicago Midway", "my Dublin–Chicago
    flight"). They count only while an endpoint is unknown. A US one touches
    the US whichever end it is, so it brings US DOT into `governing`; an EU/UK
    one does not, because EU261/UK261 turn on which end was the departure.

    EU261/UK261 attach to these US carriers only on DEPARTURE, so only the
    origin adds a foreign regime to `governing`. US DOT governs when the known
    or assumed route explicitly touches a US airport; OTHER does not count.

    An unknown endpoint is not the US, but it may be: US DOT is then put in
    `scope` (retrievable) and kept out of `governing`, so the refund rules are
    not lost from retrieval before the route is settled. "Paris -> ?" governs
    under EU261 with US DOT retrievable; "DXB -> CDG" has no US DOT at all.
    """
    endpoints = (origin_region, destination_region)
    open_end = any(r not in _KNOWN for r in endpoints)
    touches_us = "US" in endpoints or (open_end and "US" in unplaced)
    scope = set(mentioned) & set(_ORDER)
    scope |= {r for r in endpoints if r in _ORDER}
    governing: set[str] = set()
    if origin_region in _EUROPE:
        governing.add(origin_region)
    if touches_us:
        # A flight DEPARTING outside the US, EU and UK is never settled as US DOT,
        # even into the US (user's decision, 2026-09-22): "you are not sure; DOT may
        # apply; see the related government website". US law stays retrievable so
        # the answer can cite what it covers; pipeline.route_law_note says the rest.
        if origin_region != "OTHER":
            governing.add("US")
        scope.add("US")
    elif open_end:
        scope.add("US")
    return (
        tuple(j for j in _ORDER if j in scope),
        tuple(j for j in _ORDER if j in governing),
        touches_us,
    )


# --------------------------------------------------------------------------
# Finding places
# --------------------------------------------------------------------------


def _clean_words(phrase: str) -> list[str]:
    return [re.sub(r"(['’]s)?[^\w'’]*$", "", w) for w in phrase.split()]


def _find_places(text: str) -> list[Place]:
    """Every place named in the text, known or not, in order, with its role."""
    found: list[Place] = []
    for pattern, region, is_country in _PLACE_TABLE:
        for m in pattern.finditer(text):
            if m.group(1).lower() in _COMMON_WORD_PLACES and not m.group(1)[0].isupper():
                continue   # "a nice agent", "split the fare"
            found.append(Place(m.group(1), region, m.start(1), m.end(1), country=is_country))
    for m in _AMBIGUOUS_RE.finditer(text):
        found.append(Place(m.group(1), "AMBIGUOUS", m.start(1), m.end(1)))
    for m in _IATA_TOKEN_RE.finditer(text):
        code = m.group(1)
        if code in TEXT_AIRPORT_CODES and code in _AIRPORT_REGIONS:
            found.append(Place(code, region_of(code), m.start(1), m.end(1)))

    # Longest match wins where two overlap ("New York" over "York").
    found.sort(key=lambda p: (p.start, -(p.end - p.start)))
    known: list[Place] = []
    for p in found:
        if not known or p.start >= known[-1].end:
            known.append(p)

    # Places that are in no table, spotted by where they sit in the sentence.
    unknown: list[Place] = []
    for pattern in _PLACE_SLOT_RES:
        for m in pattern.finditer(text):
            s, e = m.start(1), m.end(1)
            if any(k.start < e and s < k.end for k in known + unknown):
                continue
            words = _clean_words(m.group(1))
            if not words or words[0] in _NOT_A_PLACE or any(ch.isdigit() for ch in m.group(1)):
                continue
            name = " ".join(w for w in words if w)
            unknown.append(Place(name, "UNKNOWN", s, s + len(m.group(1).rstrip(".,;:!?"))))

    places = sorted(known + unknown, key=lambda p: p.start)
    places = _merge_qualified(text, places)
    return [
        Place(p.text, p.region, p.start, p.end, p.country, _role_of_mention(text, p.start, p.end))
        for p in places
    ]


def _merge_qualified(text: str, places: list[Place]) -> list[Place]:
    """
    "Manchester, UK", "Timisoara, Romania", "London Heathrow", "Paris CDG": a
    place followed directly by a qualifier is ONE place. The qualifier settles
    the region of an ambiguous or unknown name.
    """
    out: list[Place] = []
    for p in places:
        prev = out[-1] if out else None
        if prev is not None and re.fullmatch(r"\s*,?\s*\(?\s*", text[prev.end:p.start]):
            if prev.region in ("AMBIGUOUS", "UNKNOWN") and p.region in _KNOWN:
                out[-1] = Place(f"{prev.text}, {p.text}", p.region, prev.start, p.end)
                continue
            if prev.region == p.region and prev.region in _KNOWN and not prev.country:
                out[-1] = Place(text[prev.start:p.end], p.region, prev.start, p.end)
                continue
        out.append(p)
    return out


#Is this particular place the origin or the destination?The nearest preceding direction word wins.
#"from New York to Paris", For the word Paris, the preceding direction words include: from, to.
#But the nearest one is: to. Therefore Paris is the destination
def _role_of_mention(question: str, start: int, end: int) -> str | None:
    """Classify one place mention as the origin or the destination."""
    before = question[max(0, start - 40):start]
    markers = list(_DIRECTION_RE.finditer(before))
    if markers:
        # Nearest marker wins; anything further back describes another place.
        # The marker must also be describing THIS place: in "leaving JFK for
        # Rome", "leaving" belongs to JFK, so Rome gets no role from it.
        gap = before[markers[-1].end():].split()
        if any(w.lower() not in _GAP_WORDS for w in gap):
            markers = []
    if markers:
        word = markers[-1].group(1).split()[0].lower()
        if word in _DEPART_WORDS:
            return "origin"
        if word in _ARRIVE_WORDS:
            return "destination"

    # No preposition in front, but "<place> to ..." makes it the origin. like "Paris to New York"
    if _TRAILING_TO_RE.match(question[end:end + 6]):
        return "origin"
    return None


def _as_pair(p: Place | None) -> tuple[str, str] | None:
    return (p.text, p.region) if p is not None else None


def _other_side(a: str, b: str) -> bool:
    """One place in Europe (EU/UK), the other outside it (US/OTHER)."""
    return (a in _EUROPE) != (b in _EUROPE)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

# It has two possible information sources:
# 1. live flight data ( priority)
#2. customer's natural-language question (and their reply to a clarifying question)

def detect_jurisdictions(
    question: str,
    dep_iata: str | None = None,
    arr_iata: str | None = None,
    *,
    reply_start: int | None = None,
) -> RouteJurisdiction:
    """
    Work out which regimes are retrievable and which actually govern.

    Live flight data is authoritative when present — `dep_iata` alone settles
    the question, since these are non-EU/UK carriers and only departure counts.
    Text parsing is the fallback for questions asked without a flight number.

    The text parser only records a departure the passenger actually STATED. A
    destination alone ("my flight to Paris") does not imply a US departure: a US
    carrier's flight into Paris can leave from London, where UK261 governs.

    `reply_start` marks where the passenger's answer to a clarifying question
    begins in `question`. Places in the reply with no direction word fill the
    blanks in the order they were asked: departure first, then destination. A
    country in the reply ("UK", "Romania") settles an ambiguous or unknown city.
    """
    places = _find_places(question)
    mentioned: set[str] = set()
    for p in places:
        if p.region in _ORDER:
            mentioned.add(p.region)
        elif p.region == "AMBIGUOUS":
            mentioned.add("UK")
        elif p.region == "UNKNOWN":
            mentioned |= {"EU", "UK"}

    def unresolved(p: Place | None) -> bool:
        return p is None or p.region not in _KNOWN

    in_reply = (lambda p: p.start >= reply_start) if reply_start is not None else (lambda p: False)

    origin: Place | None = None
    destination: Place | None = None
    endpoints: list[Place] = []
    reply_places: list[Place] = []
    for p in places:
        if p.role == "origin" and unresolved(origin):
            origin = p
        elif p.role == "destination" and unresolved(destination):
            destination = p
        elif p.role is None and in_reply(p):
            reply_places.append(p)
        elif p.role is None and not (p.country and p.text.lower() in _REGIME_WORDS):
            endpoints.append(p)

    # -- the passenger's reply to a clarifying question ---------------------
    # "The Manchester in England": repeating the unclear city is not a new
    # place. Drop the repeat so the country qualifies the city that was asked about.
    unclear = {t.text.lower() for t in (origin, destination, *endpoints)
               if t is not None and t.region in ("AMBIGUOUS", "UNKNOWN")}
    reply_places = [p for p in reply_places if p.country or p.text.lower() not in unclear]
    for p in reply_places:
        if p.country and p.region in _KNOWN:
            # "UK" / "Romania": which country the unclear city is in.
            target = next((t for t in (origin, destination, *endpoints)
                           if t is not None and not in_reply(t) and t.region not in _KNOWN), None)
            if target is not None:
                fixed = Place(f"{target.text}, {p.text}", p.region, target.start, target.end,
                              role=target.role)
                if target is origin:
                    origin = fixed
                elif target is destination:
                    destination = fixed
                else:
                    endpoints[endpoints.index(target)] = fixed
                continue
            if p.text.lower() in _REGIME_WORDS - {"uk", "us", "usa", "u.s."}:
                continue
        # Otherwise fill the blanks in the order they were asked about.
        if unresolved(origin):
            origin = p
        elif unresolved(destination):
            destination = p

    # -- everything else in the text ------------------------------------------
    # "Canada's rules... my flight from Toronto": a country in the same region
    # as a stated end describes that end, not the other one.
    stated = {p.region for p in (origin, destination) if p is not None and p.region in _KNOWN}
    endpoints = [e for e in endpoints if not (e.country and e.region in stated)]

    domestic = False
    if unresolved(origin):
        known_endpoints = [e for e in endpoints if e.region in _KNOWN]
        if origin is None and destination is not None and destination.region in _KNOWN \
                and len(known_endpoints) == 1 and (
                    _other_side(known_endpoints[0].region, destination.region)
                    or not {known_endpoints[0].region, destination.region} & set(_EUROPE)):
            # Both ends named: "fog at Heathrow delayed my flight to Atlanta",
            # "snowstorm in Minneapolis, my flight to Detroit". The flight goes
            # TO the destination, so the other place is where it left. (Not for
            # two European places: "to Paris, connecting in London" is not a
            # London departure.)
            origin = known_endpoints[0]
            endpoints.remove(origin)
        elif origin is None and _DOMESTIC_RE.search(question) and all(
                p.region == "US" for p in places):
            domestic = True   # a US carrier's domestic flight; no place outside the US named

    # An endpoint that is an unresolved city stays listed so it can be asked about.
    ambiguous = tuple(dict.fromkeys(
        p.text for p in (origin, destination, *endpoints) if p is not None and p.region == "AMBIGUOUS"))
    unknown = tuple(dict.fromkeys(
        p.text for p in (origin, destination, *endpoints) if p is not None and p.region == "UNKNOWN"))

    # -- live data: departure airport decides -----------------
    dep_region = region_of(dep_iata)
    arr_region = region_of(arr_iata)
    for region in (dep_region, arr_region):
        if region in _ORDER:
            mentioned.add(region)

    unplaced: tuple[str, ...] = ()
    if dep_region is not None:
        origin_region, destination_region = dep_region, arr_region
    elif domestic:
        origin_region = destination_region = "US"   # "domestic" on a US carrier
    else:
        origin_region = origin.region if origin is not None and origin.region in _KNOWN else None
        destination_region = (destination.region if destination is not None
                              and destination.region in _KNOWN else None)
        unplaced = tuple(e.region for e in endpoints if e.region in _KNOWN)

    scope, governing, touches_us = resolve_regimes(
        origin_region, destination_region, mentioned, unplaced)

    # Everything hangs on the departure airport. If the passenger is asking
    # about their own disrupted trip and we do not know where it left from, ask.
    # A non-US departure also needs its destination: only a US arrival brings
    # US DOT into `governing` (EU->UK carries no US rules). A US departure is
    # governed by US DOT whatever the destination, so it is not asked.
    needs = (
        dep_region is None
        and bool(_PERSONAL_TRIP_RE.search(question))
        and bool(_DISRUPTION_RE.search(question))
        and (origin_region is None
             or (origin_region != "US" and destination_region is None))
    )

    return RouteJurisdiction(
        scope=scope,
        governing=governing,
        origin_region=origin_region,
        destination_region=destination_region,
        direction_known=origin_region is not None or destination_region is not None,
        touches_us=touches_us,
        needs_clarification=needs,
        origin_place=_as_pair(origin),
        destination_place=_as_pair(destination),
        endpoints=tuple((e.text, e.region) for e in endpoints),
        ambiguous_places=ambiguous if dep_region is None else (),
        unknown_places=unknown if dep_region is None else (),
        domestic=domestic,
        unknown_airport_codes=tuple(c.strip().upper() for c in (dep_iata, arr_iata)
                                    if c and not is_known_airport(c)),
    )


# --------------------------------------------------------------------------
# Assumptions, when the passenger could not say
# --------------------------------------------------------------------------


# The product default when nothing about the route is known: a US carrier's
# domestic flight. The golden builder keys its "unspecified" expectation on it.
ASSUME_DOMESTIC = "Assuming this is a flight within the US."


def display_name(name: str) -> str:
    """'manchester' -> 'Manchester'; codes and mixed case are kept as typed."""
    return " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in name.split(" "))


def _uk_reading(name: str) -> str:
    """'Manchester' -> 'Manchester, UK'."""
    return f"{display_name(name.split(',')[0].strip())}, UK"


def assume_route(route: RouteJurisdiction) -> tuple[RouteJurisdiction, str]:
    """
    Pick a route to answer on after a clarifying question went unanswered, and
    the sentence the answer must open with, so the passenger sees the assumption.

        departure known          "Assuming you're flying from Romania."
        Manchester (unresolved)  "Assuming you're flying from Manchester, UK."
        a place, no direction    the first place named is the departure
        destination only         "Assuming you're flying to London from a US airport."
        nothing                  "Assuming this is a flight within the US."
        unknown city             no guess: the answer covers both cases

    An unknown city is not guessed because either guess is costly: EU/UK means
    promising €600 that may not be owed, elsewhere means withholding it.

    Only the last two rows assume a US airport (an explicit product default for
    these US carriers), so only they put US DOT in `governing` without a named
    US place. A departure with no destination ("from Paris.") is not completed
    with a guess: US DOT stays retrievable in `scope`, not governing.
    """
    if route.domestic and route.origin_place is None:
        return route, ASSUME_DOMESTIC

    origin = route.origin_place
    dest = route.destination_place
    rest = list(route.endpoints)
    if origin is None and rest:
        origin = rest.pop(0)
    if dest is None and rest and origin is not None:
        dest = rest.pop(0)

    def named(place):
        text, region = place
        return _uk_reading(text) if region == "AMBIGUOUS" else display_name(text)

    def region(place):
        if place is None:
            return None
        return "UK" if place[1] == "AMBIGUOUS" else place[1] if place[1] in _KNOWN else None

    if origin is not None and origin[1] == "UNKNOWN":
        # Departure unknown: every regime is retrievable, and only a US
        # destination (if one was named) settles anything in `governing`.
        _, governing, touches_us = resolve_regimes(None, region(dest), route.scope)
        conditional = RouteJurisdiction(
            scope=_ORDER, governing=governing, origin_region=None,
            destination_region=region(dest), direction_known=False, touches_us=touches_us,
            origin_place=origin, destination_place=dest,
        )
        return conditional, (
            f"I couldn't confirm which country {display_name(origin[0])} is in, so this answer covers both "
            "cases: a flight departing the EU or UK, and a flight departing anywhere else."
        )

    if origin is not None:
        o_region, d_region = region(origin), region(dest)
        sentence = f"Assuming you're flying from {named(origin)}" + (
            f" to {named(dest)}." if dest is not None else ".")
    elif dest is not None:
        o_region, d_region = "US", region(dest)
        sentence = f"Assuming you're flying to {named(dest)} from a US airport."
    else:
        o_region = d_region = "US"
        sentence = ASSUME_DOMESTIC

    scope, governing, touches_us = resolve_regimes(o_region, d_region, route.scope)
    assumed = RouteJurisdiction(
        scope=scope, governing=governing, origin_region=o_region, destination_region=d_region,
        direction_known=True, touches_us=touches_us,
        origin_place=origin, destination_place=dest,
    )
    return assumed, sentence

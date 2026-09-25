"""
The AirLabs flight status tool.

AirLabs was chosen over another site aviationstack because:
  - 1,000 requests/month (vs 100)
  - 250 requests/minute (vs 1 per 60 seconds)
  - HTTPS on free tier (vs HTTP only)
  - Dedicated /delays endpoint with _fields filtering
  - All critical fields (delayed, dep_delayed, arr_delayed, status) confirmed
    working on the free plan via empirical testing
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .models import FlightStatus

# The carriers this assistant covers: their documents are the whole airline side
# of the corpus. A flight on any other airline gets no AirLabs lookup and no
# airline policy (pipeline.plan_turn). The one definition; import it, don't copy it.
SUPPORTED_AIRLINES = frozenset({"AA", "DL", "UA", "WN"})

# IATA code -> name, to tell a passenger whose flight a number is. Display only:
# whether an airline is supported is decided by SUPPORTED_AIRLINES alone.
AIRLINE_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "AS": "Alaska Airlines", "B6": "JetBlue", "NK": "Spirit Airlines", "F9": "Frontier Airlines",
    "HA": "Hawaiian Airlines", "G4": "Allegiant Air", "SY": "Sun Country Airlines",
    "AC": "Air Canada", "WS": "WestJet", "AM": "Aeromexico",
    "BA": "British Airways", "VS": "Virgin Atlantic", "EI": "Aer Lingus", "AF": "Air France",
    "KL": "KLM", "LH": "Lufthansa", "LX": "Swiss", "OS": "Austrian Airlines",
    "SN": "Brussels Airlines", "IB": "Iberia", "VY": "Vueling", "TP": "TAP Air Portugal",
    "AZ": "ITA Airways", "SK": "SAS", "AY": "Finnair", "LO": "LOT Polish Airlines",
    "FI": "Icelandair", "FR": "Ryanair", "U2": "easyJet", "W6": "Wizz Air",
    "TK": "Turkish Airlines", "EK": "Emirates", "QR": "Qatar Airways", "EY": "Etihad Airways",
    "LY": "El Al", "JL": "Japan Airlines", "NH": "ANA", "QF": "Qantas",
    "SQ": "Singapore Airlines", "CX": "Cathay Pacific",
}

# Matches "UA2402", "BA 117", "aa100". Deliberately strict.
FLIGHT_RE = re.compile(r"\b([A-Z]{2}|[A-Z]\d|\d[A-Z])\s?(\d{1,4})\b")
# Shaped like flight numbers, but are not: regulation names ("UK261", "EU 261")
# and aircraft types ("A350", "B737", "E175"). Without this, "Does UK261 apply?"
# would read as a question about a flight on airline "UK".
_NOT_A_FLIGHT = re.compile(r"(?:EU|EC|UK)261|A[23]\d\d|B7\d7|E1[79]\d")
# ...and "gate B12", "seat 3A 12", "terminal 5" are places in an airport.
_NOT_A_FLIGHT_AFTER = re.compile(r"\b(?:gate|seat|terminal|row|zone|group|concourse|pier|stand)\s*$", re.I)


def extract_flight_number(text: str) -> str | None:
    """A valid-looking flight number, on any airline. Whether that airline is
    supported is a separate question: see airline_of_flight / SUPPORTED_AIRLINES."""
    for m in FLIGHT_RE.finditer(text):
        code = f"{m.group(1)}{m.group(2)}"
        if not m.group(1).isupper() or _NOT_A_FLIGHT.fullmatch(code):
            continue
        if _NOT_A_FLIGHT_AFTER.search(text[:m.start()]):
            continue
        return code
    return None


def airline_of_flight(flight_no: str | None) -> str | None:
    """'BA117' -> 'BA'. The designator prefix, supported or not."""
    return flight_no[:2].upper() if flight_no else None


class QuotaExceeded(RuntimeError):
    pass


class QuotaUnavailable(QuotaExceeded):
    """The quota could not be checked (e.g. the database is down): fail closed."""


@contextlib.contextmanager
def _file_lock(path: Path, timeout_s: float = 10.0):
    """
    An exclusive OS lock on `path` for the duration of the block.

    Serialises the read-modify-write of the quota file across worker processes on
    one host (uvicorn --workers, a Job next to the API). It does NOT coordinate
    separate machines or pods: use the database counter there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as fh:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise QuotaUnavailable(f"could not lock {path} within {timeout_s}s") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _period() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


class FileQuotaCounter:
    """
    Monthly AirLabs call counter in a local file, for a single host.
    """

    backend = "file"

    def __init__(self, path: str = ".airlabs_quota.json", monthly_quota: int = 1000):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.monthly_quota = monthly_quota

    def _load(self) -> dict:
        period = _period()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if data.get("period") == period:
                    return data
            except Exception:
                pass
        return {"period": period, "used": 0}

    def _write(self, data: dict) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    def used(self) -> int:
        return self._load()["used"]

    def remaining(self) -> int:
        return max(0, self.monthly_quota - self.used())

    def increment(self, n: int = 1) -> None:
        with _file_lock(self.lock_path):
            data = self._load()
            data["used"] += n
            self._write(data)

    def try_reserve(self, reserve: int) -> bool:
        """Count one call if that leaves more than `reserve` calls; atomic on this host."""
        with _file_lock(self.lock_path):
            data = self._load()
            if self.monthly_quota - data["used"] <= reserve:
                return False
            data["used"] += 1
            self._write(data)
            return True


class StoreQuotaCounter:
    """
    Monthly AirLabs call counter in the database, shared by every replica.

    The check and the increment are one SQL statement, so two pods can never both
    take the last call. When the database cannot be reached the counter fails
    CLOSED (QuotaUnavailable): no live call is made without a recorded reservation.
    """

    backend = "database"

    def __init__(self, store, monthly_quota: int = 1000):
        self.store = store
        self.monthly_quota = monthly_quota

    def used(self) -> int:
        try:
            return self.store.quota_used(_period())
        except Exception as e:
            raise QuotaUnavailable(f"quota counter unavailable: {e}") from e

    def remaining(self) -> int:
        return max(0, self.monthly_quota - self.used())

    def try_reserve(self, reserve: int) -> bool:
        try:
            return self.store.quota_try_reserve(_period(), self.monthly_quota - reserve)
        except Exception as e:
            raise QuotaUnavailable(f"quota counter unavailable: {e}") from e


class AirLabsTool:
    """
    QUOTA ACCOUNTING: a unit is reserved BEFORE every live request and never given
    back, so a request that times out or fails still counts. Once the request has
    left this process AirLabs may have counted it, and over-counting only makes the
    guard more conservative. The reserve (AIRLABS_RESERVE) is never spent.

    CIRCUIT BREAKER: three consecutive failures - a transport error, an HTTP error,
    or an error payload in a 200 response - open the circuit for five minutes.
    """

    FAILURES_TO_OPEN = 3
    OPEN_SECONDS = 300

    def __init__(self, settings, fixtures_dir: str = "data/fixtures/flights",
                 quota_path: str = ".airlabs_quota.json", quota=None):
        self.s = settings
        self.fixtures = Path(fixtures_dir)
        self.quota = quota or FileQuotaCounter(quota_path, settings.airlabs_monthly_quota)
        self._cache: dict[str, tuple[float, FlightStatus]] = {}
        self._failures = 0
        self._open_until = 0.0

    # -- fixtures ----------------------------------------------------------
    def _fixture(self, flight_iata: str) -> FlightStatus | None:
        p = self.fixtures / f"{flight_iata.upper()}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        fs = self._parse_flight(data, flight_iata, from_cache=True)
        # A recording from an earlier date: labelled so it is never presented as live.
        return replace(fs, source="fixture") if fs else None

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _parse_flight(payload: dict, flight_iata: str, from_cache: bool = False) -> FlightStatus | None:
        """Parse an AirLabs /flight response into our normalised model."""
        resp = payload.get("response")
        if not resp:
            return None
        # /flight returns a single object, not an array
        r = resp if isinstance(resp, dict) else resp[0] if isinstance(resp, list) and resp else None
        if not r:
            return None
        return FlightStatus(
            flight_iata=(r.get("flight_iata") or flight_iata).upper(),
            airline_iata=r.get("airline_iata"),
            status=r.get("status"),
            dep_iata=r.get("dep_iata"),
            dep_time=r.get("dep_time"),
            dep_estimated=r.get("dep_estimated"),
            dep_delayed=r.get("dep_delayed"),
            arr_iata=r.get("arr_iata"),
            arr_time=r.get("arr_time"),
            arr_estimated=r.get("arr_estimated"),
            arr_delayed=r.get("arr_delayed"),
            delayed=r.get("delayed"),
            fetched_at=datetime.now(UTC).isoformat(),
            from_cache=from_cache,
            source="cache" if from_cache else "live",
            operating_airline_iata=r.get("cs_airline_iata"),
            operating_flight_iata=r.get("cs_flight_iata"),
        )

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.FAILURES_TO_OPEN:
            self._open_until = time.time() + self.OPEN_SECONDS

    # -- main entry point --------------------------------------------------
    def get_flight(self, flight_iata: str, allow_live: bool = True) -> FlightStatus | None:
        """
        Resolution order, cheapest and safest first:
            in-memory cache -> recorded fixture (only if use_flight_fixtures)
            -> live API (allow_live, key, circuit breaker and quota permitting)
        """
        key = flight_iata.upper()

        # Not one of the carriers this assistant covers: never spend a lookup
        # (or a unit of the monthly quota) on it. plan_turn() does not ask; this
        # guards any other caller.
        if airline_of_flight(key) not in SUPPORTED_AIRLINES:
            return None

        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.s.airlabs_cache_ttl_s:
            # A copy: callers must not be able to mutate the cached record.
            return replace(hit[1], from_cache=True, source="cache")

        if self.s.use_flight_fixtures:
            fx = self._fixture(key)
            if fx is not None:
                return fx

        if not allow_live or not self.s.airlabs_key:
            return None

        if time.time() < self._open_until:
            return None  # circuit open

        if not self.quota.try_reserve(self.s.airlabs_reserve):
            raise QuotaExceeded(
                f"AirLabs quota guard: {self.quota.remaining()} calls left, "
                f"reserve is {self.s.airlabs_reserve}. Refusing live call."
            )

        try:
            with httpx.Client(timeout=20.0) as c:
                r = c.get(
                    f"{self.s.airlabs_base}/flight",
                    params={
                        "api_key": self.s.airlabs_key,
                        "flight_iata": key,
                    },
                )
                r.raise_for_status()
                payload = r.json()
        except Exception:
            self._record_failure()
            return None
        if not isinstance(payload, dict) or "error" in payload:
            self._record_failure()
            return None
        self._failures = 0

        fs = self._parse_flight(payload, key)
        if fs:
            self._cache[key] = (time.time(), fs)
            return replace(fs)
        return None


def build_tool(settings, store=None):
    """
    The AirLabs tool, counting quota in the database when the store supports it
    (shared across replicas and restarts), else in a local file (one host only).
    """
    quota = None
    if store is not None and hasattr(store, "quota_try_reserve"):
        quota = StoreQuotaCounter(store, settings.airlabs_monthly_quota)
    return AirLabsTool(
        settings,
        fixtures_dir=os.environ.get("FLIGHT_FIXTURES", "data/fixtures/flights"),
        quota=quota,
    )

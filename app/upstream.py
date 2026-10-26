"""What the kurdishtts.com endpoints answered: counts, for `/health`.

Every free and paid request is recorded once, by outcome: ok, 429, 401/403
(forbidden), another 4xx, 5xx, or no answer at all (timeout, connection
error). `/health` shows the last hour and the counts since the service
started, the last 429's time and its rate-limit headers, the current run of
free 429s in a row, and warning codes when the pattern looks like the
endpoint is blocking us rather than pacing us. We pace under the measured
limit (`app/lanes.py`), so any of these warnings is unusual.

Counts only: never a text, never a key.
"""
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable

FREE = "free"
PAID = "paid"
ENDPOINTS = (FREE, PAID)
OUTCOMES = ("requests", "ok", "rate_limited", "forbidden", "other_4xx", "server_error", "no_answer")

HOUR_S = 3600.0

# A share of 429s above this, over at least FREE_429_MIN_REQUESTS free
# requests in the hour, is more than the odd 429 a burst of taps can cause:
# the bucket keeps warming under the limit, so a fifth of requests refused
# means the endpoint's window shrank or it is refusing us.
FREE_429_RATE_THRESHOLD = 0.20
FREE_429_MIN_REQUESTS = 10
# One warm job's backoff is six requests (the first and five retries), so five
# in a row is a whole job refused end to end: not pacing, a wall.
FREE_429_STREAK_THRESHOLD = 5


def _outcome(status: int | None) -> str:
    if status is None:
        return "no_answer"
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "forbidden"
    if 400 <= status < 500:
        return "other_4xx"
    if status >= 500:
        return "server_error"
    return "ok"


def _empty() -> dict:
    return {endpoint: dict.fromkeys(OUTCOMES, 0) for endpoint in ENDPOINTS}


class Upstream:
    def __init__(self, clock: Callable = time.monotonic, wall: Callable = time.time):
        self.clock = clock
        self.wall = wall
        self.lock = threading.Lock()
        self.recent: deque = deque()  # (monotonic time, endpoint, outcome)
        self.totals = _empty()
        self.free_streak = 0
        self.last_429: dict | None = None

    def record(self, endpoint: str, status: int | None, headers: dict | None = None) -> None:
        """One request's answer; `status` None when none came (timeout, connection)."""
        outcome = _outcome(status)
        with self.lock:
            self.recent.append((self.clock(), endpoint, outcome))
            self.totals[endpoint]["requests"] += 1
            self.totals[endpoint][outcome] += 1
            if endpoint == FREE:
                self.free_streak = self.free_streak + 1 if outcome == "rate_limited" else 0
            if outcome == "rate_limited":
                stamp = datetime.fromtimestamp(self.wall(), timezone.utc).isoformat(timespec="seconds")
                self.last_429 = {"endpoint": endpoint, "at": stamp, "headers": dict(headers or {})}

    def _last_hour(self) -> dict:
        """Called with the lock held."""
        cutoff = self.clock() - HOUR_S
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()
        counts = _empty()
        for _, endpoint, outcome in self.recent:
            counts[endpoint]["requests"] += 1
            counts[endpoint][outcome] += 1
        return counts

    def streak(self) -> int:
        with self.lock:
            return self.free_streak

    def warnings(self) -> list:
        with self.lock:
            return _warnings(self._last_hour(), self.free_streak)

    def health(self) -> dict:
        with self.lock:
            hour = self._last_hour()
            warnings = _warnings(hour, self.free_streak)
            return {
                "last_hour": hour,
                "since_start": {endpoint: dict(counts) for endpoint, counts in self.totals.items()},
                "free_429_streak": self.free_streak,
                "last_429": dict(self.last_429) if self.last_429 else None,
                "warning": warnings,
            }


def _warnings(hour: dict, streak: int) -> list:
    free, paid = hour[FREE], hour[PAID]
    codes = []
    if free["requests"] >= FREE_429_MIN_REQUESTS and free["rate_limited"] > FREE_429_RATE_THRESHOLD * free["requests"]:
        codes.append("free_429_rate")
    if streak >= FREE_429_STREAK_THRESHOLD:
        codes.append("free_429_streak")
    if free["forbidden"]:
        codes.append("free_forbidden")
    if paid["requests"] - paid["ok"] - paid["no_answer"]:
        codes.append("paid_errors")
    return codes


UPSTREAM = Upstream()

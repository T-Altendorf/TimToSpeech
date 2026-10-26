"""The night lane: bulk warming on the free endpoint while he sleeps.

A list of texts (`app/night_list.py`, sent by `POST /warm/bulk`) is
synthesised one at a time, in the default voice, by one worker, and only
inside the nightly window (`NIGHT_WINDOW`, `NIGHT_TZ`). It is the lowest
lane and always yields:

- free endpoint only (`lanes.free_only`): a 429 is waited out and the text
  asked again later, never sent to the paid endpoint; a text over the free
  limit is skipped and counted;
- it starts nothing while a tap or warm job is queued or running, and only
  when the bucket holds enough tokens to leave one for the phone;
- it pauses on a run of 429s (`free_429_streak`) and stops until the next
  window on a 401/403 (`free_forbidden`), both from `app/upstream.py`.

A tap or warm request for the text it is on joins its job (`lanes.Jobs`);
a tap makes that job a tap, which may then go paid like any tap.
"""
import os
import threading
import time
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import alignment, lanes, upstream, voices
from .config import CACHE_DIR, NIGHT_TZ, NIGHT_WINDOW, log
from .night_list import CACHED, DONE, FAILED, TOO_LONG, NightList
from .tts_api import FREE_CHAR_LIMIT, RateLimited
from .utils import AUDIO_VERSION, get_cache_path, text_hash

ALWAYS = "always"
DEFAULT_WINDOW = "22:00-07:00"
DEFAULT_TZ = "Europe/Berlin"

# The bucket holds 2 (`lanes.FREE_BUCKET_CAPACITY`); a night job starts only
# when it is full, so after its one token a tap or warm request still has one.
NIGHT_MIN_TOKENS = 2
# How often an idle or paused worker looks again.
NIGHT_POLL_S = 5.0
# After a night job's own 429 backoff (5 to 60 s, `FREE_RETRY_DELAYS_S`) was
# refused end to end, five more minutes before the next request.
NIGHT_RATE_LIMITED_WAIT_S = 300.0
# A run of `upstream.FREE_429_STREAK_THRESHOLD` 429s pauses the lane this long.
NIGHT_STREAK_PAUSE_S = 1800.0
# One log line per this many clips done.
NIGHT_LOG_EVERY = 50


def parse_window(spec: str) -> tuple | None:
    """`HH:MM-HH:MM` as (start, end) minutes after midnight; `always` is None."""
    if spec.strip().lower() == ALWAYS:
        return None
    start, end = spec.split("-")
    return _minutes(start), _minutes(end)


def _minutes(clock: str) -> int:
    hours, minutes = clock.strip().split(":")
    value = int(hours) * 60 + int(minutes)
    if not (0 <= int(hours) < 24 and 0 <= int(minutes) < 60):
        raise ValueError(f"not a time of day: {clock}")
    return value


def in_window(window: tuple | None, now: datetime) -> bool:
    """Start inclusive, end exclusive; a window whose end is before its start
    runs across midnight; equal ends mean the whole day."""
    if window is None:
        return True
    start, end = window
    minute = now.hour * 60 + now.minute
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


def _configured() -> tuple:
    """(window, its spec, tz) from the environment; a bad value logs and
    falls back to the default rather than stopping the service."""
    try:
        window, spec = parse_window(NIGHT_WINDOW), NIGHT_WINDOW
    except ValueError:
        log(f"✗ NIGHT_WINDOW '{NIGHT_WINDOW}' unreadable, using {DEFAULT_WINDOW}")
        window, spec = parse_window(DEFAULT_WINDOW), DEFAULT_WINDOW
    try:
        tz = ZoneInfo(NIGHT_TZ)
    except (ZoneInfoNotFoundError, ValueError):
        log(f"✗ NIGHT_TZ '{NIGHT_TZ}' unknown, using {DEFAULT_TZ}")
        tz = ZoneInfo(DEFAULT_TZ)
    return window, spec, tz


def is_cached(text: str) -> bool:
    """A clip of the current audio version, with its manifest, in the default voice."""
    voice = voices.DEFAULT_VOICE
    return get_cache_path(text, voice).exists() and alignment.exists(text_hash(text), AUDIO_VERSION, voice)


class NightLane:
    def __init__(self, items: NightList, window: tuple | None, spec: str, tz, synth: Callable | None = None,
                 now: Callable | None = None, clock: Callable = time.monotonic, sleep: Callable = time.sleep,
                 stats: upstream.Upstream | None = None, jobs: lanes.Jobs | None = None,
                 bucket: lanes.TokenBucket | None = None):
        self.items, self.window, self.spec, self.tz = items, window, spec, tz
        self.synth = synth  # synth(job_id, text, path) -> bool; RateLimited propagates
        self.now = now or (lambda: datetime.now(self.tz))
        self.clock, self.sleep = clock, sleep
        self.stats = stats or upstream.UPSTREAM
        self.jobs = jobs or lanes.JOBS
        self.bucket = bucket or lanes.FREE_BUCKET
        self.was_open: bool | None = None
        self.forbidden = False
        self.paused_until = 0.0
        self.pause_kind: str | None = None
        self.streak_seen = 0
        self.done_count = 0
        self.paused_reason: str | None = None
        self.thread: threading.Thread | None = None
        self.start_lock = threading.Lock()

    def _window_open(self) -> bool:
        open_now = in_window(self.window, self.now())
        if open_now != self.was_open:
            if self.was_open is not None or open_now:
                log(f"Night window {'opened' if open_now else 'closed'} ({self.spec})")
            self.was_open = open_now
            if not open_now:
                self.forbidden = False
        return open_now

    def _blocked(self) -> str | None:
        """Why no night job may start now, or None."""
        if not self._window_open():
            return "outside_window"
        warnings = self.stats.warnings()
        if "free_forbidden" in warnings:
            self.forbidden = True
        elif self.window is None:
            self.forbidden = False  # `always` has no next window: wait out the hour
        if self.forbidden:
            return "free_forbidden"
        if self.clock() < self.paused_until:
            return self.pause_kind
        return self._streak()

    def _streak(self) -> str | None:
        streak = self.stats.streak()
        if streak < self.streak_seen:
            self.streak_seen = 0
        if streak >= upstream.FREE_429_STREAK_THRESHOLD and streak > self.streak_seen:
            self.streak_seen = streak
            self._pause("free_429_streak", NIGHT_STREAK_PAUSE_S)
            return "free_429_streak"
        return None

    def _pause(self, kind: str, seconds: float) -> None:
        self.pause_kind, self.paused_until = kind, self.clock() + seconds
        log(f"Night lane paused {int(seconds)}s: {kind}")

    def tick(self) -> str | None:
        """One step: runs one text and returns None, or returns why it did not."""
        self.paused_reason = self._blocked() or self._waiting()
        if self.paused_reason:
            return self.paused_reason
        generation, index, text = self.items.next_pending()
        self._process(generation, index, text)
        return None

    def _waiting(self) -> str | None:
        if self.items.next_pending() is None:
            return "empty"
        if self.jobs.busy():
            return "busy"
        if self.bucket.available() < NIGHT_MIN_TOKENS:
            return "bucket"
        return None

    def _process(self, generation: int, index: int, text: str) -> None:
        if len(text) > FREE_CHAR_LIMIT:
            self.items.mark(generation, index, TOO_LONG)
        elif is_cached(text):
            self.items.mark(generation, index, CACHED)
        else:
            outcome = self._synthesise(text)
            if outcome == "rate_limited":
                self._pause("rate_limited", NIGHT_RATE_LIMITED_WAIT_S)
            elif outcome == "failed" and "free_forbidden" not in self.stats.warnings():
                self.items.mark(generation, index, FAILED)
            elif outcome == "done":
                self.items.mark(generation, index, DONE)
                self._count_done()

    def _synthesise(self, text: str) -> str:
        """done, failed, rate_limited, or joined (a tap or warm job had it first)."""
        outcome = {}

        def work(job_id: str) -> None:
            try:
                ok = self.synth(job_id, text, get_cache_path(text, voices.DEFAULT_VOICE))
                outcome["value"] = "done" if ok else "failed"
            except RateLimited:
                outcome["value"] = "rate_limited"

        job_id = f"night-{os.urandom(6).hex()}"
        if not self.jobs.run_here((text, voices.DEFAULT_VOICE), job_id, work):
            return "joined"
        return outcome.get("value", "failed")

    def _count_done(self) -> None:
        self.done_count += 1
        if self.done_count % NIGHT_LOG_EVERY == 0:
            counts = self.items.counts()
            log(f"Night lane: {self.done_count} clips done this run, {counts['remaining']} remaining")

    def health(self) -> dict:
        counts = self.items.counts()
        counts["in_window"] = in_window(self.window, self.now())
        counts["paused_reason"] = self.paused_reason
        return counts

    def progress(self) -> dict:
        """`GET /warm/bulk`: the counts and the window, never a text."""
        return {**self.health(), "window": self.spec, "tz": str(self.tz)}

    def start(self) -> None:
        with self.start_lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._loop, name="night", daemon=True)
                self.thread.start()

    def _loop(self) -> None:
        while True:
            try:
                reason = self.tick()
            except Exception as error:
                log(f"✗ Night lane step raised: {error}")
                reason = "error"
            if reason:
                self.sleep(NIGHT_POLL_S)


def _build() -> NightLane:
    window, spec, tz = _configured()
    return NightLane(NightList(CACHE_DIR / "night-queue.json"), window, spec, tz)


LANE = _build()

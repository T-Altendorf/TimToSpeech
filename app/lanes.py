"""Two lanes for synthesis jobs: a clip the learner taps, and background warming.

The free endpoint rate-limits by address (see `FREE_RETRY_DELAYS_S` in
`app/tts_api.py`), so a clip he taps to hear must never wait behind the clips
warmed ahead of him (2026-10-26):

- `warm` (the default, and what an old phone build sends): one queue, a small
  fixed pool of workers, every free request paced by the token bucket below,
  a 429 waited out with the usual backoff before the paid fallback.
- `tap`: runs at once, never queued and never waiting for a token; one free
  try, and a 429 goes straight to the paid endpoint.

One job per (text, voice) while it is queued or running: a second request
gets the same job id. A tap on a queued warm job takes it out of the queue
and runs it now; a tap on a running one sets its urgency event, which wakes
any backoff or bucket wait and turns the rest of the job into a tap.

The job's urgency event reaches the synthesis code through a context
variable (`urgency()`), so the chunk functions keep their signatures; chunk
threads get it through `carried`. No event at all (a direct call, a test)
means unpaced, exactly as before the lanes.

A third lane, `night` (`app/night.py`), runs bulk warming inside a nightly
window on the free endpoint only: its jobs carry a night flag beside the
urgency event, and `free_only()` tells the synthesis code never to fall to
the paid endpoint. A tap on a night job makes it a tap like any other.
"""
import contextvars
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

from .config import log

TAP = "tap"
WARM = "warm"
NIGHT = "night"

# Workers that drain the warm queue. Each job may still split into parallel
# chunks of its own; the bucket, not this number, keeps the free endpoint calm.
WARM_WORKERS = 2

# The free endpoint answered 429 after about ten requests in two minutes
# (measured 2026-10-02). The bucket stays just under that: a burst of 2, then
# one token every 16 s, so no 120 s window sees more than 2 + 120 / 16 = 9.5
# paced requests. A tap takes a token when one is there (so warming backs off)
# but never waits for one, so taps can push a window past this.
FREE_BUCKET_CAPACITY = 2
FREE_REFILL_INTERVAL_S = 16.0

_urgency: contextvars.ContextVar = contextvars.ContextVar("tts_job_urgency", default=None)
_night: contextvars.ContextVar = contextvars.ContextVar("tts_job_night", default=False)


def urgency() -> threading.Event | None:
    """The running job's urgency event; None outside a job."""
    return _urgency.get()


def is_urgent(event: threading.Event | None) -> bool:
    return event is not None and event.is_set()


def free_only() -> bool:
    """A night job no tap has reached: never the paid endpoint."""
    return _night.get() and not is_urgent(urgency())


def lane_name(event: threading.Event | None) -> str:
    if event is None:
        return "direct"
    if event.is_set():
        return TAP
    return NIGHT if _night.get() else WARM


@contextmanager
def bound(event: threading.Event | None, night: bool = False):
    """Run the block as the job that owns `event`."""
    token = _urgency.set(event)
    night_token = _night.set(night)
    try:
        yield
    finally:
        _night.reset(night_token)
        _urgency.reset(token)


def carried(fn: Callable) -> Callable:
    """`fn`, run in another thread as the job calling this (chunk workers)."""
    event = urgency()
    night = _night.get()

    def run(*args):
        with bound(event, night):
            return fn(*args)

    return run


class TokenBucket:
    """Free-endpoint pacing for the warm lane."""

    def __init__(self, capacity: int, interval_s: float, clock: Callable = time.monotonic):
        self.capacity = capacity
        self.interval_s = interval_s
        self.clock = clock
        self.tokens = float(capacity)
        self.stamp = clock()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.stamp) / self.interval_s)
        self.stamp = now

    def available(self) -> int:
        with self.lock:
            self._refill()
            return int(self.tokens)

    def _try_take(self) -> float:
        """Takes a token and returns 0, or returns the seconds until one."""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            return (1 - self.tokens) * self.interval_s

    def take(self, event: threading.Event | None) -> None:
        """Before one free request. No event: unpaced. An urgent event takes
        a token if one is there and never waits; otherwise wait for a token,
        woken early if the job turns urgent."""
        if event is None:
            return
        while True:
            wait_s = self._try_take()
            if wait_s == 0 or event.is_set():
                return
            event.wait(wait_s)


FREE_BUCKET = TokenBucket(FREE_BUCKET_CAPACITY, FREE_REFILL_INTERVAL_S)


@dataclass
class Job:
    job_id: str
    key: tuple
    work: Callable  # work(job_id): synthesises and records the result
    urgent: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    state: str = "queued"  # queued, running
    lane: str = WARM  # the lane it was submitted on; a tap only sets `urgent`


class Jobs:
    """The job registry, the warm queue and its workers."""

    def __init__(self, workers: int = WARM_WORKERS):
        self.workers = workers
        self.by_key: dict = {}
        self.queue: deque = deque()
        self.cond = threading.Condition()
        self.started = False

    def submit(self, key: tuple, lane: str, job_id: str, work: Callable) -> tuple:
        """Returns (the job for `key`, whether this call created it)."""
        with self.cond:
            job = self.by_key.get(key)
            if job is not None:
                if lane == TAP:
                    self._promote(job)
                return job, False
            job = Job(job_id, key, work, lane=lane)
            self.by_key[key] = job
            if lane == TAP:
                self._start_now(job)
            else:
                self._enqueue(job)
            return job, True

    def _promote(self, job: Job) -> None:
        """A tap on a job already here. Called with the lock held."""
        job.urgent.set()
        if job.state == "queued":
            self.queue.remove(job)
            log(f"Tap on queued warm job {job.job_id}: running it now")
            self._start_now(job)
        else:
            log(f"Tap on running warm job {job.job_id}: upgraded to tap")

    def _start_now(self, job: Job) -> None:
        job.urgent.set()
        job.state = "running"
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def _enqueue(self, job: Job) -> None:
        if not self.started:
            self.started = True
            for n in range(self.workers):
                threading.Thread(target=self._drain, name=f"warm-{n}", daemon=True).start()
        self.queue.append(job)
        self.cond.notify()

    def _drain(self) -> None:
        while True:
            with self.cond:
                while not self.queue:
                    self.cond.wait()
                job = self.queue.popleft()
                job.state = "running"
            self._run(job)

    def run_here(self, key: tuple, job_id: str, work: Callable) -> bool:
        """Runs a night job in the calling thread, registered so a tap or warm
        request for the same text joins it. False, and nothing run, when a
        job for `key` is already here."""
        with self.cond:
            if key in self.by_key:
                return False
            job = Job(job_id, key, work, state="running", lane=NIGHT)
            self.by_key[key] = job
        self._run(job)
        return True

    def busy(self) -> bool:
        """Whether any tap or warm job is queued or running."""
        with self.cond:
            return bool(self.queue) or any(job.lane != NIGHT for job in self.by_key.values())

    def _run(self, job: Job) -> None:
        try:
            with bound(job.urgent, job.lane == NIGHT):
                job.work(job.job_id)
        except Exception as error:
            log(f"✗ Job {job.job_id} raised: {error}")
        finally:
            with self.cond:
                self.by_key.pop(job.key, None)
            job.done.set()

    def counts(self) -> dict:
        """Counts only, for /health."""
        with self.cond:
            running = sum(1 for job in self.by_key.values() if job.state == "running")
            queued = len(self.queue)
        return {"warm_queued": queued, "running": running, "free_tokens": FREE_BUCKET.available()}


JOBS = Jobs()


def parse_lane(value: str | None) -> str:
    """The `lane` query parameter; missing or unknown is warm."""
    return TAP if (value or "").strip().lower() == TAP else WARM

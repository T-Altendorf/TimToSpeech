"""The night lane (`app/night.py`, `app/night_list.py`): the window, the list
on disk, and a worker that yields to the phone and never asks the paid endpoint.

Run with `python -m unittest` from the repo root.
"""

import json
import sys
import tempfile
import threading
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

if "app.tts_local" not in sys.modules:
    stub = types.ModuleType("app.tts_local")
    stub.load_models = lambda: None
    stub.generate_audio = lambda text, path: False
    stub.model = None
    stub.tokenizer = None
    sys.modules["app.tts_local"] = stub

from app import alignment, lanes, night, upstream, utils, voices  # noqa: E402
from app import tts_api as t  # noqa: E402
from app.night_list import APPEND, REPLACE, NightList  # noqa: E402

BERLIN = ZoneInfo("Europe/Berlin")


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 26, hour, minute, tzinfo=BERLIN)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _Limited:
    status_code = 429
    headers = {"Retry-After": "60"}

    def close(self):
        pass


class _InstantEvent(threading.Event):
    """A night job's event: never set, its waits return at once."""

    def wait(self, timeout=None):
        return False


class WindowTest(unittest.TestCase):
    def test_parses_and_spans_midnight(self):
        window = night.parse_window("22:00-07:00")
        self.assertEqual(window, (22 * 60, 7 * 60))
        self.assertTrue(night.in_window(window, _at(22)))
        self.assertTrue(night.in_window(window, _at(3, 30)))
        self.assertFalse(night.in_window(window, _at(7)))
        self.assertFalse(night.in_window(window, _at(21, 59)))

    def test_a_day_window(self):
        window = night.parse_window("09:30-17:00")
        self.assertTrue(night.in_window(window, _at(9, 30)))
        self.assertFalse(night.in_window(window, _at(17)))
        self.assertFalse(night.in_window(window, _at(3)))

    def test_always(self):
        self.assertIsNone(night.parse_window("always"))
        self.assertTrue(night.in_window(None, _at(12)))

    def test_bad_specs_raise(self):
        for spec in ("22-07", "25:00-07:00", "nonsense", "22:00"):
            with self.assertRaises(ValueError):
                night.parse_window(spec)


class _Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for patcher in [
            mock.patch.object(utils, "CACHE_DIR", self.dir),
            mock.patch.object(alignment, "ALIGNMENT_DIR", self.dir / "alignments"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.path = self.dir / "night-queue.json"

    def cache(self, text: str) -> None:
        utils.get_cache_path(text).write_bytes(b"mp3")
        alignment.write(utils.text_hash(text), utils.AUDIO_VERSION, {"text": text}, voices.DEFAULT_VOICE)


class NightListTest(_Tmp):
    def test_append_dedupes_and_skips_cached(self):
        self.cache("cached")
        items = NightList(self.path)
        first = items.add(["a", "b", "a", "cached"], APPEND, night.is_cached)
        self.assertEqual(first, {"listed": 3, "added": 2, "already_listed": 0, "already_cached": 1, "total": 2})
        second = items.add(["b", "c"], APPEND, night.is_cached)
        self.assertEqual(second, {"listed": 2, "added": 1, "already_listed": 1, "already_cached": 0, "total": 3})

    def test_replace_starts_over(self):
        items = NightList(self.path)
        items.add(["a", "b"], APPEND, night.is_cached)
        counts = items.add(["b", "c"], REPLACE, night.is_cached)
        self.assertEqual(counts["added"], 2)
        self.assertEqual(items.texts, ["b", "c"])

    def test_persists_and_resumes(self):
        items = NightList(self.path)
        items.add(["a", "b", "c"], APPEND, night.is_cached)
        generation, index, _ = items.next_pending()
        items.mark(generation, index, "d")
        saved = json.loads(self.path.read_text("utf-8"))
        self.assertEqual(saved, {"texts": ["a", "b", "c"], "states": "d.."})
        resumed = NightList(self.path)
        self.assertEqual(resumed.next_pending()[1:], (1, "b"))
        self.assertEqual(resumed.counts()["done"], 1)
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

    def test_a_mark_for_a_replaced_list_is_dropped(self):
        items = NightList(self.path)
        items.add(["a"], APPEND, night.is_cached)
        generation, index, _ = items.next_pending()
        items.add(["z"], REPLACE, night.is_cached)
        items.mark(generation, index, "d")
        self.assertEqual(items.counts()["remaining"], 1)


class NightLaneTest(_Tmp):
    def setUp(self):
        super().setUp()
        self.clock = _Clock()
        self.stats = upstream.Upstream(clock=self.clock)
        self.jobs = lanes.Jobs()
        self.bucket = lanes.TokenBucket(2, 16.0, clock=self.clock)
        self.synthesised = []
        self.lane = self.make("22:00-07:00", _at(23))

    def make(self, spec: str, now: datetime) -> night.NightLane:
        self.items = NightList(self.path)
        self.now = now
        return night.NightLane(self.items, night.parse_window(spec), spec, BERLIN, self.synth,
                               now=lambda: self.now, clock=self.clock, sleep=lambda s: None,
                               stats=self.stats, jobs=self.jobs, bucket=self.bucket)

    def synth(self, job_id, text, path):
        self.synthesised.append(text)
        return True

    def test_runs_in_the_window_only(self):
        self.items.add(["silav"], APPEND, night.is_cached)
        self.now = _at(12)
        self.assertEqual(self.lane.tick(), "outside_window")
        self.now = _at(23)
        self.assertIsNone(self.lane.tick())
        self.assertEqual(self.synthesised, ["silav"])
        self.assertEqual(self.items.counts()["done"], 1)

    def test_always_runs_at_noon(self):
        lane = self.make("always", _at(12))
        self.items.add(["silav"], APPEND, night.is_cached)
        self.assertIsNone(lane.tick())

    def test_yields_while_a_tap_or_warm_job_is_here(self):
        self.items.add(["silav"], APPEND, night.is_cached)
        self.jobs.by_key[("other", voices.DEFAULT_VOICE)] = lanes.Job("j", ("other",), lambda _: None, state="running")
        self.assertEqual(self.lane.tick(), "busy")
        self.jobs.by_key.clear()
        self.jobs.queue.append(lanes.Job("q", ("q",), lambda _: None))
        self.assertEqual(self.lane.tick(), "busy")
        self.assertEqual(self.synthesised, [])

    def test_leaves_a_token_for_the_phone(self):
        self.items.add(["a", "b"], APPEND, night.is_cached)
        self.bucket.tokens = 1
        self.assertEqual(self.lane.tick(), "bucket")
        self.clock.t += 16
        self.assertIsNone(self.lane.tick())

    def test_skips_cached_and_over_the_free_limit(self):
        self.items.add(["later cached", "x" * (t.FREE_CHAR_LIMIT + 1)], APPEND, night.is_cached)
        self.cache("later cached")
        self.lane.tick()
        self.lane.tick()
        counts = self.items.counts()
        self.assertEqual((counts["skipped"], counts["skipped_reasons"]), (2, {"cached": 1, "too_long": 1}))
        self.assertEqual(self.synthesised, [])

    def test_a_429_keeps_the_text_and_waits(self):
        self.items.add(["silav"], APPEND, night.is_cached)
        self.lane.synth = mock.Mock(side_effect=t.RateLimited("429"))
        self.assertIsNone(self.lane.tick())
        self.assertEqual(self.items.counts()["remaining"], 1)
        self.assertEqual(self.lane.tick(), "rate_limited")
        self.clock.t += night.NIGHT_RATE_LIMITED_WAIT_S
        self.lane.synth = self.synth
        self.assertIsNone(self.lane.tick())
        self.assertEqual(self.items.counts()["done"], 1)

    def test_a_failure_is_counted(self):
        self.items.add(["silav"], APPEND, night.is_cached)
        self.lane.synth = lambda *args: False
        self.lane.tick()
        self.assertEqual(self.items.counts()["failed"], 1)

    def test_pauses_half_an_hour_on_a_429_streak(self):
        self.items.add(["silav"], APPEND, night.is_cached)
        for _ in range(upstream.FREE_429_STREAK_THRESHOLD):
            self.stats.record(upstream.FREE, 429)
        self.assertEqual(self.lane.tick(), "free_429_streak")
        self.clock.t += night.NIGHT_STREAK_PAUSE_S - 1
        self.assertEqual(self.lane.tick(), "free_429_streak")
        self.clock.t += 17  # past the pause, and the bucket refilled
        self.assertIsNone(self.lane.tick())

    def test_stops_until_the_next_window_on_forbidden(self):
        self.items.add(["a", "b"], APPEND, night.is_cached)
        self.stats.record(upstream.FREE, 403)
        self.assertEqual(self.lane.tick(), "free_forbidden")
        self.clock.t += upstream.HOUR_S + 1  # the warning is gone, the window is not over
        self.assertEqual(self.lane.tick(), "free_forbidden")
        self.now = _at(12)
        self.assertEqual(self.lane.tick(), "outside_window")
        self.now = _at(22)
        self.assertIsNone(self.lane.tick())

    def test_a_job_failed_by_a_refusal_stays_listed(self):
        self.items.add(["silav"], APPEND, night.is_cached)

        def refused(*args):
            self.stats.record(upstream.FREE, 401)
            return False

        self.lane.synth = refused
        self.lane.tick()
        self.assertEqual(self.items.counts()["remaining"], 1)

    def test_health_is_counts_only(self):
        self.items.add(["silav"], APPEND, night.is_cached)
        self.lane.tick()
        self.assertEqual(self.lane.health(), {
            "listed": 1, "done": 1, "skipped": 0, "skipped_reasons": {"cached": 0, "too_long": 0},
            "failed": 0, "remaining": 0, "in_window": True, "paused_reason": None,
        })


class NightNeverPaidTest(_Tmp):
    def test_a_night_job_raises_on_429_and_never_asks_paid(self):
        with mock.patch.object(lanes, "FREE_BUCKET", lanes.TokenBucket(1000, 1.0)), \
                mock.patch.object(t.requests, "post", return_value=_Limited()) as post, \
                mock.patch.object(t, "_synthesize_paid") as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                mock.patch.object(t.response_cache, "RESPONSE_CACHE_DIR", self.dir / "responses"), \
                lanes.bound(_InstantEvent(), night=True):
            self.assertTrue(lanes.free_only())
            self.assertEqual(lanes.lane_name(lanes.urgency()), lanes.NIGHT)
            with self.assertRaises(t.RateLimited):
                t.call_kurdish_tts_api("silav", self.dir / "out.mp3")
        paid.assert_not_called()
        self.assertEqual(post.call_count, len(t.FREE_RETRY_DELAYS_S) + 1)

    def test_a_tap_on_a_night_job_may_go_paid(self):
        tapped = threading.Event()
        tapped.set()
        with lanes.bound(tapped, night=True):
            self.assertFalse(lanes.free_only())
            self.assertEqual(lanes.lane_name(tapped), lanes.TAP)


if __name__ == "__main__":
    unittest.main()

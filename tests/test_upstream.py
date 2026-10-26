"""What the endpoints answered (`app/upstream.py`), the blocking warnings,
`/health`'s new fields, and `/warm/bulk`'s key (`app/night_routes.py`).

Run with `python -m unittest` from the repo root.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

if "app.tts_local" not in sys.modules:
    stub = types.ModuleType("app.tts_local")
    stub.load_models = lambda: None
    stub.generate_audio = lambda text, path: False
    stub.model = None
    stub.tokenizer = None
    sys.modules["app.tts_local"] = stub

import requests  # noqa: E402

from app import alignment, config, night, upstream, utils  # noqa: E402
from app import main as app_main  # noqa: E402
from app import tts_api as t  # noqa: E402
from app.night_list import NightList  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class CountersTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.stats = upstream.Upstream(clock=self.clock, wall=lambda: 0.0)

    def test_counts_by_outcome_and_rolls_the_hour(self):
        for status in (200, 429, 401, 404, 503, None):
            self.stats.record(upstream.FREE, status)
        hour = self.stats.health()["last_hour"][upstream.FREE]
        self.assertEqual(hour, {"requests": 6, "ok": 1, "rate_limited": 1, "forbidden": 1,
                                "other_4xx": 1, "server_error": 1, "no_answer": 1})
        self.clock.t += upstream.HOUR_S + 1
        health = self.stats.health()
        self.assertEqual(health["last_hour"][upstream.FREE]["requests"], 0)
        self.assertEqual(health["since_start"][upstream.FREE]["requests"], 6)

    def test_last_429_keeps_its_headers_and_the_streak_resets(self):
        self.stats.record(upstream.FREE, 429, {"Retry-After": "60"})
        self.stats.record(upstream.FREE, 429)
        self.assertEqual(self.stats.streak(), 2)
        self.stats.record(upstream.FREE, 200)
        health = self.stats.health()
        self.assertEqual(health["free_429_streak"], 0)
        self.assertEqual(health["last_429"]["at"], "1970-01-01T00:00:00+00:00")
        self.assertEqual(health["last_429"]["headers"], {})

    def test_no_warning_for_the_odd_429(self):
        for _ in range(9):
            self.stats.record(upstream.FREE, 200)
        self.stats.record(upstream.FREE, 429)
        self.assertEqual(self.stats.warnings(), [])

    def test_free_429_rate(self):
        for status in [200] * 7 + [429, 200, 429, 200, 429]:
            self.stats.record(upstream.FREE, status)
        self.assertEqual(self.stats.warnings(), ["free_429_rate"])

    def test_free_429_streak(self):
        for _ in range(upstream.FREE_429_STREAK_THRESHOLD):
            self.stats.record(upstream.FREE, 429)
        self.assertIn("free_429_streak", self.stats.warnings())

    def test_free_forbidden_lasts_an_hour(self):
        self.stats.record(upstream.FREE, 403)
        self.assertEqual(self.stats.warnings(), ["free_forbidden"])
        self.clock.t += upstream.HOUR_S + 1
        self.assertEqual(self.stats.warnings(), [])

    def test_paid_errors(self):
        self.stats.record(upstream.PAID, 200)
        self.stats.record(upstream.PAID, None)
        self.assertEqual(self.stats.warnings(), [])
        self.stats.record(upstream.PAID, 500)
        self.assertEqual(self.stats.warnings(), ["paid_errors"])


class RecordingTest(unittest.TestCase):
    def test_the_free_and_paid_requests_are_counted(self):
        stats = upstream.Upstream()
        limited = mock.Mock(status_code=429, headers={"Retry-After": "9", "Content-Type": "x"})
        with mock.patch.object(upstream, "UPSTREAM", stats), \
                mock.patch.object(t.requests, "post", side_effect=[limited, requests.exceptions.Timeout()]):
            t._post_counted(upstream.FREE, t.FREE_URL)
            with self.assertRaises(requests.exceptions.Timeout):
                t._post_counted(upstream.PAID, t.PAID_URL)
        health = stats.health()
        self.assertEqual(health["last_429"]["headers"], {"Retry-After": "9"})
        self.assertEqual(health["since_start"][upstream.PAID]["no_answer"], 1)


class _Routes(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lane = night.NightLane(NightList(Path(tmp.name) / "night-queue.json"),
                                    night.parse_window("always"), "always", night.ZoneInfo(night.DEFAULT_TZ))
        self.stats = upstream.Upstream()
        for patcher in [
            mock.patch.object(night, "LANE", self.lane),
            mock.patch.object(self.lane, "start"),
            mock.patch.object(upstream, "UPSTREAM", self.stats),
            mock.patch.object(utils, "CACHE_DIR", Path(tmp.name)),
            mock.patch.object(alignment, "ALIGNMENT_DIR", Path(tmp.name) / "alignments"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app_main.app.test_client()


class BulkRouteTest(_Routes):
    def post(self, body, key="right"):
        return self.client.post("/warm/bulk", json=body, headers={"Authorization": f"Bearer {key}"})

    def test_503_without_a_key(self):
        with mock.patch.object(config, "WARM_BULK_KEY", ""):
            self.assertEqual(self.post({"texts": ["a"]}).status_code, 503)

    def test_401_with_a_wrong_or_missing_key(self):
        with mock.patch.object(config, "WARM_BULK_KEY", "right"):
            self.assertEqual(self.post({"texts": ["a"]}, key="wrong").status_code, 401)
            self.assertEqual(self.client.get("/warm/bulk").status_code, 401)

    def test_200_lists_and_counts(self):
        with mock.patch.object(config, "WARM_BULK_KEY", "right"):
            response = self.post({"texts": ["a", " a ", "b", ""]})
            again = self.post({"texts": ["b", "c"], "mode": "append"})
            replaced = self.post({"texts": ["c"], "mode": "replace"})
            progress = self.client.get("/warm/bulk", headers={"Authorization": "Bearer right"}).get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"listed": 2, "added": 2, "already_listed": 0,
                                               "already_cached": 0, "total": 2})
        self.assertEqual(again.get_json()["already_listed"], 1)
        self.assertEqual(replaced.get_json()["total"], 1)
        self.assertEqual((progress["listed"], progress["window"]), (1, "always"))
        self.lane.start.assert_called()

    def test_bad_bodies(self):
        with mock.patch.object(config, "WARM_BULK_KEY", "right"):
            self.assertEqual(self.post({"texts": "a"}).status_code, 400)
            self.assertEqual(self.post({"texts": [1]}).status_code, 400)
            self.assertEqual(self.post({"texts": ["a"], "mode": "merge"}).status_code, 400)
            with mock.patch("app.night_routes.MAX_BULK_TEXTS", 1):
                self.assertEqual(self.post({"texts": ["a", "b"]}).status_code, 413)


class HealthTest(_Routes):
    def test_night_and_upstream_without_texts(self):
        self.lane.items.add(["silav"], "append", night.is_cached)
        self.stats.record(upstream.FREE, 403)
        body = self.client.get("/health").get_json()
        self.assertEqual(body["status"], "healthy")
        self.assertTrue(body["attention"])
        self.assertEqual(body["upstream"]["warning"], ["free_forbidden"])
        self.assertLessEqual({"listed", "done", "skipped", "failed", "remaining", "in_window", "paused_reason"},
                             set(body["night"]))
        self.assertNotIn("silav", str(body))

    def test_no_attention_when_calm(self):
        self.assertFalse(self.client.get("/health").get_json()["attention"])


if __name__ == "__main__":
    unittest.main()

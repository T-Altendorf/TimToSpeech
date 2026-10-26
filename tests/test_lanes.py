"""The tap and warm lanes (`app/lanes.py`): a clip he taps never waits behind
warming; warm requests share one paced queue; one job per text and voice.

Run with `python -m unittest` from the repo root.
"""

import base64
import sys
import tempfile
import threading
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

from app import alignment, lanes, response_cache, utils, voices  # noqa: E402
from app import main as app_main  # noqa: E402
from app import tts_api as t  # noqa: E402


class _Limited:
    status_code = 429
    headers = {"Retry-After": "60", "X-RateLimit-Remaining": "0", "Content-Type": "text/plain"}

    def close(self):
        pass


class _Ok:
    status_code = 200

    def raise_for_status(self):
        pass


class _Audio(_Ok):
    """A free answer with 50 ms of silence in its SSE stream."""

    def iter_lines(self):
        pcm = base64.b64encode(b"\x00\x00" * 1102).decode()
        return iter([('data: {"type": "speech.audio.delta", "audio": "%s"}' % pcm).encode(), b"data: [DONE]"])


V5 = "studio_elder_m"


class _RecordingEvent(threading.Event):
    """An urgency event whose waits return at once and are recorded; a tap can
    be made to arrive during the first wait."""

    def __init__(self, tap_during_wait: bool = False):
        super().__init__()
        self.waits = []
        self.tap_during_wait = tap_during_wait

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.tap_during_wait:
            self.set()
        return self.is_set()


def _paid():
    return t.PaidAudio(t.AudioSegment.silent(duration=50), [], None, None)


class _Unpaced(unittest.TestCase):
    """The shared bucket replaced by one that never runs dry."""

    def setUp(self):
        patcher = mock.patch.object(lanes, "FREE_BUCKET", lanes.TokenBucket(1000, 1.0))
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in [
            mock.patch.object(utils, "CACHE_DIR", Path(tmp.name)),
            mock.patch.object(alignment, "ALIGNMENT_DIR", Path(tmp.name) / "alignments"),
            mock.patch.object(response_cache, "RESPONSE_CACHE_DIR", Path(tmp.name) / "responses"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)


class FreeRequestLaneTest(_Unpaced):
    def test_a_tap_goes_paid_on_the_first_429_without_sleeping(self):
        tap = threading.Event()
        tap.set()
        with mock.patch.object(t.requests, "post", return_value=_Limited()) as post, \
                mock.patch.object(t.time, "sleep") as sleep, \
                mock.patch.object(t, "_synthesize_paid", return_value=_paid()) as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(tap):
            chunk = t._synthesize_chunk(0, "silav")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        paid.assert_called_once_with("silav", voices.DEFAULT_VOICE)
        self.assertEqual(chunk.endpoint, "paid")

    def test_a_tap_without_a_key_sits_out_the_backoff_then_raises(self):
        tap = threading.Event()
        tap.set()
        with mock.patch.object(t.requests, "post", return_value=_Limited()), \
                mock.patch.object(t.time, "sleep") as sleep, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", None), \
                lanes.bound(tap):
            with self.assertRaises(t.RateLimited):
                t._free_chunk(0, "silav", voices.DEFAULT_VOICE)
        self.assertEqual(sleep.call_count, len(t.FREE_RETRY_DELAYS_S))

    def test_a_warm_429_backs_off_as_before(self):
        warm = _RecordingEvent()
        replies = [_Limited(), _Limited(), _Ok()]
        with mock.patch.object(t.requests, "post", side_effect=replies) as post, lanes.bound(warm):
            t._post_free({}, hurry=True)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(warm.waits, list(t.FREE_RETRY_DELAYS_S[:2]))

    def test_a_warm_job_upgraded_while_asleep_goes_paid_with_no_more_sleeps(self):
        upgraded = _RecordingEvent(tap_during_wait=True)
        with mock.patch.object(t.requests, "post", return_value=_Limited()) as post, \
                mock.patch.object(t, "_synthesize_paid", return_value=_paid()) as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(upgraded):
            chunk = t._free_chunk(0, "silav", voices.DEFAULT_VOICE)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(upgraded.waits, [t.FREE_RETRY_DELAYS_S[0]])
        paid.assert_called_once()
        self.assertEqual(chunk.endpoint, "paid")

    def test_a_tap_on_a_v5_voice_sits_out_the_backoff_and_plays(self):
        tap = threading.Event()
        tap.set()
        replies = [_Limited(), _Limited(), _Audio()]
        with mock.patch.object(t.requests, "post", side_effect=replies) as post, \
                mock.patch.object(t.time, "sleep") as sleep, \
                mock.patch.object(t, "_synthesize_paid") as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(tap):
            chunk = t._free_chunk(0, "silav", V5)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], list(t.FREE_RETRY_DELAYS_S[:2]))
        paid.assert_not_called()
        self.assertEqual(chunk.endpoint, "free")
        self.assertGreater(len(chunk.audio), 0)

    def test_a_tap_on_a_v5_voice_still_raises_after_the_whole_backoff(self):
        tap = threading.Event()
        tap.set()
        with mock.patch.object(t.requests, "post", return_value=_Limited()) as post, \
                mock.patch.object(t.time, "sleep") as sleep, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(tap):
            with self.assertRaises(t.RateLimited):
                t._free_chunk(0, "silav", V5)
        self.assertEqual(post.call_count, len(t.FREE_RETRY_DELAYS_S) + 1)
        self.assertEqual(sleep.call_count, len(t.FREE_RETRY_DELAYS_S))

    def test_an_upgraded_warm_job_on_a_v5_voice_keeps_its_backoff(self):
        upgraded = _RecordingEvent(tap_during_wait=True)
        replies = [_Limited(), _Limited(), _Audio()]
        with mock.patch.object(t.requests, "post", side_effect=replies) as post, \
                mock.patch.object(t.time, "sleep") as sleep, \
                mock.patch.object(t, "_synthesize_paid") as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(upgraded):
            upgraded.set()
            chunk = t._free_chunk(0, "silav", V5)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(upgraded.waits, [])
        paid.assert_not_called()
        self.assertEqual(chunk.endpoint, "free")

    def test_a_warm_job_on_a_v5_voice_upgraded_while_asleep_keeps_asking_free(self):
        warm = _RecordingEvent()
        replies = [_Limited(), _Audio()]

        def tap_arrives(delay):
            warm.set()

        with mock.patch.object(t.requests, "post", side_effect=replies) as post, \
                mock.patch.object(t.time, "sleep", side_effect=tap_arrives) as sleep, \
                mock.patch.object(t, "_synthesize_paid") as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(warm):
            chunk = t._free_chunk(0, "silav", V5)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(t.FREE_RETRY_DELAYS_S[0])
        paid.assert_not_called()
        self.assertEqual(chunk.endpoint, "free")

    def test_a_check_retry_in_a_warm_job_still_gives_up_on_one_429(self):
        warm = _RecordingEvent()
        with mock.patch.object(t.requests, "post", return_value=_Limited()) as post, \
                mock.patch.object(t, "_synthesize_paid") as paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"), \
                lanes.bound(warm):
            with self.assertRaises(t.RateLimited):
                t._free_chunk(0, "silav", voices.DEFAULT_VOICE, attempt=1)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(warm.waits, [])
        paid.assert_not_called()

    def test_a_429_logs_its_rate_limit_headers(self):
        headers = t._rate_limit_headers(_Limited())
        self.assertEqual(headers, {"Retry-After": "60", "X-RateLimit-Remaining": "0"})
        self.assertEqual(t._rate_limit_headers(mock.Mock()), {})

    def test_chunk_threads_carry_the_job_urgency(self):
        tap = threading.Event()
        seen = []
        with lanes.bound(tap):
            carried = lanes.carried(lambda: seen.append(lanes.urgency()))
        worker = threading.Thread(target=carried)
        worker.start()
        worker.join()
        self.assertIs(seen[0], tap)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _ClockEvent(threading.Event):
    """Waiting moves the fake clock on by the time waited."""

    def __init__(self, clock: _Clock):
        super().__init__()
        self.clock = clock
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.clock.now += timeout
        return self.is_set()


class TokenBucketTest(unittest.TestCase):
    def test_warm_requests_are_paced_after_the_burst(self):
        clock = _Clock()
        bucket = lanes.TokenBucket(2, 16.0, clock)
        warm = _ClockEvent(clock)
        for _ in range(3):
            bucket.take(warm)
        self.assertEqual(warm.waits, [16.0])

    def test_a_tap_takes_a_token_but_never_waits(self):
        clock = _Clock()
        bucket = lanes.TokenBucket(1, 16.0, clock)
        tap = _ClockEvent(clock)
        tap.set()
        bucket.take(tap)
        self.assertEqual(bucket.available(), 0)
        bucket.take(tap)
        self.assertEqual(tap.waits, [])

    def test_no_job_is_unpaced(self):
        bucket = lanes.TokenBucket(1, 16.0, _Clock())
        bucket.take(None)
        bucket.take(None)
        self.assertEqual(bucket.available(), 1)

    def test_the_constants_stay_under_the_measured_limit(self):
        per_window = lanes.FREE_BUCKET_CAPACITY + 120 / lanes.FREE_REFILL_INTERVAL_S
        self.assertLess(per_window, 10)


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.jobs = lanes.Jobs(workers=1)
        self.release = threading.Event()
        self.calls = []

    def tearDown(self):
        self.release.set()

    def _blocking(self, job_id):
        self.calls.append((job_id, lanes.urgency().is_set()))
        self.release.wait(5)

    def _quick(self, job_id):
        self.calls.append((job_id, lanes.urgency().is_set()))

    def _until_running(self):
        for _ in range(500):
            if self.calls:
                return
            threading.Event().wait(0.01)

    def test_one_job_per_text_and_voice(self):
        first, created = self.jobs.submit(("silav", "v"), lanes.WARM, "a", self._blocking)
        second, again = self.jobs.submit(("silav", "v"), lanes.WARM, "b", self._blocking)
        self.assertTrue(created)
        self.assertFalse(again)
        self.assertIs(first, second)
        self.release.set()
        self.assertTrue(first.done.wait(5))
        self.assertEqual([job_id for job_id, _ in self.calls], ["a"])

    def test_a_tap_takes_a_queued_warm_job_and_runs_it_now(self):
        busy, _ = self.jobs.submit(("one", "v"), lanes.WARM, "a", self._blocking)
        self._until_running()
        queued, _ = self.jobs.submit(("two", "v"), lanes.WARM, "b", self._quick)
        self.assertEqual(self.jobs.counts()["warm_queued"], 1)
        tapped, created = self.jobs.submit(("two", "v"), lanes.TAP, "c", self._quick)
        self.assertFalse(created)
        self.assertIs(tapped, queued)
        self.assertTrue(queued.done.wait(5))
        self.assertFalse(busy.done.is_set())
        self.assertIn(("b", True), self.calls)
        self.assertEqual(self.jobs.counts()["warm_queued"], 0)

    def test_a_tap_upgrades_a_running_warm_job(self):
        running, _ = self.jobs.submit(("one", "v"), lanes.WARM, "a", self._blocking)
        self._until_running()
        self.assertEqual(self.calls, [("a", False)])
        same, created = self.jobs.submit(("one", "v"), lanes.TAP, "b", self._quick)
        self.assertIs(same, running)
        self.assertFalse(created)
        self.assertTrue(running.urgent.is_set())
        self.release.set()
        self.assertTrue(running.done.wait(5))
        self.assertEqual(len(self.calls), 1)


class LaneRouteTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in [
            mock.patch.object(utils, "CACHE_DIR", Path(tmp.name)),
            mock.patch.object(alignment, "ALIGNMENT_DIR", Path(tmp.name) / "alignments"),
            mock.patch.object(lanes, "JOBS", lanes.Jobs()),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app_main.app.test_client()

    def _lane_of(self, query: dict) -> str:
        def fake_generate(text, output_path, voice):
            Path(output_path).write_bytes(b"mp3")
            return True

        with mock.patch.object(app_main, "call_kurdish_tts_api", side_effect=fake_generate), \
                mock.patch.object(lanes.JOBS, "submit", wraps=lanes.JOBS.submit) as submit:
            response = self.client.get("/tts", query_string=query)
        self.assertEqual(response.status_code, 200)
        return submit.call_args.args[1]

    def test_missing_or_unknown_lane_is_warm(self):
        self.assertEqual(self._lane_of({"text": "silav"}), lanes.WARM)
        self.assertEqual(self._lane_of({"text": "rojbaş", "lane": "fast"}), lanes.WARM)

    def test_a_tap_is_a_tap(self):
        self.assertEqual(self._lane_of({"text": "silav", "lane": "tap"}), lanes.TAP)

    def test_health_reports_the_queue_in_counts(self):
        queue = self.client.get("/health").get_json()["queue"]
        self.assertEqual(set(queue), {"warm_queued", "running", "free_tokens"})


if __name__ == "__main__":
    unittest.main()

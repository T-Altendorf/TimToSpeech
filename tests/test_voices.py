"""The voice table (kmj-daily backlog P50, 2026-10-02): the default voice keeps
every key it had, any other voice gets its own, an unknown voice is refused,
the upstream request carries the voice, and a 429 is retried within bounds.

Stubs out `app.tts_local` (torch/transformers) before importing `app.main`,
so this stays a fast, offline unit test.

Run with `python -m unittest` from the repo root.
"""

import base64
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

if "app.tts_local" not in sys.modules:
    stub = types.ModuleType("app.tts_local")
    stub.load_models = lambda: None
    stub.generate_audio = lambda text, path: False
    stub.model = None
    stub.tokenizer = None
    sys.modules["app.tts_local"] = stub

from app import alignment, response_cache, utils, voices  # noqa: E402
from app import main as app_main  # noqa: E402
from app import tts_api as t  # noqa: E402

ELDER = "studio_elder_m"


def _sse_body(ms: int = 50) -> bytes:
    pcm = np.zeros(22050 * ms // 1000, dtype=np.int16).tobytes()
    delta = '{"type": "speech.audio.delta", "audio": "%s"}' % base64.b64encode(pcm).decode()
    return f"data: {delta}\ndata: [DONE]".encode("utf-8")


class _FakeResponse:
    def __init__(self, status_code: int = 200, body: bytes = b""):
        self.status_code = status_code
        self._lines = body.split(b"\n")
        self.content = body

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        pass

    def json(self):
        return json.loads(self.content)


def _ok():
    return _FakeResponse(200, _sse_body())


def _limited():
    return _FakeResponse(429)


class _TempCache(unittest.TestCase):
    """Every cache directory pointed at a fresh temp folder."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.cache = Path(self.tmpdir.name)
        for patcher in [
            mock.patch.object(utils, "CACHE_DIR", self.cache),
            mock.patch.object(alignment, "ALIGNMENT_DIR", self.cache / "alignments"),
            mock.patch.object(response_cache, "RESPONSE_CACHE_DIR", self.cache / "responses"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app_main.app.test_client()


class KeysTest(_TempCache):
    def test_default_voice_keys_are_exactly_as_before(self):
        sha = hashlib.sha256("silav".encode()).hexdigest()
        self.assertEqual(utils.get_cache_path("silav").name, f"{sha}.a{utils.AUDIO_VERSION}.mp3")
        self.assertEqual(
            utils.get_cache_path("silav", voices.DEFAULT_VOICE).name, f"{sha}.a{utils.AUDIO_VERSION}.mp3"
        )
        self.assertEqual(alignment.path_for(sha, 6).name, f"{sha}.a6.json")
        before = hashlib.sha256("kurmanji/kurmanji_236\nsilav".encode()).hexdigest()
        self.assertEqual(response_cache.key("silav", "free", voices.variant(voices.DEFAULT_VOICE)), f"{before}.free")

    def test_another_voice_has_its_own_keys(self):
        sha = hashlib.sha256("silav".encode()).hexdigest()
        clip = utils.get_cache_path("silav", ELDER)
        self.assertEqual(clip.name, f"{sha}.{ELDER}.a{utils.AUDIO_VERSION}.mp3")
        self.assertNotEqual(clip, utils.get_cache_path("silav"))
        self.assertEqual(alignment.path_for(sha, 6, ELDER).name, f"{sha}.{ELDER}.a6.json")
        self.assertEqual(voices.variant(ELDER), f"kurmanji/{ELDER}")
        self.assertNotEqual(
            response_cache.key("silav", "free", voices.variant(ELDER)),
            response_cache.key("silav", "free", voices.variant(voices.DEFAULT_VOICE)),
        )

    def test_prune_and_counts_cover_both_shapes(self):
        keep = [utils.get_cache_path("a"), utils.get_cache_path("a", ELDER)]
        stale = [self.cache / f"x.a{utils.AUDIO_VERSION - 1}.mp3", self.cache / f"x.{ELDER}.a{utils.AUDIO_VERSION - 1}.mp3"]
        for path in [*keep, *stale]:
            path.write_bytes(b"x")
        self.assertEqual(utils.cache_counts(), {"current": 2, "stale": 2})
        self.assertEqual(utils.prune_stale_cache(), 2)
        self.assertTrue(all(path.exists() for path in keep))


class TableTest(unittest.TestCase):
    def test_the_allow_list(self):
        v5 = [
            "studio_elder_m", "studio_docu_m", "studio_host_f", "studio_teacher_f",
            "studio_docu_f", "cast_f3", "cast_f2", "studio_host_m", "cast_m2", "kurmanji_270",
        ]
        self.assertEqual(voices.allowed(), ["kurmanji_236", *v5])
        self.assertEqual(voices.model_version("kurmanji_236"), "v4")
        for voice in v5:
            self.assertEqual(voices.model_version(voice), "v5")
            self.assertFalse(voices.accepts_paid(voice))


class RoutesTest(_TempCache):
    def test_unknown_voice_is_a_400_with_the_allowed_ids(self):
        for route in ["/tts", "/alignment"]:
            response = self.client.get(route, query_string={"text": "silav", "voice": "anyone"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["allowed_voices"], voices.allowed())

    def test_alignment_reads_that_voices_manifest(self):
        alignment.write(utils.text_hash("silav"), utils.AUDIO_VERSION, {"who": ELDER}, ELDER)
        response = self.client.get("/alignment", query_string={"text": "silav", "voice": ELDER})
        self.assertEqual(response.get_json(), {"who": ELDER})
        self.assertEqual(self.client.get("/alignment", query_string={"text": "silav"}).status_code, 404)

    def test_force_regen_deletes_only_that_voices_clip(self):
        default_clip, elder_clip = utils.get_cache_path("silav"), utils.get_cache_path("silav", ELDER)
        default_clip.write_bytes(b"default")
        elder_clip.write_bytes(b"elder")

        def fake_generate(text, output_path, voice):
            Path(output_path).write_bytes(f"new {voice}".encode())
            return True

        with mock.patch.object(app_main, "call_kurdish_tts_api", side_effect=fake_generate) as generate:
            response = self.client.get("/tts", query_string={"text": "silav", "voice": ELDER, "force_regen": "1"})
        self.assertEqual(generate.call_args.args[2], ELDER)
        self.assertEqual(response.data, f"new {ELDER}".encode())
        self.assertEqual(default_clip.read_bytes(), b"default")

    def test_health_lists_the_voices(self):
        body = self.client.get("/health").get_json()
        self.assertEqual(body["default_voice"], "kurmanji_236")
        self.assertEqual(body["voices"][ELDER], "v5")


class FallbackTest(unittest.TestCase):
    def _run(self, voice):
        with mock.patch.object(app_main, "call_kurdish_tts_api", return_value=False), \
                mock.patch.object(app_main, "generate_audio", return_value=True) as local:
            app_main.generate_tts_async("job", "silav", "/nowhere.mp3", True, voice)
        return local, app_main.processing_jobs["job"]["status"]

    def test_no_local_fallback_for_another_voice(self):
        local, status = self._run(ELDER)
        local.assert_not_called()
        self.assertEqual(status, "failed")

    def test_default_voice_keeps_the_local_fallback(self):
        local, status = self._run(voices.DEFAULT_VOICE)
        local.assert_called_once()
        self.assertEqual(status, "completed")


class UpstreamTest(_TempCache):
    def test_free_payload_carries_voice_and_model_version(self):
        with mock.patch.object(t.requests, "post", return_value=_ok()) as post:
            t._synthesize_free("silav", ELDER)
        payload = post.call_args.kwargs["json"]
        self.assertEqual((payload["voice"], payload["model_version"]), (ELDER, "v5"))

    def test_paid_payload_carries_voice_and_model_version(self):
        body = b'{"audio": "AAAAAA==", "timestamps": [], "generation": {"collapsed": false}}'
        with mock.patch.object(t.requests, "post", return_value=_FakeResponse(200, body)) as post, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            t._synthesize_paid("silav", ELDER)
        payload = post.call_args.kwargs["json"]
        self.assertEqual((payload["speaker_id"], payload["model_version"]), (ELDER, "v5"))

    def test_a_429_is_retried_and_then_succeeds(self):
        with mock.patch.object(t.requests, "post", side_effect=[_limited(), _limited(), _ok()]) as post, \
                mock.patch.object(t.time, "sleep") as sleep:
            t._synthesize_free("silav", ELDER)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], list(t.FREE_RETRY_DELAYS_S[:2]))

    def test_the_retries_are_bounded(self):
        with mock.patch.object(t.requests, "post", side_effect=lambda *a, **k: _limited()) as post, \
                mock.patch.object(t.time, "sleep") as sleep:
            with self.assertRaises(t.RateLimited):
                t._synthesize_free("silav", ELDER)
        self.assertEqual(post.call_count, len(t.FREE_RETRY_DELAYS_S) + 1)
        self.assertEqual(sleep.call_count, len(t.FREE_RETRY_DELAYS_S))

    def test_default_voice_falls_through_to_paid_when_still_limited(self):
        paid = t.PaidAudio(t.AudioSegment.silent(duration=50), [], None, None)
        with mock.patch.object(t, "_synthesize_free", side_effect=t.RateLimited("429")), \
                mock.patch.object(t, "_synthesize_paid", return_value=paid) as synthesize_paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            chunk = t._synthesize_chunk(0, "silav")
        synthesize_paid.assert_called_once_with("silav", voices.DEFAULT_VOICE)
        self.assertEqual(chunk.endpoint, "paid")

    def test_a_voice_the_paid_endpoint_refuses_never_goes_there(self):
        with mock.patch.object(t, "_synthesize_free", side_effect=t.RateLimited("429")), \
                mock.patch.object(t, "_synthesize_paid") as synthesize_paid, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            with self.assertRaises(t.RateLimited):
                t._synthesize_chunk(0, "silav", ELDER)
        synthesize_paid.assert_not_called()

    def test_a_long_sentence_stays_on_the_free_endpoint_for_a_free_only_voice(self):
        sentence = " ".join(["peyv"] * 60)  # 299 chars, one sentence
        self.assertEqual(len(t.split_text(sentence)), 1)
        chunks = t.split_text(sentence, sentence_limit=t.FREE_CHAR_LIMIT)
        self.assertTrue(all(len(chunk) <= t.FREE_CHAR_LIMIT for chunk in chunks))
        self.assertEqual(" ".join(chunks), sentence)


def _tone(ms: int) -> "t.AudioSegment":
    rate = 22050
    n = rate * ms // 1000
    pcm = (8000 * np.sin(2 * np.pi * 180 * np.arange(n) / rate)).astype(np.int16).tobytes()
    return t.AudioSegment(data=pcm, sample_width=2, frame_rate=rate, channels=1)


def _silence(ms: int) -> "t.AudioSegment":
    return t.AudioSegment.silent(duration=ms, frame_rate=22050)


class V5TrimTest(unittest.TestCase):
    """The shape measured on studio_docu_m: a 20ms burst at 65ms with only
    100ms of silence after it, and stray sound 1.2s after the last word."""

    def setUp(self):
        self.segment = _silence(65) + _tone(20) + _silence(100) + _tone(600) + _silence(1200) + _tone(300)
        self.words = [{"word": "silav", "start": 0.0, "end": 0.8}]

    def test_default_voice_trims_byte_for_byte_as_before(self):
        self.assertIsNone(t.trim_rule(voices.DEFAULT_VOICE))
        before = t._merge_segments([self.segment])
        now = t._merge_segments([self.segment], t.trim_rule(voices.DEFAULT_VOICE), [self.words])
        self.assertEqual(now.audio.raw_data, before.audio.raw_data)
        self.assertEqual(now.trim_plans, before.trim_plans)
        self.assertLessEqual(before.trim_plans[0][0][0], 65)  # v4 keeps this burst, as it always did

    def test_a_v5_voice_loses_the_head_burst_and_the_stray_tail(self):
        plan = t._trim_plan(self.segment, t.trim_rule(ELDER), self.words)
        self.assertEqual(len(plan), 1)
        self.assertGreater(plan[0][0], 85)  # starts after the burst
        self.assertLess(plan[0][1], 1985)  # ends before the stray sound
        self.assertGreaterEqual(plan[0][1], 785)  # the speech is all there

    def test_no_word_times_skips_the_tail_rule(self):
        plan = t._trim_plan(self.segment, t.trim_rule(ELDER), None)
        self.assertGreater(plan[-1][1], 1985)

    def test_a_short_first_word_is_not_taken_for_the_burst(self):
        # "Tu" on studio_teacher_f: a 95ms first island, 65ms of silence after it.
        segment = _silence(90) + _tone(95) + _silence(65) + _tone(450) + _silence(100)
        plan = t._trim_plan(segment, t.trim_rule(ELDER), [{"word": "yî", "start": 0.48, "end": 0.7}])
        self.assertLessEqual(plan[0][0], 90)


if __name__ == "__main__":
    unittest.main()

"""The alignment check: word times held against the clip's own sound.

A word lit at the wrong moment is worse than none (his word, 2026-10-02):
a chunk that fails has `words: null` and keeps the engine's words under
`rejected_words`. Covers the check itself, the two real bad lines, the
retries at generation, the lazy check, the sweep, the bounded regeneration
of a cast clip and `/health`.

Run with `python -m unittest` from the repo root.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pydub import AudioSegment

if "app.tts_local" not in sys.modules:
    stub = types.ModuleType("app.tts_local")
    stub.load_models = lambda: None
    stub.generate_audio = lambda text, path: False
    stub.model = None
    stub.tokenizer = None
    sys.modules["app.tts_local"] = stub

from app import alignment, alignment_check, alignment_sweep, response_cache, utils, voices  # noqa: E402
from app import main as app_main  # noqa: E402
from app import tts_api as t  # noqa: E402

RATE = 22050
FIXTURES = Path(__file__).parent / "fixtures"
ELDER = "studio_elder_m"


def _word(word: str, start_ms: int, end_ms: int) -> dict:
    return {"word": word, "start_ms": start_ms, "end_ms": end_ms}


def _codes(found: list) -> list:
    return [reason["code"] for reason in found]


class ReasonsTest(unittest.TestCase):
    """Synthetic islands and words on the clip timeline, one reason each."""

    SPAN = (50, 1100)
    ISLANDS = [(60, 1000)]

    def test_a_clean_chunk_passes(self):
        words = [_word("Tu", 50, 200), _word("li", 200, 400), _word("malê", 400, 980)]
        self.assertEqual(alignment_check.reasons(words, self.SPAN, self.ISLANDS), [])

    def test_a_word_under_the_floor(self):
        words = [_word("Tu", 50, 270), _word("ku", 270, 270), _word("dijî", 270, 980)]
        found = alignment_check.reasons(words, self.SPAN, self.ISLANDS)
        self.assertEqual(_codes(found), ["short_word"])
        self.assertEqual(found[0]["word"], "ku")

    def test_a_word_with_no_sound_in_it(self):
        words = [_word("Tu", 50, 400), _word("li", 420, 600), _word("malê", 600, 980)]
        found = alignment_check.reasons(words, self.SPAN, [(60, 410), (610, 1000)])
        self.assertEqual(_codes(found), ["silent_word"])

    def test_sound_after_the_words(self):
        words = [_word("Tu", 50, 300), _word("dijî", 300, 700)]
        found = alignment_check.reasons(words, (50, 1300), [(60, 720), (1050, 1250)])
        self.assertEqual(_codes(found), ["sound_after_words", "words_end_early"])

    def test_a_stop_release_and_a_tick_are_not_sound_after_the_words(self):
        words = [_word("dest", 50, 600)]
        islands = [(60, 590), (680, 700), (1000, 1005)]  # release 80ms on, a 5ms tick 400ms on
        self.assertEqual(alignment_check.reasons(words, (50, 1045), islands), [])

    def test_words_that_end_long_before_the_chunk(self):
        words = [_word("Tu", 50, 300)]
        found = alignment_check.reasons(words, (50, 900), [(60, 860)])
        self.assertEqual(_codes(found), ["words_end_early"])

    def test_no_words_no_reasons(self):
        self.assertEqual(alignment_check.reasons([], self.SPAN, self.ISLANDS), [])


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class RealManifestsTest(unittest.TestCase):
    """Two lines from the live cache whose engine word times are wrong, and a good one."""

    def test_tu_li_ku_diji_fails(self):
        fixture = _fixture("docu_m_tu_li_ku_diji")
        checked = alignment_check.apply(fixture["manifest"], fixture["islands"])
        self.assertFalse(checked["check"]["ok"])
        self.assertEqual(
            sorted(set(_codes(checked["check"]["reasons"]))),
            ["short_word", "sound_after_words", "words_end_early"],
        )
        self.assertTrue(alignment_check.has_stray_tail(checked))

    def test_bi_me_ra_were_fails_on_the_zero_length_word(self):
        fixture = _fixture("docu_m_bi_me_ra_were")
        checked = alignment_check.apply(fixture["manifest"], fixture["islands"])
        self.assertEqual(_codes(checked["check"]["reasons"]), ["short_word"])
        chunk = checked["chunks"][0]
        self.assertIsNone(chunk["words"])
        self.assertEqual([w["word"] for w in chunk["rejected_words"]], ["Bi", "me", "ra", "were."])
        self.assertFalse(alignment_check.has_stray_tail(checked))

    def test_a_good_line_keeps_its_words(self):
        fixture = _fixture("docu_m_good_line")
        checked = alignment_check.apply(fixture["manifest"], fixture["islands"])
        self.assertTrue(checked["check"]["ok"])
        self.assertIsNotNone(checked["chunks"][0]["words"])
        self.assertNotIn("rejected_words", checked["chunks"][0])

    def test_apply_is_idempotent_and_gives_words_back(self):
        fixture = _fixture("docu_m_bi_me_ra_were")
        once = alignment_check.apply(json.loads(json.dumps(fixture["manifest"])), fixture["islands"])
        twice = alignment_check.apply(json.loads(json.dumps(once)), fixture["islands"])
        self.assertEqual(once, twice)
        with mock.patch.object(alignment_check, "MIN_WORD_MS", 0):
            restored = alignment_check.apply(twice, fixture["islands"])
        self.assertTrue(restored["check"]["ok"])
        self.assertEqual(len(restored["chunks"][0]["words"]), 4)


def _tone_chunk() -> AudioSegment:
    n = RATE * 600 // 1000
    tone = 8000 * np.sin(2 * np.pi * 180 * np.arange(n) / RATE)
    data = np.concatenate([np.zeros(RATE // 10), tone, np.zeros(RATE // 5)]).astype(np.int16)
    return AudioSegment(data=data.tobytes(), sample_width=2, frame_rate=RATE, channels=1)


GOOD_WORDS = [{"word": "silav", "start": 0.10, "end": 0.40}, {"word": "heval", "start": 0.40, "end": 0.70}]
BAD_WORDS = [{"word": "silav", "start": 0.10, "end": 0.40}, {"word": "heval", "start": 0.40, "end": 0.40}]


def _chunk(words: list, attempt: int = 0) -> t.ChunkAudio:
    return t.ChunkAudio(0, "silav heval", "free", f"key.r{attempt}", _tone_chunk(), words)


class _TempCache(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.cache = Path(self.tmpdir.name)
        for patcher in [
            mock.patch.object(utils, "CACHE_DIR", self.cache),
            mock.patch.object(alignment, "ALIGNMENT_DIR", self.cache / "alignments"),
            mock.patch.object(response_cache, "RESPONSE_CACHE_DIR", self.cache / "responses"),
            mock.patch.object(alignment_sweep, "SWEEP_PAUSE_S", 0),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app_main.app.test_client()

    def manifest(self, voice: str = voices.DEFAULT_VOICE) -> dict:
        return alignment.read(utils.text_hash("silav heval"), utils.AUDIO_VERSION, voice)


class GenerationTest(_TempCache):
    def _generate(self, side_effect):
        with mock.patch.object(t, "_synthesize_chunk", side_effect=side_effect) as synth:
            ok = t.call_kurdish_tts_api("silav heval", self.cache / "out.mp3")
        self.assertTrue(ok)
        return synth

    def test_a_clean_chunk_is_asked_for_once(self):
        synth = self._generate(lambda i, text, voice, attempt=0: _chunk(GOOD_WORDS, attempt))
        self.assertEqual(synth.call_count, 1)
        self.assertEqual(self.manifest()["check"], {
            "version": alignment_check.VERSION, "ok": True, "reasons": [], "attempts": [1], "regenerated": False,
        })

    def test_the_first_attempt_that_passes_is_kept(self):
        synth = self._generate(lambda i, text, voice, attempt=0: _chunk(GOOD_WORDS if attempt else BAD_WORDS, attempt))
        self.assertEqual(synth.call_count, 2)
        manifest = self.manifest()
        self.assertEqual(manifest["check"]["attempts"], [2])
        self.assertEqual(manifest["chunks"][0]["response_key"], "key.r1")
        self.assertEqual(len(manifest["chunks"][0]["words"]), 2)

    def test_retries_are_bounded_then_the_words_are_withheld(self):
        synth = self._generate(lambda i, text, voice, attempt=0: _chunk(BAD_WORDS, attempt))
        self.assertEqual(synth.call_count, 1 + t.MAX_CHECK_RETRIES)
        manifest = self.manifest()
        self.assertFalse(manifest["check"]["ok"])
        self.assertEqual(manifest["check"]["attempts"], [3])
        chunk = manifest["chunks"][0]
        self.assertIsNone(chunk["words"])
        self.assertEqual(chunk["response_key"], "key.r0")  # the first, on a tie
        self.assertEqual(_codes(chunk["rejected_reasons"]), ["short_word"])
        self.assertTrue((self.cache / "out.mp3").exists())

    def test_a_rate_limited_retry_ends_the_retries(self):
        def synth(i, text, voice, attempt=0):
            if attempt:
                raise t.RateLimited("429")
            return _chunk(BAD_WORDS)

        calls = self._generate(synth)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(self.manifest()["check"]["attempts"], [1])

    def test_a_retry_never_waits_out_a_429_nor_falls_through_to_paid(self):
        limited = mock.Mock(status_code=429)
        with mock.patch.object(t.requests, "post", return_value=limited) as post, \
                mock.patch.object(t.time, "sleep") as sleep, \
                mock.patch.object(t, "_synthesize_paid") as paid:
            with self.assertRaises(t.RateLimited):
                t._free_chunk(0, "silav", voices.DEFAULT_VOICE, attempt=1)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        paid.assert_not_called()

    def test_a_retry_is_cached_under_its_own_key(self):
        self.assertEqual(t._cache_endpoint("free", 0), "free")
        self.assertEqual(t._cache_endpoint("paid_ts", 2), "paid_ts.r2")


def _write_clip(voice: str, words: list) -> None:
    """An mp3 and an unchecked manifest, as every clip cached before the check."""
    merged = t._merge_segments([_tone_chunk()])
    merged.audio.export(str(utils.get_cache_path("silav heval", voice)), format="mp3")
    chunk = {"index": 0, "text": "silav heval", "endpoint": "free", "response_key": "k", "raw_ms": 900,
             "plan": merged.trim_plans[0], "chunk_start_ms": merged.chunk_starts_ms[0], "words": words}
    manifest = alignment.build("silav heval", utils.text_hash("silav heval"), utils.AUDIO_VERSION,
                               voices.variant(voice), len(merged.audio), [chunk])
    alignment.write(utils.text_hash("silav heval"), utils.AUDIO_VERSION, manifest, voice)


class LazyCheckTest(_TempCache):
    def test_an_unchecked_manifest_is_checked_once_then_served(self):
        _write_clip(voices.DEFAULT_VOICE, BAD_WORDS)
        with mock.patch.object(alignment_sweep, "check_one", wraps=alignment_sweep.check_one) as check:
            first = self.client.get("/alignment", query_string={"text": "silav heval"}).get_json()
            second = self.client.get("/alignment", query_string={"text": "silav heval"}).get_json()
        self.assertEqual(check.call_count, 1)
        self.assertEqual(first, second)
        self.assertFalse(first["check"]["ok"])
        self.assertIsNone(first["chunks"][0]["words"])
        self.assertEqual(self.manifest(), first)

    def test_a_manifest_whose_clip_is_gone_is_served_without_words(self):
        _write_clip(voices.DEFAULT_VOICE, GOOD_WORDS)
        utils.get_cache_path("silav heval").unlink()
        served = self.client.get("/alignment", query_string={"text": "silav heval"}).get_json()
        self.assertIsNone(served["chunks"][0]["words"])
        self.assertNotIn("check", self.manifest())  # nothing written: it is checked when it can be

    def test_the_sweep_checks_everything_once_and_yields_to_a_running_one(self):
        _write_clip(voices.DEFAULT_VOICE, GOOD_WORDS)
        _write_clip(ELDER, BAD_WORDS)
        self.assertEqual(alignment_sweep.sweep(), (2, 1))
        self.assertEqual(alignment_sweep.sweep(), (0, 0))
        self.assertEqual(alignment_sweep._count(), {"version": 1, "checked": 2, "failed": 1, "pending": 0})
        with mock.patch.object(alignment_sweep.fcntl, "flock", side_effect=BlockingIOError):
            self.assertEqual(alignment_sweep.sweep(), (0, 0))


STRAY_WORDS = [{"word": "silav", "start": 0.10, "end": 0.15}]  # the tone runs on to 0.70


class RegenerationTest(_TempCache):
    def _request(self, voice: str):
        def generate(text, output_path, voice, regenerated=False):
            Path(output_path).write_bytes(b"new clip")
            return True

        with mock.patch.object(app_main, "call_kurdish_tts_api", side_effect=generate) as call:
            response = self.client.get("/tts", query_string={"text": "silav heval", "voice": voice})
        return call, response

    def test_a_cast_clip_with_stray_sound_is_made_again_once(self):
        _write_clip(ELDER, STRAY_WORDS)
        call, response = self._request(ELDER)
        self.assertEqual(response.data, b"new clip")
        self.assertEqual(call.call_args.kwargs, {"regenerated": True})

    def test_a_regenerated_clip_is_never_made_again(self):
        _write_clip(ELDER, STRAY_WORDS)
        manifest = alignment_sweep.ensure_checked(utils.text_hash("silav heval"), ELDER, self.manifest(ELDER))
        manifest["check"]["regenerated"] = True
        alignment.write(utils.text_hash("silav heval"), utils.AUDIO_VERSION, manifest, ELDER)
        call, response = self._request(ELDER)
        call.assert_not_called()
        self.assertEqual(response.status_code, 200)

    def test_a_default_voice_clip_is_never_made_again_by_the_check(self):
        _write_clip(voices.DEFAULT_VOICE, STRAY_WORDS)
        call, response = self._request(voices.DEFAULT_VOICE)
        call.assert_not_called()
        self.assertEqual(response.status_code, 200)


class HealthTest(_TempCache):
    def test_the_old_fields_stay_and_the_check_is_counted(self):
        alignment_sweep._health_memo.clear()
        _write_clip(ELDER, GOOD_WORDS)
        body = self.client.get("/health").get_json()
        for field in ["status", "audio_version", "cache", "default_voice", "voices", "tts_model_loaded"]:
            self.assertIn(field, body)
        self.assertEqual(body["cache"], {"current": 1, "stale": 0})
        self.assertEqual(body["default_voice"], voices.DEFAULT_VOICE)
        self.assertEqual(body["alignment_check"], {"version": 1, "checked": 0, "failed": 0, "pending": 1})


if __name__ == "__main__":
    unittest.main()

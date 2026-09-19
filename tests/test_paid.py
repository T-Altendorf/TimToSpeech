"""The authenticated endpoint, requested with word timestamps.

Kurdishtts.com's /api/tts-proxy contract: `include_timestamps: true` answers
JSON (base64 PCM, `timestamps`, `sample_rate`, `generation`) instead of a
bare WAV; `format` is ignored once timestamps are requested.

Run with `python -m unittest` from the repo root.
"""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import response_cache
from app import tts_api as t


def _pcm(ms: int, rate: int = 24000) -> bytes:
    n = rate * ms // 1000
    tone = (8000 * np.sin(2 * np.pi * 180 * np.arange(n) / rate)).astype(np.int16)
    return tone.tobytes()


def _paid_body(pcm: bytes, words: list, collapsed: bool = False, sample_rate: int = 24000) -> bytes:
    body = {
        "audio": base64.b64encode(pcm).decode(),
        "timestamps": words,
        "sample_rate": sample_rate,
        "audio_duration": len(pcm) / 2 / sample_rate,
        "generation": {
            "collapsed": collapsed,
            "seed_used": 1,
            "temperature_used": 0.7,
            "retries_used": 0,
            "chunk_count": 1,
        },
    }
    return json.dumps(body).encode("utf-8")


def _sse_body(pcm: bytes) -> bytes:
    delta = '{"type": "speech.audio.delta", "audio": "%s"}' % base64.b64encode(pcm).decode()
    done = '{"type": "speech.audio.done", "usage": {}}'
    return f"data: {delta}\ndata: {done}\ndata: [DONE]".encode("utf-8")


class _FakeJsonResponse:
    def __init__(self, body: bytes):
        self.content = body

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.content)


class _FakeSseResponse:
    def __init__(self, body: bytes):
        self._lines = body.split(b"\n")

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


class PaidTimestampsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.patcher = mock.patch.object(
            response_cache, "RESPONSE_CACHE_DIR", Path(self.tmpdir.name) / "responses"
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_requests_timestamps_and_returns_words(self):
        words = [{"word": "yek", "start": 0.0, "end": 0.1}]
        body = _paid_body(_pcm(300), words)
        with mock.patch.object(t.requests, "post", return_value=_FakeJsonResponse(body)) as post, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            result = t._synthesize_paid("a long sentence over the free limit")

        self.assertEqual(result.words, words)
        self.assertFalse(result.generation["collapsed"])
        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["include_timestamps"])
        self.assertNotIn("format", payload)

    def test_collapsed_generation_raises(self):
        body = _paid_body(_pcm(100), [], collapsed=True)
        with mock.patch.object(t.requests, "post", return_value=_FakeJsonResponse(body)), \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            with self.assertRaises(RuntimeError):
                t._synthesize_paid("a long sentence over the free limit")

    def test_old_wav_cache_under_plain_paid_is_not_used(self):
        response_cache.write("long sentence", "paid", b"RIFF....WAVEfmt ", "audio/wav", t.CACHE_VARIANT)
        body = _paid_body(_pcm(150), [{"word": "du", "start": 0.0, "end": 0.1}])
        with mock.patch.object(t.requests, "post", return_value=_FakeJsonResponse(body)) as post, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            result = t._synthesize_paid("long sentence")
        post.assert_called_once()
        self.assertEqual(len(result.words), 1)

    def test_manifest_carries_paid_words_generation_and_null_quality(self):
        chunk = t.ChunkAudio(
            index=0,
            text="long sentence",
            endpoint="paid",
            response_key="key-0",
            audio=t.AudioSegment(data=_pcm(300), sample_width=2, frame_rate=24000, channels=1),
            words=[{"word": "yek", "start": 0.05, "end": 0.15}],
            generation={"collapsed": False, "seed_used": 1, "temperature_used": 0.7, "retries_used": 0, "chunk_count": 1},
            voice=None,
        )
        merged = t._merge_segments([chunk.audio])
        from app import alignment

        manifest = alignment.build(
            "long sentence", "hash", t.AUDIO_VERSION, t.CACHE_VARIANT, len(merged.audio),
            [{
                "index": 0, "text": chunk.text, "endpoint": chunk.endpoint,
                "response_key": chunk.response_key, "raw_ms": len(chunk.audio),
                "plan": merged.trim_plans[0], "chunk_start_ms": merged.chunk_starts_ms[0],
                "words": chunk.words, "generation": chunk.generation, "voice": chunk.voice,
            }],
        )
        word = manifest["chunks"][0]["words"][0]
        self.assertIsNone(word["alignment_quality"])
        self.assertIsNone(word["probability"])
        self.assertEqual(manifest["chunks"][0]["generation"]["chunk_count"], 1)


class VoiceConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.patcher = mock.patch.object(
            response_cache, "RESPONSE_CACHE_DIR", Path(self.tmpdir.name) / "responses"
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_free_and_paid_request_the_same_voice_and_model(self):
        with mock.patch.object(t.requests, "post", return_value=_FakeSseResponse(_sse_body(_pcm(50, 22050)))) as post:
            t._synthesize_free("test free voice")
        free_payload = post.call_args.kwargs["json"]
        self.assertEqual(free_payload["voice"], t.VOICE)
        self.assertEqual(free_payload["model_version"], t.MODEL_VERSION)

        body = _paid_body(_pcm(50), [])
        with mock.patch.object(t.requests, "post", return_value=_FakeJsonResponse(body)) as post, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            t._synthesize_paid("test paid voice, a long enough sentence to be routed here")
        paid_payload = post.call_args.kwargs["json"]
        self.assertEqual(paid_payload["speaker_id"], t.VOICE)
        self.assertEqual(paid_payload["model_version"], t.MODEL_VERSION)
        self.assertEqual(free_payload["voice"], paid_payload["speaker_id"])
        self.assertEqual(free_payload["model_version"], paid_payload["model_version"])


if __name__ == "__main__":
    unittest.main()

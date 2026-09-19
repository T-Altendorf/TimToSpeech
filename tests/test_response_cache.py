"""A cache hit reuses the full upstream response and never calls out again.

Run with `python -m unittest` from the repo root.
"""

import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import response_cache
from app import tts_api as t


def _sse_body(pcm: bytes) -> bytes:
    """One SSE response shaped like the free endpoint's: a delta then done."""
    delta = '{"type": "speech.audio.delta", "audio": "%s"}' % base64.b64encode(pcm).decode()
    done = '{"type": "speech.audio.done", "usage": {"total_tokens": 3}}'
    return f"data: {delta}\ndata: {done}\ndata: [DONE]".encode("utf-8")


class _FakeResponse:
    def __init__(self, body: bytes):
        self._lines = body.split(b"\n")
        self.content = body

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


class ResponseCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patched_dir = Path(self.tmpdir.name) / "responses"
        self.patcher = mock.patch.object(response_cache, "RESPONSE_CACHE_DIR", patched_dir)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_second_identical_request_skips_upstream(self):
        pcm = b"\x11\x22" * 100
        body = _sse_body(pcm)

        with mock.patch.object(t.requests, "post", return_value=_FakeResponse(body)) as post:
            first = t._synthesize_free("Silav")
            self.assertEqual(post.call_count, 1)

        with mock.patch.object(t.requests, "post") as post:
            second = t._synthesize_free("Silav")
            post.assert_not_called()

        self.assertEqual(first.raw_data, second.raw_data)

    def test_stored_body_is_byte_identical_to_upstream(self):
        body = _sse_body(b"\x33\x44" * 50)
        with mock.patch.object(t.requests, "post", return_value=_FakeResponse(body)):
            t._synthesize_free("Rojbaş")

        stored = response_cache.read("Rojbaş", "free", t.CACHE_VARIANT)
        # The stored body round-trips through str/bytes (utf-8 text), which
        # this payload survives unchanged: no upstream byte is lost.
        self.assertEqual(stored, body)

    def test_paid_cache_hit_needs_no_key(self):
        wav = b"RIFF....WAVEfmt "
        with mock.patch.object(t.requests, "post", return_value=_FakeResponse(wav)) as post, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", "a-key"):
            response_cache.write("long sentence", "paid", wav, "audio/wav")
            self.assertEqual(post.call_count, 0)

        with mock.patch.object(t.requests, "post") as post, \
                mock.patch.object(t, "KURDISH_TTS_API_KEY", ""):
            cached = response_cache.read("long sentence", "paid")
            self.assertEqual(cached, wav)
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()

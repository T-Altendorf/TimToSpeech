"""The /alignment route, and the cache-hit rule that every clip must have a
manifest (2026-09-19): an mp3 without one is not a valid cache entry.

Stubs out `app.tts_local` (torch/transformers) before importing `app.main`,
so this stays a fast, offline unit test of the Flask routes.

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

from app import alignment  # noqa: E402
from app import main as app_main  # noqa: E402
from app import utils  # noqa: E402


class AlignmentRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.patcher = mock.patch.object(alignment, "ALIGNMENT_DIR", Path(self.tmpdir.name) / "alignments")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.client = app_main.app.test_client()

    def test_200_with_a_manifest(self):
        manifest = {"schema": 1, "audio_version": utils.AUDIO_VERSION, "text": "silav", "chunks": []}
        alignment.write(utils.text_hash("silav"), utils.AUDIO_VERSION, manifest)

        response = self.client.get("/alignment", query_string={"text": "silav"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "silav")

    def test_404_without_one(self):
        response = self.client.get("/alignment", query_string={"text": "no such text"})
        self.assertEqual(response.status_code, 404)

    def test_400_without_text(self):
        response = self.client.get("/alignment")
        self.assertEqual(response.status_code, 400)

    def test_reads_a_specific_audio_version(self):
        alignment.write(utils.text_hash("silav"), 3, {"schema": 1, "audio_version": 3, "chunks": []})
        response = self.client.get("/alignment", query_string={"text": "silav", "audio_version": "3"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["audio_version"], 3)
        # the current version has no manifest of its own here
        current = self.client.get("/alignment", query_string={"text": "silav"})
        self.assertEqual(current.status_code, 404)


class ManifestGatedCacheHitTest(unittest.TestCase):
    """Every clip must have a manifest: a cached mp3 without one is not a hit."""

    def setUp(self):
        self.cache_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache_tmp.cleanup)
        self.align_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.align_tmp.cleanup)
        self.patchers = [
            mock.patch.object(utils, "CACHE_DIR", Path(self.cache_tmp.name)),
            mock.patch.object(alignment, "ALIGNMENT_DIR", Path(self.align_tmp.name) / "alignments"),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app_main.app.test_client()

    def test_clip_with_a_manifest_is_served_from_cache(self):
        text = "silav"
        cache_path = utils.get_cache_path(text)
        cache_path.write_bytes(b"already-generated-mp3")
        alignment.write(utils.text_hash(text), utils.AUDIO_VERSION, {"schema": 1, "chunks": []})

        with mock.patch.object(app_main, "call_kurdish_tts_api") as regenerate:
            response = self.client.get("/tts", query_string={"text": text})
        regenerate.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"already-generated-mp3")

    def test_clip_without_a_manifest_is_dropped_and_regenerated(self):
        text = "silav"
        cache_path = utils.get_cache_path(text)
        cache_path.write_bytes(b"orphaned-mp3-no-manifest")

        def fake_generate(text, output_path):
            Path(output_path).write_bytes(b"freshly-generated-mp3")
            return True

        with mock.patch.object(app_main, "call_kurdish_tts_api", side_effect=fake_generate) as regenerate:
            response = self.client.get("/tts", query_string={"text": text})
        regenerate.assert_called_once()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"freshly-generated-mp3")
        self.assertFalse(cache_path.read_bytes() == b"orphaned-mp3-no-manifest")


if __name__ == "__main__":
    unittest.main()

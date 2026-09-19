"""A deploy cleans up after itself: old-version clips go when the service starts."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import alignment, response_cache, utils


class PruneTest(unittest.TestCase):
    def test_only_clips_of_the_current_version_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            current = cache / f"abc.a{utils.AUDIO_VERSION}.mp3"
            stale = [cache / "abc.mp3", cache / f"abc.a{utils.AUDIO_VERSION - 1}.mp3"]
            other = cache / "notes.txt"
            for path in [current, other, *stale]:
                path.write_bytes(b"x")
            with mock.patch.object(utils, "CACHE_DIR", cache):
                self.assertEqual(utils.cache_counts(), {"current": 1, "stale": 2})
                self.assertEqual(utils.prune_stale_cache(), 2)
                self.assertEqual(utils.cache_counts(), {"current": 1, "stale": 0})
                self.assertEqual(utils.prune_stale_cache(), 0)
                self.assertEqual(utils.get_cache_path("text").parent, cache)
            self.assertTrue(current.exists())
            self.assertTrue(other.exists())
            self.assertFalse(any(path.exists() for path in stale))

    def test_manifests_and_response_bodies_are_never_pruned(self):
        # `prune_stale_cache` only globs `*.mp3` at the top level, so a
        # subdirectory file never matches, whatever its own version tag - a
        # phone may still hold an older clip, and its manifest must still
        # answer when asked.
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with mock.patch.object(utils, "CACHE_DIR", cache), \
                    mock.patch.object(alignment, "ALIGNMENT_DIR", cache / "alignments"), \
                    mock.patch.object(response_cache, "RESPONSE_CACHE_DIR", cache / "responses"):
                alignment.write("abc", utils.AUDIO_VERSION - 1, {"schema": 1, "chunks": []})
                response_cache.write("some text", "free", b"raw body", "text/event-stream")
                utils.prune_stale_cache()
                self.assertIsNotNone(alignment.read("abc", utils.AUDIO_VERSION - 1))
                self.assertEqual(response_cache.read("some text", "free"), b"raw body")


if __name__ == "__main__":
    unittest.main()

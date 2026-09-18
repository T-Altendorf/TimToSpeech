"""A deploy cleans up after itself: old-version clips go when the service starts."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import utils


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


if __name__ == "__main__":
    unittest.main()

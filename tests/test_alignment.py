"""A saved clip's manifest: how a raw word time lands on the clip's timeline.

Run with `python -m unittest` from the repo root.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pydub import AudioSegment

from app import alignment
from app import tts_api as t

RATE = 22050


def _tone(ms: int, amplitude: float = 8000.0, hz: float = 180.0) -> np.ndarray:
    n = RATE * ms // 1000
    return amplitude * np.sin(2 * np.pi * hz * np.arange(n) / RATE)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(RATE * ms // 1000)


def _segment(*parts) -> AudioSegment:
    data = np.concatenate(parts).astype(np.int16).tobytes()
    return AudioSegment(data=data, sample_width=2, frame_rate=RATE, channels=1)


def _two_sentence_chunk() -> AudioSegment:
    burst = _tone(15, amplitude=600.0, hz=900.0)
    return _segment(
        _silence(65), burst, _silence(450), _tone(800),
        _silence(960), burst, _silence(480), _tone(700), _silence(90),
    )


class MapRawMsTest(unittest.TestCase):
    def test_inside_a_kept_piece_shifts_linearly(self):
        plan = [(0, 100), (200, 300)]
        starts = [1000, 1100]
        self.assertEqual(alignment.map_raw_ms(90, plan, starts), 1090)

    def test_inside_a_cut_region_clamps_to_the_nearest_edge(self):
        plan = [(0, 100), (200, 300)]
        starts = [1000, 5000]  # not contiguous, so the two edges differ
        # 101 is 1ms past piece 0's end: clamps to piece 0's own clip edge.
        self.assertEqual(alignment.map_raw_ms(101, plan, starts), 1100)
        # 199 is 1ms before piece 1's start: clamps to piece 1's clip edge.
        self.assertEqual(alignment.map_raw_ms(199, plan, starts), 5000)


class WordStraddlingACutTest(unittest.TestCase):
    def test_maps_into_the_right_islands_of_the_merged_clip(self):
        chunk = _two_sentence_chunk()
        plan = t._trim_plan(chunk)
        self.assertEqual(len(plan), 2)
        starts = alignment.piece_starts(0, plan)

        words = [
            {"word": "yek", "start": 0.7, "end": 0.9, "alignment_quality": 0.9, "probability": 0.95},
            {"word": "straddle", "start": 1.9, "end": 2.9, "alignment_quality": 0.5, "probability": 0.6},
            {"word": "sê", "start": 2.95, "end": 3.15, "alignment_quality": 0.9, "probability": 0.9},
        ]
        mapped = alignment._map_words(words, plan, starts)

        merged = t._merge_segments([chunk]).audio
        islands = t._sound_islands(merged)
        self.assertEqual(len(islands), 2)
        frame = t.FRAME_MS

        self.assertGreaterEqual(mapped[0]["start_ms"], islands[0][0] - 2 * frame)
        self.assertLessEqual(mapped[0]["end_ms"], islands[0][1] + 2 * frame)

        # The straddling word: its start belongs to the first island's piece,
        # its end to the second's - the two pieces meet at `starts[1]` on
        # the clip's own timeline (raw and clip timelines are not the same
        # scale, so the boundary is compared in clip ms here, not raw ms).
        boundary = starts[1]
        self.assertLessEqual(mapped[1]["start_ms"], boundary)
        self.assertGreaterEqual(mapped[1]["end_ms"], boundary)

        self.assertGreaterEqual(mapped[2]["start_ms"], islands[1][0] - 2 * frame)
        self.assertLessEqual(mapped[2]["end_ms"], islands[1][1] + 2 * frame)


def _flat_chunk(speech_ms: int) -> AudioSegment:
    """A single speech island, no burst: one trivial trim-plan range."""
    return _segment(_silence(100), _tone(speech_ms), _silence(100))


class ManifestGenerationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.patcher = mock.patch.object(alignment, "ALIGNMENT_DIR", Path(self.tmpdir.name) / "alignments")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.output_path = Path(self.tmpdir.name) / "clip.mp3"

    def _chunk_result(self, index, text, endpoint="free", speech_ms=500, words=None):
        return t.ChunkAudio(
            index=index,
            text=text,
            endpoint=endpoint,
            response_key=f"key-{index}",
            audio=_flat_chunk(speech_ms),
            words=words,
            generation=None,
        )

    def test_manifest_written_on_generation_and_matches_chunk_starts(self):
        chunk1 = self._chunk_result(0, "yek du", words=[{"word": "yek", "start": 0.2, "end": 0.35}])
        chunk2 = self._chunk_result(1, "sê çar", words=[{"word": "sê", "start": 0.2, "end": 0.35}])
        with mock.patch.object(t, "split_text", return_value=["yek du", "sê çar"]), \
                mock.patch.object(t, "_synthesize_chunk", side_effect=[chunk1, chunk2]):
            self.assertTrue(t.call_kurdish_tts_api("yek du. sê çar.", self.output_path))

        manifest = alignment.read(t.text_hash("yek du. sê çar."), t.AUDIO_VERSION)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(len(manifest["chunks"]), 2)
        self.assertIsNotNone(manifest["chunks"][0]["words"])
        # The second chunk's word starts after lead-in plus the first chunk
        # plus the gap - the same place `_merge_segments` puts the chunk.
        second_word_start = manifest["chunks"][1]["words"][0]["start_ms"]
        first_chunk_kept_ms = len(t._trim_silence(_flat_chunk(500)))
        expected_floor = t.LEAD_IN_MS + first_chunk_kept_ms + t.CHUNK_GAP_MS
        self.assertGreaterEqual(second_word_start, expected_floor)

    def test_manifest_not_written_on_failure(self):
        with mock.patch.object(t, "split_text", return_value=["yek"]), \
                mock.patch.object(t, "_synthesize_chunk", side_effect=RuntimeError("boom")):
            self.assertFalse(t.call_kurdish_tts_api("yek", self.output_path))
        self.assertIsNone(alignment.read(t.text_hash("yek"), t.AUDIO_VERSION))

    def test_write_error_does_not_fail_the_call(self):
        chunk = self._chunk_result(0, "yek")
        with mock.patch.object(t, "split_text", return_value=["yek"]), \
                mock.patch.object(t, "_synthesize_chunk", return_value=chunk), \
                mock.patch.object(alignment, "write", side_effect=OSError("disk full")):
            self.assertTrue(t.call_kurdish_tts_api("yek", self.output_path))
        self.assertTrue(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()

"""The join must be silent: no engine artifact, no step, no tick.

Synthetic signals only, shaped like what the engine returns (measured
2026-09-18): a ~15 ms burst some 65 ms in, a long silence, then the speech.
Run with `python -m unittest` from the repo root.
"""

import unittest

import numpy as np
from pydub import AudioSegment

from app import tts_api as t

RATE = 22050


def _tone(ms: int, amplitude: float = 8000.0, hz: float = 180.0) -> np.ndarray:
    n = RATE * ms // 1000
    return amplitude * np.sin(2 * np.pi * hz * np.arange(n) / RATE)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(RATE * ms // 1000)


def _hum(ms: int, amplitude: float = 100.0, hz: float = 2000.0) -> np.ndarray:
    """Quiet room tone: well under ACTIVE_DBFS, but never exactly zero -
    the engine's real "silence" is like this, not digital zero."""
    return _tone(ms, amplitude=amplitude, hz=hz)


def _segment(*parts: np.ndarray, rate: int = RATE) -> AudioSegment:
    data = np.concatenate(parts).astype(np.int16).tobytes()
    return AudioSegment(data=data, sample_width=2, frame_rate=rate, channels=1)


def _engine_chunk(speech_ms: int = 900) -> AudioSegment:
    burst = _tone(15, amplitude=600.0, hz=900.0)
    return _segment(_silence(65), burst, _silence(450), _tone(speech_ms), _silence(90))


def _short_islands(segment: AudioSegment) -> list:
    return [i for i in t._sound_islands(segment) if i[1] - i[0] <= t.ARTIFACT_MAX_MS]


class TrimTest(unittest.TestCase):
    def test_engine_artifact_is_dropped(self):
        trimmed = t._trim_silence(_engine_chunk())
        self.assertEqual(_short_islands(trimmed), [])
        expected = 900 + t.HEAD_PAD_MS + t.TAIL_PAD_MS
        self.assertLessEqual(abs(len(trimmed) - expected), 2 * t.FRAME_MS)

    def test_short_sound_close_to_speech_is_kept(self):
        plosive = _tone(20, amplitude=3000.0, hz=1200.0)
        chunk = _segment(_silence(100), plosive, _silence(80), _tone(600), _silence(100))
        trimmed = t._trim_silence(chunk)
        self.assertGreaterEqual(len(trimmed), 20 + 80 + 600)

    def test_burst_between_sentences_inside_one_clip_is_cut(self):
        burst = _tone(15, amplitude=600.0, hz=900.0)
        chunk = _segment(
            _silence(65), burst, _silence(450), _tone(800),
            _silence(960), burst, _silence(480), _tone(700), _silence(90),
        )
        trimmed = t._trim_silence(chunk)
        self.assertEqual(_short_islands(trimmed), [])
        islands = t._sound_islands(trimmed)
        self.assertEqual(len(islands), 2)
        pause = islands[1][0] - islands[0][1]
        self.assertLessEqual(abs(pause - (960 + t.HEAD_PAD_MS - t.ARTIFACT_CUT_MS)), 3 * t.FRAME_MS)

    def test_burst_cut_fades_instead_of_stepping(self):
        # The engine's own silence carries a little noise, not true zero
        # (measured on a live burst cut, 2026-09-19: a 125-of-32768 step).
        # `_hum` stands in for that noise; a raw splice there still ticks.
        burst = _tone(15, amplitude=600.0, hz=900.0)
        chunk = _segment(
            _hum(100), _tone(600), _hum(960), burst, _hum(480), _tone(700), _hum(90),
        )
        islands = t._sound_islands(chunk)
        bursts = [i for i in range(len(islands)) if t._is_artifact(islands, i)]
        self.assertEqual(len(bursts), 1)
        cursor = max(0, islands[0][0] - t.HEAD_PAD_MS)
        join_ms = islands[bursts[0]][0] - t.ARTIFACT_CUT_MS - cursor
        join_idx = round(join_ms * RATE / 1000)

        trimmed = t._trim_silence(chunk)
        samples = t._samples(trimmed)
        window = samples[join_idx - 10 : join_idx + 10]
        self.assertLess(np.abs(np.diff(window)).max(), 20)

    def test_final_plosive_after_a_short_closure_is_kept(self):
        plosive = _tone(20, amplitude=3000.0, hz=1200.0)
        chunk = _segment(_silence(100), _tone(600), _silence(90), plosive, _silence(400), _tone(500))
        self.assertEqual(len(t._sound_islands(t._trim_silence(chunk))), 3)

    def test_all_silence_chunk_is_left_alone(self):
        chunk = _segment(_silence(300))
        self.assertEqual(len(t._trim_silence(chunk)), len(chunk))


class SealTest(unittest.TestCase):
    def test_edges_reach_zero_without_a_step(self):
        sealed = t._samples(t._seal_edges(_segment(_tone(500, hz=440.0))))
        self.assertEqual(sealed[0], 0)
        self.assertEqual(sealed[-1], 0)
        edge = RATE * 2 // 1000
        self.assertLess(np.abs(np.diff(sealed[:edge])).max(), 200)
        self.assertLess(np.abs(np.diff(sealed[-edge:])).max(), 200)


class MergeTest(unittest.TestCase):
    def test_join_is_true_silence_and_carries_no_artifact(self):
        merged = t._merge_segments([_engine_chunk(900), _engine_chunk(700)])
        self.assertEqual(_short_islands(merged), [])
        islands = t._sound_islands(merged)
        self.assertEqual(len(islands), 2)
        gap_start, gap_end = islands[0][1], islands[1][0]
        self.assertGreaterEqual(gap_end - gap_start, t.CHUNK_GAP_MS)
        samples = t._samples(merged)
        middle = samples[RATE * (gap_start + 60) // 1000 : RATE * (gap_end - 30) // 1000]
        self.assertEqual(np.abs(middle).max(), 0)

    def test_single_chunk_is_cleaned_too(self):
        merged = t._merge_segments([_engine_chunk(800)])
        self.assertEqual(_short_islands(merged), [])
        self.assertEqual(len(t._sound_islands(merged)), 1)

    def test_mixed_rates_join_at_the_higher_rate(self):
        other = _engine_chunk(600).set_frame_rate(24000)
        merged = t._merge_segments([_engine_chunk(600), other])
        self.assertEqual(merged.frame_rate, 24000)
        self.assertEqual(_short_islands(merged), [])


if __name__ == "__main__":
    unittest.main()

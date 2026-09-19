"""Word-for-word timing for a saved clip: what the manifest holds and how.

A manifest is written once, right after a clip is generated
(`tts_api.call_kurdish_tts_api`), and never touched again. The app does not
read it yet (2026-09-19); this exists so a saved clip can still give word for
word timing later. It records, per chunk of the request, how the raw engine
audio was cut down to what the clip plays, so a raw word time - in seconds,
on the engine's own timeline - can always be found again on the clip's
timeline, the PCM before MP3 encoding.

A raw time inside a range the trim kept maps by a straight shift; a raw time
in a cut region (the engine's burst, the silence around it) clamps to the
nearest kept edge.
"""

import json
from pathlib import Path

from .config import CACHE_DIR, log

ALIGNMENT_DIR = CACHE_DIR / "alignments"
SCHEMA = 1


def path_for(text_hash: str, audio_version: int) -> Path:
    return ALIGNMENT_DIR / f"{text_hash}.a{audio_version}.json"


def map_raw_ms(raw_ms: float, plan: list, piece_starts: list) -> int:
    """One raw-timeline ms onto the clip's timeline.

    `plan` is the chunk's kept raw ranges (from `tts_api._trim_plan`),
    `piece_starts` where each of those ranges lands on the clip, in order.
    """
    for (raw_start, raw_end), clip_start in zip(plan, piece_starts):
        if raw_start <= raw_ms <= raw_end:
            return round(clip_start + (raw_ms - raw_start))
    edges = []
    for (raw_start, raw_end), clip_start in zip(plan, piece_starts):
        edges.append((abs(raw_ms - raw_start), clip_start))
        edges.append((abs(raw_ms - raw_end), clip_start + (raw_end - raw_start)))
    return round(min(edges)[1])


def piece_starts(chunk_start_ms: int, plan: list) -> list:
    """Where each of a chunk's kept ranges lands on the clip.

    The ranges a trim keeps are joined back to back with no gap between
    them (the cut is what is removed), so each one starts where the last
    one ended.
    """
    starts, offset = [], chunk_start_ms
    for raw_start, raw_end in plan:
        starts.append(offset)
        offset += raw_end - raw_start
    return starts


def _map_words(words: list, plan: list, starts: list) -> list:
    mapped = []
    for word in words:
        raw_start_ms = word["start"] * 1000
        raw_end_ms = word["end"] * 1000
        mapped.append(
            {
                "word": word["word"],
                "start_ms": map_raw_ms(raw_start_ms, plan, starts),
                "end_ms": map_raw_ms(raw_end_ms, plan, starts),
                "raw_start_ms": round(raw_start_ms),
                "raw_end_ms": round(raw_end_ms),
                "alignment_quality": word.get("alignment_quality"),
                "probability": word.get("probability"),
            }
        )
    return mapped


def build(text: str, text_hash: str, audio_version: int, variant: str, clip_ms: int, chunk_infos: list) -> dict:
    """The manifest dict for one clip.

    `chunk_infos`: one dict per chunk, with `index`, `text`, `endpoint`
    ("free" or "paid"), `response_key`, `raw_ms`, `plan` (that chunk's kept
    raw ranges), `chunk_start_ms`, `words` (raw, seconds, or None),
    `generation` (the paid endpoint's generation info, or None) and `voice`
    (only set when a paid response reported its own voice or speaker id).
    """
    chunks = []
    for info in chunk_infos:
        starts = piece_starts(info["chunk_start_ms"], info["plan"])
        pieces = [
            {"raw_start_ms": start, "raw_end_ms": end, "clip_start_ms": clip_start}
            for (start, end), clip_start in zip(info["plan"], starts)
        ]
        words = _map_words(info["words"], info["plan"], starts) if info["words"] is not None else None
        chunks.append(
            {
                "index": info["index"],
                "text": info["text"],
                "endpoint": info["endpoint"],
                "response_key": info["response_key"],
                "raw_ms": info["raw_ms"],
                "pieces": pieces,
                "words": words,
                "generation": info.get("generation"),
                "voice": info.get("voice"),
            }
        )
    return {
        "schema": SCHEMA,
        "audio_version": audio_version,
        "text": text,
        "variant": variant,
        "clip_ms": clip_ms,
        "timeline": "pcm_before_mp3_encoding",
        "chunks": chunks,
    }


def write(text_hash: str, audio_version: int, manifest: dict) -> None:
    """Save a manifest. Never raises: a write failure must not fail the clip."""
    try:
        ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
        path_for(text_hash, audio_version).write_text(json.dumps(manifest))
    except OSError as error:
        log(f"Could not write alignment manifest for {text_hash}: {error}")


def read(text_hash: str, audio_version: int) -> dict | None:
    """A previously written manifest, or None if there is not one."""
    try:
        return json.loads(path_for(text_hash, audio_version).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def exists(text_hash: str, audio_version: int) -> bool:
    """Whether a manifest was written for this clip.

    Every clip must have one (2026-09-19); a cached mp3 without a matching
    manifest is not a valid cache entry.
    """
    return path_for(text_hash, audio_version).exists()

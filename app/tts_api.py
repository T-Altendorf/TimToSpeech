import io
import math
import os
import re
import time
import requests
import traceback
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from pydub import AudioSegment
from .config import log, KURDISH_TTS_API_KEY

BASE_URL = "https://www.kurdishtts.com"
FREE_URL = f"{BASE_URL}/api/tts-demo"
PAID_URL = f"{BASE_URL}/api/tts-proxy"

# The free demo endpoint rejects anything above this with HTTP 400.
FREE_CHAR_LIMIT = 150
# Paid plan ceiling for /api/tts-proxy (500 on the free tier, 5000 on paid).
PAID_CHAR_LIMIT = 5000

DIALECT = "kurmanji"
VOICE = "kurmanji_236"
MODEL_VERSION = "v4"

# The demo endpoint streams raw 16-bit mono PCM at this rate (verified against
# the total_duration reported by the speech.audio.done event).
PCM_SAMPLE_RATE = 22050
PCM_SAMPLE_WIDTH = 2

MAX_PARALLEL_CHUNKS = 4
REQUEST_TIMEOUT = 90

# Seamless-merge tuning. Measured on the engine's own output (2026-09-18): every
# clip it returns opens with the same ~20 ms burst at about -35 dBFS, some 60 ms
# in, followed by roughly half a second of silence before the speech starts. A
# plain threshold trim takes that burst for the start of speech and keeps it, so
# every join carried a tick. The trim below works on islands of sound instead: a
# short island with a long silence after it is an engine artifact, not speech.
ACTIVE_DBFS = -50.0  # a frame louder than this counts as sound
FRAME_MS = 5
ISLAND_BRIDGE_MS = 30  # quieter dips shorter than this stay inside one island
ARTIFACT_MAX_MS = 60  # a leading island at most this long ...
ARTIFACT_SILENCE_MS = 150  # ... with at least this much silence after it is dropped
HEAD_PAD_MS = 15  # kept before the first speech frame, so soft onsets survive
TAIL_PAD_MS = 40  # kept after the last speech frame, so decays are not chopped
FADE_IN_MS = 10
FADE_OUT_MS = 40
# The pause between chunks. Chunks end on sentence boundaries, and the engine's
# own lead silence (now trimmed) used to supply most of the pause.
CHUNK_GAP_MS = 350
LEAD_IN_MS = 50  # silence before the first chunk, room for the MP3 encoder delay
LEAD_OUT_MS = 80
MAX_GAIN_MATCH_DB = 4.0
# Peak headroom always left when gaining a chunk up, so matching cannot clip.
CLIP_HEADROOM_DB = 0.5

# Sentence terminators we split on. Periods are the primary boundary; the
# others are included so a text that uses them is not left as one long chunk.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟۔])\s+")


def _hard_split(text: str, limit: int) -> list:
    """Split on word boundaries when a single sentence still exceeds `limit`."""
    words = text.split()
    chunks, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def split_text(text: str, limit: int = FREE_CHAR_LIMIT) -> list:
    """
    Split text into chunks that the free endpoint can handle.

    Text at or below `limit` is returned untouched. Longer text is split into
    sentences, which are then greedily packed back together so each chunk stays
    within `limit`. A sentence that is itself longer than `limit` becomes its
    own chunk - those are the ones that need the authenticated endpoint.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text]

    chunks, current = [], ""
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > limit:
            # Cannot be packed with anything - flush and keep it standalone.
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, PAID_CHAR_LIMIT))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def _synthesize_free(text: str) -> AudioSegment:
    """Synthesize one chunk via the free demo endpoint (SSE stream of PCM)."""
    payload = {
        "text": text,
        "dialect": DIALECT,
        "voice": VOICE,
        "model_version": MODEL_VERSION,
        "stream_format": "sse",
    }

    response = requests.post(
        FREE_URL, json=payload, timeout=REQUEST_TIMEOUT, stream=True
    )
    response.raise_for_status()

    audio_content = b""
    for line in response.iter_lines():
        if not line:
            continue

        decoded_line = line.decode("utf-8")
        data_str = None
        if decoded_line.startswith("data: "):
            data_str = decoded_line[len("data: ") :]
        elif decoded_line.startswith("message | "):
            data_str = decoded_line[len("message | ") :]
        elif decoded_line.startswith("{"):
            data_str = decoded_line

        if not data_str or data_str.strip() == "[DONE]":
            if data_str:
                break
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            log(f"Warning: Could not decode JSON from SSE line: {data_str[:50]}...")
            continue

        event_type = data.get("type")
        if event_type == "speech.audio.delta":
            audio_b64 = data.get("audio")
            if audio_b64:
                audio_content += base64.b64decode(audio_b64)
        elif event_type == "speech.audio.done":
            log(f"API Usage: {data.get('usage', {})}")

    if not audio_content:
        raise RuntimeError("No audio content received from free endpoint")

    return AudioSegment(
        data=audio_content,
        sample_width=PCM_SAMPLE_WIDTH,
        frame_rate=PCM_SAMPLE_RATE,
        channels=1,
    )


def _synthesize_paid(text: str) -> AudioSegment:
    """Synthesize one chunk via the authenticated endpoint (returns a WAV)."""
    if not KURDISH_TTS_API_KEY:
        raise RuntimeError(
            f"Chunk is {len(text)} chars (over the {FREE_CHAR_LIMIT} char free "
            "limit) but KURDISH_TTS_API_KEY is not set"
        )

    payload = {
        "text": text,
        "speaker_id": VOICE,
        "model_version": MODEL_VERSION,
        "format": "wav",
    }

    response = requests.post(
        PAID_URL,
        json=payload,
        headers={"x-api-key": KURDISH_TTS_API_KEY},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    if not response.content:
        raise RuntimeError("No audio content received from authenticated endpoint")

    return AudioSegment.from_file(io.BytesIO(response.content), format="wav")


def _samples(segment: AudioSegment) -> np.ndarray:
    return np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float64)


def _sound_islands(segment: AudioSegment) -> list:
    """Stretches of sound as (start_ms, end_ms), short dips bridged."""
    samples = _samples(segment)
    frame = max(1, segment.frame_rate * FRAME_MS // 1000)
    count = len(samples) // frame
    if count == 0:
        return []
    frames = samples[: count * frame].reshape(count, frame)
    floor = 32768.0 * 10 ** (ACTIVE_DBFS / 20)
    active = np.sqrt((frames**2).mean(axis=1)) > floor

    islands = []
    for index in np.flatnonzero(active):
        start, end = int(index) * FRAME_MS, (int(index) + 1) * FRAME_MS
        if islands and start - islands[-1][1] <= ISLAND_BRIDGE_MS:
            islands[-1][1] = end
        else:
            islands.append([start, end])
    return islands


def _drop_head_artifacts(islands: list) -> list:
    """Drop leading islands too short to be speech that stand alone in silence."""
    while len(islands) > 1:
        (start, end), following = islands[0], islands[1]
        if end - start > ARTIFACT_MAX_MS or following[0] - end < ARTIFACT_SILENCE_MS:
            break
        log(f"  dropped a {end - start}ms engine artifact at {start}ms")
        islands = islands[1:]
    return islands


def _trim_silence(segment: AudioSegment) -> AudioSegment:
    """Cut a chunk down to its speech: no pad silence, no engine artifact."""
    islands = _drop_head_artifacts(_sound_islands(segment))
    if not islands:
        # An all-silence chunk has nothing to keep - leave it as it came.
        return segment
    start = max(0, islands[0][0] - HEAD_PAD_MS)
    end = min(len(segment), islands[-1][1] + TAIL_PAD_MS)
    return segment[start:end]


def _seal_edges(segment: AudioSegment) -> AudioSegment:
    """Raised-cosine fades to exactly zero, so a splice cannot step or tick."""
    samples = _samples(segment)
    per_ms = segment.frame_rate / 1000
    fade_in = min(int(FADE_IN_MS * per_ms), len(samples) // 2)
    fade_out = min(int(FADE_OUT_MS * per_ms), len(samples) // 2)
    if fade_in > 0:
        samples[:fade_in] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_in))
    if fade_out > 0:
        samples[-fade_out:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade_out))
    sealed = np.clip(np.round(samples), -32768, 32767).astype(np.int16)
    return segment._spawn(sealed.tobytes())


def _merge_segments(segments: list) -> AudioSegment:
    """Clean every chunk (a single one too) and join them at true silence."""
    # Chunks can come back at different rates (the free endpoint streams
    # 22050 Hz, the authenticated one 24 kHz), so normalize the format first.
    rate = max(segment.frame_rate for segment in segments)
    prepared = [
        _trim_silence(segment.set_frame_rate(rate).set_channels(1).set_sample_width(2))
        for segment in segments
    ]

    # Match levels to the first chunk so free/authenticated chunks do not step
    # in loudness. Capped, so a quiet chunk is never blown up, and never louder
    # than the chunk's own headroom - the authenticated endpoint already peaks
    # near full scale, and gaining it up would clip into audible distortion.
    reference_dbfs = prepared[0].dBFS
    if math.isfinite(reference_dbfs):
        for i, segment in enumerate(prepared[1:], start=1):
            if not math.isfinite(segment.dBFS):
                continue
            gain = max(-MAX_GAIN_MATCH_DB, min(MAX_GAIN_MATCH_DB, reference_dbfs - segment.dBFS))
            if gain > 0 and math.isfinite(segment.max_dBFS):
                headroom = -segment.max_dBFS - CLIP_HEADROOM_DB
                gain = min(gain, max(0.0, headroom))
            if abs(gain) > 0.1:
                log(f"  chunk {i + 1}: level matched by {gain:+.2f} dB")
                prepared[i] = segment.apply_gain(gain)

    gap = AudioSegment.silent(duration=CHUNK_GAP_MS, frame_rate=rate)
    combined = AudioSegment.silent(duration=LEAD_IN_MS, frame_rate=rate)
    for index, segment in enumerate(prepared):
        combined += (gap if index else AudioSegment.empty()) + _seal_edges(segment)
    return combined + AudioSegment.silent(duration=LEAD_OUT_MS, frame_rate=rate)


def _synthesize_chunk(index: int, text: str) -> AudioSegment:
    """Route one chunk to the free or authenticated endpoint by length."""
    use_paid = len(text) > FREE_CHAR_LIMIT
    endpoint = "authenticated" if use_paid else "free"
    log(f"  chunk {index + 1}: {len(text)} chars via {endpoint} endpoint")

    start = time.perf_counter()
    segment = _synthesize_paid(text) if use_paid else _synthesize_free(text)
    elapsed = time.perf_counter() - start
    log(f"  chunk {index + 1}: done in {elapsed:.3f}s ({len(segment)}ms audio)")
    return segment


def call_kurdish_tts_api(text: str, output_path: Path) -> bool:
    """
    Call the Kurdish TTS API, splitting long text across parallel requests.

    Text within the free 150 char limit goes to the free demo endpoint as a
    single request. Longer text is split on sentence boundaries; each chunk
    still uses the free endpoint, and only sentences that are themselves over
    150 chars fall through to the authenticated endpoint. Chunks are generated
    in parallel and concatenated in order.

    Args:
        text: The text to convert to speech
        output_path: Path where the audio file should be saved

    Returns:
        True if successful, False otherwise
    """
    log("=== Kurdish TTS API Call Started ===")
    log(f"Input: '{text}'")
    start_time = time.perf_counter()

    try:
        chunks = split_text(text)
        paid_count = sum(1 for chunk in chunks if len(chunk) > FREE_CHAR_LIMIT)
        log(
            f"Split {len(text)} chars into {len(chunks)} chunk(s) "
            f"({len(chunks) - paid_count} free, {paid_count} authenticated)"
        )

        if paid_count and not KURDISH_TTS_API_KEY:
            log(
                f"✗ {paid_count} chunk(s) exceed the {FREE_CHAR_LIMIT} char free "
                "limit but KURDISH_TTS_API_KEY is not set"
            )
            return False

        if len(chunks) == 1:
            segments = [_synthesize_chunk(0, chunks[0])]
        else:
            workers = min(len(chunks), MAX_PARALLEL_CHUNKS)
            log(f"Generating {len(chunks)} chunks with {workers} parallel workers")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # executor.map preserves input order and re-raises failures.
                segments = list(executor.map(_synthesize_chunk, range(len(chunks)), chunks))

        combined = _merge_segments(segments)
        combined.export(str(output_path), format="mp3", bitrate="128k")

        file_size = os.path.getsize(output_path)
        elapsed = time.perf_counter() - start_time
        log(
            f"✓ Kurdish TTS API succeeded | Chunks: {len(chunks)} | "
            f"Size: {file_size} bytes | Audio: {len(combined) / 1000:.2f}s | "
            f"Time: {elapsed:.3f}s"
        )
        log("=== Kurdish TTS API Call Completed ===")
        return True

    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start_time
        log(f"✗ Kurdish TTS API failed: {str(e)} | Time: {elapsed:.3f}s")
        return False
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        log(f"✗ Unexpected error in Kurdish TTS API: {str(e)} | Time: {elapsed:.3f}s")
        traceback.print_exc()
        return False

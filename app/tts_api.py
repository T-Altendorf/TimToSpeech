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
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
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

# Seamless-merge tuning. Chunks are generated independently, so their edges do
# not line up: splicing them raw leaves an amplitude step at every join, which
# is what you hear as a click. Each chunk is trimmed to its speech, faded to
# true zero at both ends, level-matched to the first chunk, and separated by a
# short silence, so every join happens at silence with no discontinuity.
SILENCE_THRESHOLD_DBFS = -50.0
EDGE_FADE_MS = 8
CHUNK_GAP_MS = 120
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


def _trim_silence(segment: AudioSegment) -> AudioSegment:
    """Strip the leading/trailing silence the engine pads each chunk with."""
    lead = detect_leading_silence(segment, silence_threshold=SILENCE_THRESHOLD_DBFS)
    tail = detect_leading_silence(
        segment.reverse(), silence_threshold=SILENCE_THRESHOLD_DBFS
    )
    trimmed = segment[lead : len(segment) - tail]
    # An all-silence chunk trims to nothing - keep the original in that case.
    return trimmed if len(trimmed) else segment


def _seal_edges(segment: AudioSegment) -> AudioSegment:
    """Fade both ends to exactly zero so a splice cannot step in amplitude."""
    fade = min(EDGE_FADE_MS, len(segment) // 2)
    if fade <= 0:
        return segment
    return segment.fade_in(fade).fade_out(fade)


def _merge_segments(segments: list) -> AudioSegment:
    """Join chunks into one seamless track (no clicks at the boundaries)."""
    if len(segments) == 1:
        return segments[0]

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
    combined = _seal_edges(prepared[0])
    for segment in prepared[1:]:
        combined += gap + _seal_edges(segment)
    return combined


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

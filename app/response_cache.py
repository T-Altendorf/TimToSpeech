"""Full upstream response bodies, cached beside the audio cache.

The upstream TTS API returns more than audio - word or phoneme alignment,
timings - that today's pipeline throws away once it has the PCM. This keeps
the response body exactly as it arrived, so a repeat of the same request
never has to pay upstream again and that extra data survives for later use.

Keyed on a hash of the exact text sent upstream, the same way the audio
cache hashes text, but never on AUDIO_VERSION: a change to how we turn the
response into a clip does not make the response itself stale.
"""

import hashlib
import json
from pathlib import Path

from .config import CACHE_DIR

RESPONSE_CACHE_DIR = CACHE_DIR / "responses"


def _key(text: str, endpoint: str, variant: str = "") -> str:
    # The voice and dialect are part of what was asked of upstream: a changed
    # voice must never be answered with the old voice's response.
    text_hash = hashlib.sha256(f"{variant}\n{text}".encode()).hexdigest()
    return f"{text_hash}.{endpoint}"


def read(text: str, endpoint: str, variant: str = "") -> bytes | None:
    """The raw response body from a previous identical request, or None."""
    path = RESPONSE_CACHE_DIR / f"{_key(text, endpoint, variant)}.body"
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def write(
    text: str, endpoint: str, body: bytes, content_type: str, variant: str = ""
) -> None:
    """Save a full upstream response body and a little metadata about it."""
    RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key(text, endpoint, variant)
    (RESPONSE_CACHE_DIR / f"{key}.body").write_bytes(body)
    meta = {
        "content_type": content_type,
        "chars": len(text),
        "bytes": len(body),
        "variant": variant,
    }
    (RESPONSE_CACHE_DIR / f"{key}.meta.json").write_text(json.dumps(meta))

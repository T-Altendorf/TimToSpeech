"""The alignment check for manifests written before it existed.

Every clip cached before the check (2341 on 2026-10-02) has word times no
one has held against its sound. None of them may reach the app unchecked:

- `ensure_checked` checks one manifest the moment a request reads it
  (`/alignment`, and `/tts` for a voice other than the default), from the
  cached mp3 alone: no upstream call.
- `start_sweep` checks the whole cache in the background, one clip at a
  time, at low priority. Only one worker sweeps (an exclusive lock file);
  every manifest is replaced by an atomic rename (`alignment.write`), and a
  manifest whose `check.version` is current is skipped, so a sweep that is
  interrupted or runs twice does nothing twice.
"""

import copy
import fcntl
import json
import os
import threading
import time

from pydub import AudioSegment

from . import alignment, alignment_check, voices
from .config import log
from .tts_api import _sound_islands
from .utils import AUDIO_VERSION, clip_path_for_hash

# A pause between two clips, so the sweep yields to requests. Decoding one
# short mp3 took about 150ms on the Mac (74 real clips in 11.4s), so 2341
# clips take about six to eight minutes.
SWEEP_PAUSE_S = 0.04
SWEEP_NICE = 10  # on Linux this lowers only the sweep thread and its ffmpeg children
LOCK_NAME = ".check-sweep.lock"
HEALTH_TTL_S = 30  # /health counts the manifests at most this often

_health_lock = threading.Lock()
_health_memo = {}


def _withheld(manifest: dict) -> dict:
    """A copy with every chunk's words set aside, for a manifest that could
    not be checked (its mp3 is gone or will not decode). Never written."""
    served = copy.deepcopy(manifest)
    for chunk in served.get("chunks", []):
        if chunk.get("words") is not None:
            chunk["rejected_words"], chunk["words"] = chunk["words"], None
    return served


def check_one(hashed: str, voice: str, manifest: dict) -> dict:
    """Check one manifest against its cached mp3 and save the verdict."""
    islands = _sound_islands(AudioSegment.from_file(clip_path_for_hash(hashed, voice), format="mp3"))
    alignment_check.apply(manifest, islands)
    alignment.write(hashed, AUDIO_VERSION, manifest, voice)
    return manifest


def ensure_checked(hashed: str, voice: str, manifest: dict | None) -> dict | None:
    """The manifest as it may be served: checked now if it was not yet."""
    if manifest is None or "chunks" not in manifest or alignment_check.is_current(manifest):
        return manifest
    try:
        return check_one(hashed, voice, manifest)
    except Exception as error:  # a missing or broken mp3 must not fail the request
        log(f"Alignment check could not run for {hashed} ({voice}): {error}")
        return _withheld(manifest)


def _parse_name(name: str) -> tuple | None:
    """(hash, voice) from `{hash}[.{voice}].a{v}.json`, or None for another version."""
    suffix = f".a{AUDIO_VERSION}.json"
    if not name.endswith(suffix):
        return None
    hashed, _, tag = name[: -len(suffix)].partition(".")
    voice = tag or voices.DEFAULT_VOICE
    return (hashed, voice) if voice in voices.VOICES else None


def _unchecked(path) -> tuple | None:
    """(hash, voice, manifest) for a manifest the check has still to see.

    None when it is checked, of another version, unreadable, or its clip is
    gone (it is checked when a request makes the clip again).
    """
    parsed, manifest = _parse_name(path.name), _read(path)
    if parsed is None or manifest is None or alignment_check.is_current(manifest):
        return None
    return (*parsed, manifest) if clip_path_for_hash(*parsed).exists() else None


def _read(path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def sweep() -> tuple:
    """Check every unchecked manifest of the current version: (checked, failed).

    Returns (0, 0) at once when another worker holds the sweep lock.
    """
    alignment.ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(alignment.ALIGNMENT_DIR / LOCK_NAME, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0, 0
        checked = failed = 0
        for path in sorted(alignment.ALIGNMENT_DIR.glob(f"*.a{AUDIO_VERSION}.json")):
            found = _unchecked(path)
            if found is None:
                continue
            result = ensure_checked(*found)
            checked += 1
            failed += not result.get("check", {}).get("ok", False)
            time.sleep(SWEEP_PAUSE_S)
        return checked, failed


def _run_sweep() -> None:
    try:
        os.nice(SWEEP_NICE)
    except OSError:
        pass
    started = time.perf_counter()
    checked, failed = sweep()
    log(f"Alignment check sweep: {checked} checked, {failed} failed, {time.perf_counter() - started:.1f}s")


def start_sweep() -> threading.Thread:
    """Run `sweep` once in a background daemon thread (from `wsgi.py`)."""
    thread = threading.Thread(target=_run_sweep, name="alignment-check-sweep", daemon=True)
    thread.start()
    return thread


def _count() -> dict:
    checked = failed = pending = 0
    for path in alignment.ALIGNMENT_DIR.glob(f"*.a{AUDIO_VERSION}.json"):
        manifest = _read(path) or {}
        if alignment_check.is_current(manifest):
            checked += 1
            failed += not manifest["check"]["ok"]
        else:
            pending += _unchecked(path) is not None
    return {"version": alignment_check.VERSION, "checked": checked, "failed": failed, "pending": pending}


def health() -> dict:
    """`/health`'s `alignment_check`: version, checked, failed, pending."""
    key = str(alignment.ALIGNMENT_DIR)
    with _health_lock:
        at, counts = _health_memo.get(key, (None, None))
        if at is None or time.monotonic() - at > HEALTH_TTL_S:
            counts = _count()
            _health_memo[key] = (time.monotonic(), counts)
        return counts

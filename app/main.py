#!/usr/bin/env python3
"""
TimToSpeech - Lightweight TTS service for Kurmanji text
"""
import os
import hashlib
from threading import Thread
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

from .config import log, CACHE_DIR, TTS_WAIT_TIMEOUT, CORS_ORIGINS
from .utils import AUDIO_VERSION, cache_counts, get_cache_path, prune_stale_cache, text_hash
from .tts_local import load_models, generate_audio
from . import alignment, alignment_check, alignment_sweep, tts_local, voices
from .tts_api import call_kurdish_tts_api

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# A new audio version retires every older clip the moment the service starts.
_pruned = prune_stale_cache()
if _pruned:
    log(f"Deleted {_pruned} cached clip(s) from an older audio version")

# Processing jobs tracker
processing_jobs = {}


def generate_tts_async(
    job_id: str,
    text: str,
    output_path: str,
    use_api: bool,
    voice: str = voices.DEFAULT_VOICE,
    regenerated: bool = False,
):
    """
    Wrapper for async TTS generation (either API or local model)

    Args:
        job_id: Unique identifier for this job
        text: The text to convert to speech
        output_path: Path where the audio file should be saved
        use_api: If True, try Kurdish TTS API; if False, use local model
        voice: A voice id from `app/voices.py`. Only the default voice falls
            back to the local model: it has one voice, and a fallback clip
            saved under another voice's name would be a lie in the cache.
        regenerated: the clip replaces one the alignment check found stray
            sound in (`_replaces_stray_tail`); never replaced a second time.
    """
    processing_jobs[job_id] = {"status": "processing", "path": None}

    success = False
    if use_api:
        # The flag is passed only when set, so the call is exactly as before otherwise.
        extra = {"regenerated": True} if regenerated else {}
        success = call_kurdish_tts_api(text, output_path, voice, **extra)
        if not success and voice == voices.DEFAULT_VOICE:
            log("Kurdish TTS API failed, falling back to local model")
            success = generate_audio(text, output_path)
        elif not success:
            log(f"Kurdish TTS API failed for voice {voice}; no local fallback for it")
    else:
        success = generate_audio(text, output_path)

    if success:
        processing_jobs[job_id] = {"status": "completed", "path": str(output_path)}
    else:
        processing_jobs[job_id] = {"status": "failed", "path": None}


def _requested_voice() -> tuple:
    """The `voice` query parameter, resolved: (voice, None), or (None, a 400)."""
    try:
        return voices.resolve(request.args.get("voice")), None
    except voices.UnknownVoice as error:
        refusal = {"error": f"Unknown voice '{error}'", "allowed_voices": voices.allowed()}
        return None, (jsonify(refusal), 400)


def _replaces_stray_tail(text: str, voice: str) -> bool:
    """Whether a cached clip is to be synthesised once more (2026-10-02).

    A clip in a voice other than the default whose check found stray sound
    after its words is not a valid cache entry, once: the regenerated clip's
    manifest says `regenerated`, so it is never replaced again. A default-voice
    clip is never replaced by the check (he has those on his phone); only
    its words are withheld.
    """
    if voice == voices.DEFAULT_VOICE:
        return False
    hashed = text_hash(text)
    manifest = alignment_sweep.ensure_checked(hashed, voice, alignment.read(hashed, AUDIO_VERSION, voice))
    if not manifest or not alignment_check.is_current(manifest):
        return False
    return alignment_check.has_stray_tail(manifest) and not manifest["check"]["regenerated"]


def _was_regenerated(text: str, voice: str) -> bool:
    manifest = alignment.read(text_hash(text), AUDIO_VERSION, voice) or {}
    return bool(manifest.get("check", {}).get("regenerated"))


@app.route("/tts", methods=["GET"])
def text_to_speech():
    """
    TTS endpoint - converts text to speech
    Uses the Kurdish TTS API, splitting text over 150 chars into parallel
    chunks; falls back to the local TTS model if the API fails
    Query parameters:
      - text: The text to convert to speech (required)
      - voice: a voice id from `app/voices.py` (default kurmanji_236); any
        other id is a 400 with the allowed ones
      - force_regen: If true, delete this voice's cached audio for this text and regenerate
    """
    text = request.args.get("text", "").strip()
    force_regen = request.args.get("force_regen", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not text:
        return jsonify({"error": "Please provide 'text' parameter"}), 400
    voice, refused = _requested_voice()
    if refused:
        return refused

    # Check cache
    cache_path = get_cache_path(text, voice)
    if force_regen and cache_path.exists():
        try:
            cache_path.unlink()
            log(f"Cache invalidated for text: '{text[:50]}...'")
        except OSError as e:
            log(f"Failed to invalidate cache for text: '{text[:50]}...': {e}")
            return jsonify({"error": "Failed to invalidate cache"}), 500

    regenerating = False
    if cache_path.exists():
        if not alignment.exists(text_hash(text), AUDIO_VERSION, voice):
            # Every clip must have a manifest (2026-09-19). One that does not
            # is not a valid cache entry - drop it and fall through to regenerate.
            log(f"Cached clip has no alignment manifest, regenerating: '{text[:50]}...'")
        elif _replaces_stray_tail(text, voice):
            log(f"Cached clip has stray sound after its words, regenerating once: '{text[:50]}...'")
            regenerating = True
        else:
            log(f"Cache hit for text: '{text[:50]}...'")
            return send_file(cache_path, mimetype="audio/mpeg")
        try:
            cache_path.unlink()
        except OSError:
            pass

    # The API handles any length now - long text is split into parallel chunks
    use_api = True
    log(f"Text is {len(text)} characters, using Kurdish TTS API")

    # Generate job ID
    job_id = hashlib.sha256(f"{text}{os.urandom(8).hex()}".encode()).hexdigest()[:16]

    # Start async generation
    regenerated = regenerating or _was_regenerated(text, voice)
    thread = Thread(target=generate_tts_async, args=(job_id, text, cache_path, use_api, voice, regenerated))
    thread.daemon = True
    thread.start()

    # Wait briefly to see if generation completes quickly
    thread.join(timeout=TTS_WAIT_TIMEOUT)

    if cache_path.exists():
        # Generation completed within timeout
        return send_file(cache_path, mimetype="audio/mpeg")
    else:
        # Still processing, return job ID for polling
        return (
            jsonify(
                {
                    "message": "Audio generation started. This may take a while. Check status using job_id.",
                    "job_id": job_id,
                    "status_url": f"/status/{job_id}",
                }
            ),
            202,
        )


@app.route("/status/<job_id>", methods=["GET"])
def check_status(job_id):
    """Check status of a TTS generation job"""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = processing_jobs[job_id]
    if job["status"] == "completed":
        return send_file(job["path"], mimetype="audio/mpeg")
    elif job["status"] == "failed":
        return jsonify({"error": "Audio generation failed"}), 500
    else:
        return jsonify({"status": "processing"}), 202


@app.route("/alignment", methods=["GET"])
def get_alignment():
    """The saved word-for-word timing manifest for a clip, if one was written.

    A manifest of the current version is checked against its clip's sound
    before it is served, if it was not yet (`alignment_sweep.ensure_checked`);
    a chunk that fails has `words: null` and the engine's under `rejected_words`.
    Query parameters:
      - text: the exact text the clip was generated from (required)
      - audio_version: which pipeline version's manifest to read (default: current)
      - voice: whose clip's manifest (default kurmanji_236)
    """
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please provide 'text' parameter"}), 400
    voice, refused = _requested_voice()
    if refused:
        return refused

    version_param = request.args.get("audio_version", "").strip()
    try:
        version = int(version_param) if version_param else AUDIO_VERSION
    except ValueError:
        return jsonify({"error": "'audio_version' must be an integer"}), 400

    manifest = alignment.read(text_hash(text), version, voice)
    if version == AUDIO_VERSION:
        # Never serve word times the check has not seen (2026-10-02).
        manifest = alignment_sweep.ensure_checked(text_hash(text), voice, manifest)
    if manifest is None:
        return jsonify({"error": "No alignment manifest for this text"}), 404
    return jsonify(manifest)


@app.route("/favicon.ico")
def favicon():
    return "", 204  # 204 = No Content


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            # Clients key their own clip cache on this, so a pipeline change
            # here retires their old clips too.
            "audio_version": AUDIO_VERSION,
            # `stale` must read 0: clips of an older version are deleted at start.
            "cache": cache_counts(),
            "default_voice": voices.DEFAULT_VOICE,
            "voices": {voice: voices.model_version(voice) for voice in voices.allowed()},
            "tts_model_loaded": tts_local.model is not None
            and tts_local.tokenizer is not None,
            # version, checked, failed, pending: `pending` falls to 0 once the
            # startup sweep is through (`app/alignment_sweep.py`).
            "alignment_check": alignment_sweep.health(),
        }
    )


if __name__ == "__main__":
    log("Starting TimToSpeech service...")
    load_models()
    port = int(os.getenv("PORT", "8000"))
    log(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

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
from .utils import get_cache_path
from .tts_local import load_models, generate_audio
from . import tts_local
from .tts_api import call_kurdish_tts_api

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# Processing jobs tracker
processing_jobs = {}


def generate_tts_async(job_id: str, text: str, output_path: str, use_api: bool):
    """
    Wrapper for async TTS generation (either API or local model)

    Args:
        job_id: Unique identifier for this job
        text: The text to convert to speech
        output_path: Path where the audio file should be saved
        use_api: If True, try Kurdish TTS API; if False, use local model
    """
    processing_jobs[job_id] = {"status": "processing", "path": None}

    success = False
    if use_api:
        success = call_kurdish_tts_api(text, output_path)
        if not success:
            log("Kurdish TTS API failed, falling back to local model")
            success = generate_audio(text, output_path)
    else:
        success = generate_audio(text, output_path)

    if success:
        processing_jobs[job_id] = {"status": "completed", "path": str(output_path)}
    else:
        processing_jobs[job_id] = {"status": "failed", "path": None}


@app.route("/tts", methods=["GET"])
def text_to_speech():
    """
    TTS endpoint - converts text to speech
    Uses the Kurdish TTS API, splitting text over 150 chars into parallel
    chunks; falls back to the local TTS model if the API fails
    Query parameters:
      - text: The text to convert to speech (required)
      - force_regen: If true, delete cached audio for this text and regenerate
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

    # Check cache
    cache_path = get_cache_path(text)
    if force_regen and cache_path.exists():
        try:
            cache_path.unlink()
            log(f"Cache invalidated for text: '{text[:50]}...'")
        except OSError as e:
            log(f"Failed to invalidate cache for text: '{text[:50]}...': {e}")
            return jsonify({"error": "Failed to invalidate cache"}), 500

    if cache_path.exists():
        log(f"Cache hit for text: '{text[:50]}...'")
        return send_file(cache_path, mimetype="audio/mpeg")

    # The API handles any length now - long text is split into parallel chunks
    use_api = True
    log(f"Text is {len(text)} characters, using Kurdish TTS API")

    # Generate job ID
    job_id = hashlib.sha256(f"{text}{os.urandom(8).hex()}".encode()).hexdigest()[:16]

    # Start async generation
    thread = Thread(target=generate_tts_async, args=(job_id, text, cache_path, use_api))
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


@app.route("/favicon.ico")
def favicon():
    return "", 204  # 204 = No Content


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "tts_model_loaded": tts_local.model is not None
            and tts_local.tokenizer is not None,
        }
    )


if __name__ == "__main__":
    log("Starting TimToSpeech service...")
    load_models()
    port = int(os.getenv("PORT", "8000"))
    log(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

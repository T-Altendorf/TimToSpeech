#!/usr/bin/env python3
"""
TimToSpeech - Lightweight TTS service for Kurmanji text
"""
import os
import hashlib
import tempfile
import re
import traceback
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from transformers import (
    VitsModel,
    AutoTokenizer,
)
import torch
import numpy as np
import scipy.io.wavfile
from pydub import AudioSegment
from threading import Thread
import time
import datetime

app = Flask(__name__)

# Configuration
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/app/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for models
model = None
tokenizer = None

# Processing jobs tracker
processing_jobs = {}


import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log(message: str):
    """Log message with timestamp"""
    logging.info(message)


def load_models():
    """Load TTS model at startup"""
    global model, tokenizer

    log("Loading TTS model...")
    try:
        model = VitsModel.from_pretrained("facebook/mms-tts-kmr-script_latin")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kmr-script_latin")
        log("✓ TTS model loaded successfully")
    except Exception as e:
        log(f"✗ Error loading TTS model: {e}")
        model = None
        tokenizer = None


# Numbers to words mapping
num2word = {
    "0": "sifir",
    "1": "yek",
    "2": "du",
    "3": "sê",
    "4": "çar",
    "5": "pênc",
    "6": "şeş",
    "7": "heft",
    "8": "heşt",
    "9": "neh",
    "10": "deh",
}


def replace_numbers_with_words(text):
    """Replace numbers with their word equivalents"""

    def repl(match):
        num = match.group()
        return num2word.get(num, num)

    return re.sub(r"\b\d+\b", repl, text)


# Abbreviations
abbrev_as_word = {
    "KCK": "Keceke",
    "PKK": "Pekeke",
    "PAJK": "Pajek",
    "PYD": "Peyede",
    "YPG": "Yepege",
    "YPJ": "Yepeje",
    "HDP": "Hedepe",
    "DBP": "Debepe",
    "KDP": "Kedepe",
    "PDK": "Pedeke",
    "PUK": "Pûk",
    "YNK": "Yeneke",
    "TAK": "Tak",
    "PJAK": "Pejak",
    "ENKS": "Enekese",
    "TEV-DEM": "Tevdem",
    "KOMKAR": "Komkar",
    "NATO": "Nato",
    "UNESCO": "Yunesko",
    "UNICEF": "Yunîsef",
    "VOA": "Voa",
    "RAM": "Rem",
    "ram": "Rem",
}

abbrev_spelled = {
    "UN": "Û En",
    "EU": "E Û",
    "NGO": "En Cî O",
    "KRG": "Ke Re Ge",
    "BBC": "Bî Bî Sî",
    "CNN": "Sî En En",
    "DW": "De We",
    "TRT": "Te Re Te",
    "RT": "Er Te",
    "USB": "U Se Be",
    "PDF": "Pe De Fe",
    "AI": "A Î",
    "IT": "Ay Tî",
    "HTTP": "He Te Te Pe",
    "HTML": "He Te Me Le",
    "URL": "U Re Le",
    "IP": "Ay Pî",
    "CPU": "Sî Pî U",
    "GPU": "Cî Pî U",
    "SMS": "Es Em Es",
    "GPS": "Cî Pî Es",
}

abbrev_map = {}
abbrev_map.update(abbrev_as_word)
abbrev_map.update(abbrev_spelled)


def expand_abbreviations(text: str) -> str:
    """Expand abbreviations to their full forms"""
    for abbr, full in abbrev_map.items():
        pattern = r"(?<!\w)" + re.escape(abbr) + r"(?!\w)"
        text = re.sub(pattern, full, text)
    return text


def normalize_text(text: str) -> str:
    """Normalize quotation marks and apostrophes"""
    text = text.replace(""", "\"").replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    return text


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline"""
    text = normalize_text(text)
    text = replace_numbers_with_words(text)
    text = expand_abbreviations(text)
    return text


def generate_audio(text: str, output_path: Path):
    """Generate audio from text and save to output_path, with timing and filesize logs"""

    def sizeof_fmt(num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi"]:
            if abs(num) < 1024.0:
                return f"{num:.2f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.2f} Ti{suffix}"

    log(f"=== TTS Generation Started ===")
    log(f"Input: '{text}'")
    start_total = time.perf_counter()

    try:
        if model is None or tokenizer is None:
            raise Exception("TTS model not loaded properly")

        # Preprocessing
        t0 = time.perf_counter()
        processed_text = preprocess_text(text.strip())
        t1 = time.perf_counter()
        log(f"Preprocessing | Time: {t1 - t0:.3f}s")

        # Tokenizing
        t_tok_start = time.perf_counter()
        inputs = tokenizer(processed_text, return_tensors="pt")
        t_tok_end = time.perf_counter()
        log(f"Tokenizing | Time: {t_tok_end - t_tok_start:.3f}s")

        # Generating audio (model inference)
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            output = model(**inputs).waveform
        t_gen_end = time.perf_counter()
        log(f"Model inference | Time: {t_gen_end - t_gen_start:.3f}s")

        # Convert waveform tensor to numpy and normalize
        t_wave_start = time.perf_counter()
        waveform = output.squeeze()
        if hasattr(waveform, "cpu"):
            waveform = waveform.cpu()
        waveform = waveform.numpy()
        max_val = np.max(np.abs(waveform)) if waveform.size else 0.0
        if max_val > 0:
            waveform = waveform / max_val
        else:
            log("⚠️ Waveform max is zero, skipping normalization")
        t_wave_end = time.perf_counter()
        log(
            f"Waveform conversion & normalization | Time: {t_wave_end - t_wave_start:.3f}s"
        )

        # Save WAV to temporary file
        t_wav_start = time.perf_counter()
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        sampling_rate = getattr(model.config, "sampling_rate", 16000)
        # Ensure data type acceptable to scipy: prefer float32
        try:
            scipy.io.wavfile.write(tmp_wav_path, rate=sampling_rate, data=waveform)
        except Exception:
            # fallback: convert to float32
            scipy.io.wavfile.write(
                tmp_wav_path, rate=sampling_rate, data=waveform.astype(np.float32)
            )
        t_wav_end = time.perf_counter()
        wav_size = os.path.getsize(tmp_wav_path) if os.path.exists(tmp_wav_path) else 0
        log(f"WAV write: {sizeof_fmt(wav_size)} | Time: {t_wav_end - t_wav_start:.3f}s")

        # Convert WAV to MP3
        t_mp3_start = time.perf_counter()
        audio = AudioSegment.from_wav(tmp_wav_path)
        audio.export(str(output_path), format="mp3", bitrate="128k")
        t_mp3_end = time.perf_counter()
        mp3_size = (
            os.path.getsize(str(output_path)) if os.path.exists(str(output_path)) else 0
        )
        log(
            f"MP3 conversion: {sizeof_fmt(mp3_size)} | Time: {t_mp3_end - t_mp3_start:.3f}s"
        )

        # Show size comparison
        if wav_size and mp3_size:
            ratio = wav_size / mp3_size if mp3_size else float("inf")
            log(
                f"Size: WAV {sizeof_fmt(wav_size)} / MP3 {sizeof_fmt(mp3_size)} = {ratio:.2f}x"
            )

        # Clean up temporary WAV file
        try:
            os.unlink(tmp_wav_path)
        except Exception:
            pass

        total_time = time.perf_counter() - start_total
        log(f"✓ Total time: {total_time:.3f}s")
        log("=== TTS Generation Completed ===")
        return True

    except Exception as e:
        error_msg = f"Error in TTS: {str(e)}"
        log(f"✗ {error_msg}")
        traceback.print_exc()
        return False


def generate_audio_async(job_id: str, text: str, output_path: Path):
    """Wrapper for async audio generation"""
    processing_jobs[job_id] = {"status": "processing", "path": None}
    success = generate_audio(text, output_path)
    if success:
        processing_jobs[job_id] = {"status": "completed", "path": str(output_path)}
    else:
        processing_jobs[job_id] = {"status": "failed", "path": None}


def get_cache_path(text: str) -> Path:
    """Generate cache file path based on text hash"""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return CACHE_DIR / f"{text_hash}.mp3"


@app.route("/tts", methods=["GET"])
def text_to_speech():
    """
    TTS endpoint - converts text to speech
    Query parameters:
      - text: The text to convert to speech (required)
    """
    text = request.args.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide 'text' parameter"}), 400

    # Check cache
    cache_path = get_cache_path(text)
    if cache_path.exists():
        log(f"Cache hit for text: '{text[:50]}...'")
        return send_file(cache_path, mimetype="audio/mpeg")

    # Generate job ID
    job_id = hashlib.sha256(f"{text}{os.urandom(8).hex()}".encode()).hexdigest()[:16]

    # Start async generation
    thread = Thread(target=generate_audio_async, args=(job_id, text, cache_path))
    thread.daemon = True
    thread.start()

    # Wait briefly to see if generation completes quickly
    thread.join(timeout=7.0)

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
            "tts_model_loaded": model is not None and tokenizer is not None,
        }
    )


if __name__ == "__main__":
    log("Starting TimToSpeech service...")
    load_models()
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)

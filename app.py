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
from transformers import VitsModel, AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np
import scipy.io.wavfile
from pydub import AudioSegment
from threading import Thread

app = Flask(__name__)

# Configuration
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/app/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for models
punct_pipe = None
model = None
tokenizer = None

# Processing jobs tracker
processing_jobs = {}


def load_models():
    """Load TTS and punctuation models at startup"""
    global punct_pipe, model, tokenizer
    
    print("Loading punctuation model...")
    try:
        punctuation_model_id = "oliverguhr/fullstop-punctuation-multilang-large"
        punct_tokenizer = AutoTokenizer.from_pretrained(punctuation_model_id)
        punct_model = AutoModelForTokenClassification.from_pretrained(punctuation_model_id)
        punct_pipe = pipeline("token-classification", model=punct_model, tokenizer=punct_tokenizer, aggregation_strategy="simple")
        print("✓ Punctuation model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading punctuation model: {e}")
        punct_pipe = None
    
    print("Loading TTS model...")
    try:
        model = VitsModel.from_pretrained("facebook/mms-tts-kmr-script_latin")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kmr-script_latin")
        print("✓ TTS model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading TTS model: {e}")
        model = None
        tokenizer = None


# Numbers to words mapping
num2word = {
    "0": "sifir", "1": "yek", "2": "du", "3": "sê", "4": "çar", "5": "pênc",
    "6": "şeş", "7": "heft", "8": "heşt", "9": "neh", "10": "deh"
}


def replace_numbers_with_words(text):
    """Replace numbers with their word equivalents"""
    def repl(match):
        num = match.group()
        return num2word.get(num, num)
    return re.sub(r'\b\d+\b', repl, text)


# Abbreviations
abbrev_as_word = {
    "KCK": "Keceke", "PKK": "Pekeke", "PAJK": "Pajek", "PYD": "Peyede",
    "YPG": "Yepege", "YPJ": "Yepeje", "HDP": "Hedepe", "DBP": "Debepe",
    "KDP": "Kedepe", "PDK": "Pedeke", "PUK": "Pûk", "YNK": "Yeneke",
    "TAK": "Tak", "PJAK": "Pejak", "ENKS": "Enekese", "TEV-DEM": "Tevdem",
    "KOMKAR": "Komkar", "NATO": "Nato", "UNESCO": "Yunesko",
    "UNICEF": "Yunîsef", "VOA": "Voa", "RAM": "Rem", "ram": "Rem",
}

abbrev_spelled = {
    "UN": "Û En", "EU": "E Û", "NGO": "En Cî O", "KRG": "Ke Re Ge",
    "BBC": "Bî Bî Sî", "CNN": "Sî En En", "DW": "De We", "TRT": "Te Re Te",
    "RT": "Er Te", "USB": "U Se Be", "PDF": "Pe De Fe", "AI": "A Î",
    "IT": "Ay Tî", "HTTP": "He Te Te Pe", "HTML": "He Te Me Le",
    "URL": "U Re Le", "IP": "Ay Pî", "CPU": "Sî Pî U", "GPU": "Cî Pî U",
    "SMS": "Es Em Es", "GPS": "Cî Pî Es",
}

abbrev_map = {}
abbrev_map.update(abbrev_as_word)
abbrev_map.update(abbrev_spelled)


def expand_abbreviations(text: str) -> str:
    """Expand abbreviations to their full forms"""
    for abbr, full in abbrev_map.items():
        pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
        text = re.sub(pattern, full, text)
    return text


def normalize_text(text: str) -> str:
    """Normalize quotation marks and apostrophes"""
    text = text.replace(""", "\"").replace(""", "\"")
    text = text.replace("'", "'").replace("'", "'")
    return text


def restore_punctuation(text):
    """Restore punctuation using ML model"""
    if punct_pipe is None:
        print("Punctuation model not available, skipping...")
        return text
    
    try:
        results = punct_pipe(text)
        punctuated = ""
        for token in results:
            word = token['word']
            punct = token.get('entity_group', '')
            if punct == "PERIOD":
                punctuated += word + ". "
            elif punct == "COMMA":
                punctuated += word + ", "
            else:
                punctuated += word + " "
        return punctuated.strip()
    except Exception as e:
        print(f"Punctuation error: {e}")
        return text


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline"""
    text = normalize_text(text)
    text = replace_numbers_with_words(text)
    text = expand_abbreviations(text)
    text = restore_punctuation(text)
    return text


def generate_audio(text: str, output_path: Path):
    """Generate audio from text and save to output_path"""
    print(f"=== TTS Generation Started ===")
    print(f"Input text: '{text}'")
    
    try:
        if model is None or tokenizer is None:
            raise Exception("TTS model not loaded properly")
        
        print("Processing text...")
        processed_text = preprocess_text(text.strip())
        print(f"Processed text: '{processed_text}'")
        
        print("Tokenizing...")
        inputs = tokenizer(processed_text, return_tensors="pt")
        print(f"Tokenized successfully, input_ids shape: {inputs['input_ids'].shape}")
        
        print("Generating audio...")
        with torch.no_grad():
            output = model(**inputs).waveform
        print(f"Audio generated, shape: {output.shape}")
        
        waveform = output.squeeze().numpy()
        waveform = waveform / np.max(np.abs(waveform))  # normalize
        print(f"Waveform shape: {waveform.shape}")
        
        print("Saving WAV file...")
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()
        
        sampling_rate = getattr(model.config, "sampling_rate", 16000)
        scipy.io.wavfile.write(tmp_wav_path, rate=sampling_rate, data=waveform)
        
        print("Converting to MP3...")
        audio = AudioSegment.from_wav(tmp_wav_path)
        audio.export(str(output_path), format="mp3", bitrate="128k")
        
        # Clean up temporary WAV file
        os.unlink(tmp_wav_path)
        
        print(f"✓ Audio saved to: {output_path}")
        print("=== TTS Generation Completed Successfully ===")
        return True
        
    except Exception as e:
        error_msg = f"Error in TTS: {str(e)}"
        print(f"✗ {error_msg}")
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


@app.route('/tts', methods=['GET'])
def text_to_speech():
    """
    TTS endpoint - converts text to speech
    Query parameters:
      - text: The text to convert to speech (required)
    """
    text = request.args.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "Please provide 'text' parameter"}), 400
    
    # Check cache
    cache_path = get_cache_path(text)
    if cache_path.exists():
        print(f"Cache hit for text: '{text[:50]}...'")
        return send_file(cache_path, mimetype='audio/mpeg')
    
    # Generate job ID
    job_id = hashlib.sha256(f"{text}{os.urandom(8).hex()}".encode()).hexdigest()[:16]
    
    # Start async generation
    thread = Thread(target=generate_audio_async, args=(job_id, text, cache_path))
    thread.daemon = True
    thread.start()
    
    # Wait briefly to see if generation completes quickly
    thread.join(timeout=2.0)
    
    if cache_path.exists():
        # Generation completed within timeout
        return send_file(cache_path, mimetype='audio/mpeg')
    else:
        # Still processing, return job ID for polling
        return jsonify({
            "message": "Audio generation started. This may take a while. Check status using job_id.",
            "job_id": job_id,
            "status_url": f"/status/{job_id}"
        }), 202


@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check status of a TTS generation job"""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    if job["status"] == "completed":
        return send_file(job["path"], mimetype='audio/mpeg')
    elif job["status"] == "failed":
        return jsonify({"error": "Audio generation failed"}), 500
    else:
        return jsonify({"status": "processing"}), 202


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "tts_model_loaded": model is not None and tokenizer is not None,
        "punct_model_loaded": punct_pipe is not None
    })


if __name__ == '__main__':
    print("Starting TimToSpeech service...")
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

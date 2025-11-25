import os
import tempfile
import time
import traceback
import numpy as np
import scipy.io.wavfile
import torch
from transformers import VitsModel, AutoTokenizer
from pydub import AudioSegment
from pathlib import Path

from .config import log
from .utils import preprocess_text

# Global variables for models
model = None
tokenizer = None


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

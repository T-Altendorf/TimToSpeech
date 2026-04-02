import os
import time
import requests
import traceback
import json
import base64
from pathlib import Path
from pydub import AudioSegment
from .config import log


def call_kurdish_tts_api(text: str, output_path: Path) -> bool:
    """
    Call the Kurdish TTS API (SSE stream)

    Args:
        text: The text to convert to speech
        output_path: Path where the audio file should be saved

    Returns:
        True if successful, False otherwise
    """
    log(f"=== Kurdish TTS API Call Started ===")
    log(f"Input: '{text}'")
    start_time = time.perf_counter()

    try:
        url = "https://www.kurdishtts.com/api/tts-demo"
        payload = {
            "text": text,
            "dialect": "kurmanji",
            "voice": "kurmanji_236",
            "model_version": "v4",
            "stream_format": "sse",
        }

        log(f"Calling Kurdish TTS API...")
        response = requests.post(url, json=payload, timeout=30, stream=True)
        response.raise_for_status()

        audio_content = b""

        # Iterate over the response lines to handle SSE
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                data_str = None
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                elif decoded_line.startswith("message | "):
                    data_str = decoded_line[10:]
                elif decoded_line.startswith("{"):
                    data_str = decoded_line

                if not data_str:
                    continue

                # Check for [DONE] or similar end markers if applicable,
                # though usually we just process until the stream ends.
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    event_type = data.get("type")

                    if event_type == "speech.audio.delta":
                        audio_b64 = data.get("audio")
                        if audio_b64:
                            audio_content += base64.b64decode(audio_b64)

                    elif event_type == "speech.audio.done":
                        # We could log usage stats here if needed
                        usage = data.get("usage", {})
                        log(f"API Usage: {usage}")

                except json.JSONDecodeError:
                    log(
                        f"Warning: Could not decode JSON from SSE line: {data_str[:50]}..."
                    )
                    continue

        if not audio_content:
            log("✗ No audio content received from API")
            return False

        # Convert raw PCM to MP3
        # The API returns raw PCM 16-bit mono at 22050 Hz
        try:
            audio = AudioSegment(
                data=audio_content,
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=22050,
                channels=1,
            )
            audio.export(str(output_path), format="mp3", bitrate="128k")
        except Exception as e:
            log(f"✗ Error converting audio: {e}")
            # Fallback: save raw data if conversion fails (though it won't play as mp3)
            with open(output_path, "wb") as f:
                f.write(audio_content)

        file_size = os.path.getsize(output_path)
        elapsed = time.perf_counter() - start_time

        log(
            f"✓ Kurdish TTS API succeeded | Size: {file_size} bytes | Time: {elapsed:.3f}s"
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

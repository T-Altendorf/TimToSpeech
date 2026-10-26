import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log(message: str):
    """Log message with timestamp"""
    logging.info(message)


# this is for local development; in production, use environment variables
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TTS_WAIT_TIMEOUT = float(os.getenv("TTS_WAIT_TIMEOUT", "7.0"))

# Only needed for sentences longer than the free endpoint's 150 char limit
KURDISH_TTS_API_KEY = os.getenv("KURDISH_TTS_API_KEY", "").strip()

# The speech-to-text key at kurdishtts.com: a separate key, the TTS key gets 401 there.
KURDISH_TTS_STT_KEY = os.getenv("KURDISH_TTS_STT_KEY", "").strip()

# Browser clients (Expo web, web frontends) need CORS headers to read responses.
# Comma-separated list of allowed origins; "*" allows any.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# The night lane (`app/night.py`): bulk warming on the free endpoint inside a
# nightly window. `NIGHT_WINDOW` is HH:MM-HH:MM in `NIGHT_TZ`, or `always`.
NIGHT_WINDOW = os.getenv("NIGHT_WINDOW", "22:00-07:00").strip()
NIGHT_TZ = os.getenv("NIGHT_TZ", "Europe/Berlin").strip()

# Bearer key for POST/GET /warm/bulk; unset, the endpoint answers 503.
WARM_BULK_KEY = os.getenv("WARM_BULK_KEY", "").strip()

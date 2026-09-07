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

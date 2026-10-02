from app.main import app
from app.tts_local import load_models
from app.config import log
from app.alignment_sweep import start_sweep

# Load models when starting with Gunicorn
log("Starting TimToSpeech service via WSGI...")
load_models()
# Check the word times of every clip cached before the alignment check existed.
start_sweep()

if __name__ == "__main__":
    app.run()

from app.main import app
from app.tts_local import load_models
from app.config import log

# Load models when starting with Gunicorn
log("Starting TimToSpeech service via WSGI...")
load_models()

if __name__ == "__main__":
    app.run()

# TimToSpeech

A lightweight Docker-based Text-to-Speech (TTS) service for Kurmanji text using the `facebook/mms-tts-kmr-script_latin` model from Hugging Face.

## Features

- 🎯 **Lightweight Flask-based API** - Simple and efficient web service
- 🗣️ **Kurmanji TTS** - Uses Facebook's MMS TTS model for Kurmanji (Latin script)
- 💾 **Smart Caching** - Automatically caches generated audio files to avoid regeneration
- 🎵 **MP3 Output** - Converts audio to MP3 format for smaller file sizes
- 📝 **Text Preprocessing** - Handles numbers, abbreviations, and punctuation restoration
- 🐳 **Docker Compose** - Easy deployment with persistent volume storage
- ⚡ **Fast Response** - Models loaded into memory at startup for quick generation

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of RAM (models are loaded into memory)
- Internet connection for initial model downloads

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/T-Altendorf/TimToSpeech.git
   cd TimToSpeech
   ```

2. **Build and start the service:**
   ```bash
   make build
   make up
   ```

   Or without make:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. **Check the logs to ensure models are loaded:**
   ```bash
   make logs
   ```

   Wait until you see:
   ```
   ✓ Punctuation model loaded successfully
   ✓ TTS model loaded successfully
   ```

4. **Test the service:**
   ```bash
   curl "http://localhost:5000/tts?text=Silav%20ji%20te%20re" --output test.mp3
   ```

### Using Makefile Commands

The project includes a Makefile for convenient operations:

```bash
make build       # Build the Docker image
make up          # Start the service
make down        # Stop the service
make logs        # View service logs
make restart     # Restart the service
make clean       # Stop and remove volumes
make test-local  # Run locally without Docker (for development)
make help        # Show all available commands
```

## API Endpoints

### `GET /tts`

Convert text to speech.

**Query Parameters:**
- `text` (required): The Kurmanji text to convert to speech

**Response:**
- `200 OK`: Returns MP3 audio file (from cache if available)
- `202 Accepted`: Generation started, returns job info for polling
- `400 Bad Request`: Missing or invalid text parameter

**Example:**
```bash
curl "http://localhost:5000/tts?text=Rojbaş" --output greeting.mp3
```

**Long-running requests:**
For first-time generation of complex text, the service may take time. The endpoint will:
1. Return immediately with a 202 status and job_id if processing takes longer than 2 seconds
2. Continue processing in the background even if the client disconnects
3. Cache the result for future requests

**Example with polling:**
```bash
# Initial request (may return 202 with job_id)
curl "http://localhost:5000/tts?text=Very%20long%20text..."

# Response:
# {
#   "message": "Audio generation started...",
#   "job_id": "abc123...",
#   "status_url": "/status/abc123..."
# }

# Poll for completion
curl "http://localhost:5000/status/abc123..." --output result.mp3
```

### `GET /status/<job_id>`

Check the status of a TTS generation job.

**Response:**
- `200 OK`: Returns MP3 audio file if completed
- `202 Accepted`: Still processing
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Generation failed

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "tts_model_loaded": true,
  "punct_model_loaded": true
}
```

## Configuration

### Environment Variables

Configuration is done via environment variables. Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

**Available Variables:**
- `CACHE_DIR`: Directory path for caching generated audio files (default: `/app/cache`)

### Volume Persistence

The Docker Compose configuration includes a named volume (`tts-cache`) that persists generated audio files between container restarts. This means:
- Cached audio files survive container restarts
- No need to regenerate previously requested text
- Faster response times for repeated requests

## Architecture

### Text Preprocessing Pipeline

The service applies several preprocessing steps before TTS generation:

1. **Text Normalization**: Standardizes quotation marks and apostrophes
2. **Number Expansion**: Converts digits to Kurmanji words (e.g., "1" → "yek")
3. **Abbreviation Expansion**: Expands common abbreviations (e.g., "NATO" → "Nato")
4. **Punctuation Restoration**: Uses ML model to restore proper punctuation

### Caching Strategy

- Audio files are cached using SHA-256 hash of the input text
- Cache directory is persisted via Docker volume
- Identical requests return cached files instantly

### Async Processing

- Long-running generations continue in background threads
- Client can disconnect without stopping generation
- Job status tracking via job_id for polling

## Development

### Running Locally

For development without Docker:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install ffmpeg** (required for MP3 conversion):
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

3. **Create cache directory:**
   ```bash
   mkdir -p cache
   export CACHE_DIR=./cache
   ```

4. **Run the service:**
   ```bash
   python app.py
   ```

The service will be available at `http://localhost:5000`.

### Project Structure

```
TimToSpeech/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
├── Makefile              # Convenience commands
├── .env.example          # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Models Used

- **TTS Model**: `facebook/mms-tts-kmr-script_latin` - Facebook's Massively Multilingual Speech model for Kurmanji
- **Punctuation Model**: `oliverguhr/fullstop-punctuation-multilang-large` - Multilingual punctuation restoration

Both models are automatically downloaded on first run and loaded into memory for fast inference.

## Troubleshooting

### Models not loading

If you see errors about models not loading:
1. Check internet connection (models download from Hugging Face on first run)
2. Ensure sufficient disk space (models are ~500MB total)
3. Check logs: `make logs`

### Out of memory errors

The models require significant RAM:
- Increase Docker memory limit (Docker Desktop → Settings → Resources)
- Minimum recommended: 4GB RAM

### Audio generation fails

1. Check that ffmpeg is installed in the container
2. Verify cache directory is writable
3. Check logs for specific error messages

### Port already in use

If port 5000 is already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "5001:5000"  # Use port 5001 instead
```

## License

This project uses models from Hugging Face. Please check individual model licenses:
- [facebook/mms-tts-kmr-script_latin](https://huggingface.co/facebook/mms-tts-kmr-script_latin)
- [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

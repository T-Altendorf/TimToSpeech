# TimToSpeech

A lightweight Docker-based Text-to-Speech (TTS) service for Kurmanji text using the `facebook/mms-tts-kmr-script_latin` model from Hugging Face.

## Features

- 🎯 **Lightweight Flask-based API** - Simple and efficient web service
- 🗣️ **Kurdish TTS API with parallel chunking** - long text is split on sentence boundaries, generated in parallel and merged seamlessly; local Facebook MMS model as fallback
- 💾 **Smart Caching** - Automatically caches generated audio files to avoid regeneration
- 🎵 **MP3 Output** - Converts audio to MP3 format for smaller file sizes
- 📝 **Text Preprocessing** - Handles numbers and abbreviations
- 🐳 **Docker Compose** - Easy deployment with persistent volume storage
- ⚡ **Fast Response** - Models loaded into memory at startup for quick generation
- 🔄 **Automatic Fallback** - Falls back to local model if Kurdish TTS API fails

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of RAM (models are loaded into memory)
- Internet connection for initial model downloads

## Quick Start

### Using Docker Compose (Recommended)

The service can be run in two modes:

- **Development mode** (with host port exposed): For local testing and development
- **Production mode** (no host port): For deployment behind a reverse proxy/load balancer

#### Development Mode (Local Testing)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/T-Altendorf/TimToSpeech.git
   cd TimToSpeech
   ```

2. **Build and start the service:**

   ```bash
   make dev
   ```

   Or without make:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
   ```

3. **Check the logs to ensure models are loaded:**

   ```bash
   make dev-logs
   ```

   Wait until you see:

   ```
   ✓ TTS model loaded successfully
   ```

4. **Test the service:**
   ```bash
   curl "http://localhost:8000/tts?text=Silav%20ji%20te%20re" --output test.mp3
   ```

#### Production Mode (Behind Reverse Proxy)

For production deployments (Dokploy, Kubernetes, etc.):

```bash
make build
make up
```

This mode only exposes the port internally (using Docker's `expose` directive), expecting an external reverse proxy or orchestrator to handle external access.

### Using Makefile Commands

The project includes a Makefile for convenient operations:

**Production mode (no host port exposure):**

```bash
make build       # Build the Docker image
make up          # Start the service (no host port)
make down        # Stop the service
make logs        # View service logs
make restart     # Restart the service
make clean       # Stop and remove volumes
```

**Development mode (with host port exposure):**

```bash
make dev         # Start service with host port exposed
make dev-down    # Stop dev service
make dev-logs    # View dev service logs
make dev-restart # Restart dev service
make dev-build   # Build dev image
```

**Other commands:**

```bash
make test-local  # Run locally without Docker (for development)
make help        # Show all available commands
```

## API Endpoints

### `GET /tts`

Convert text to speech.

**Query Parameters:**

- `text` (required): The Kurmanji text to convert to speech
- `force_regen` (optional): If true (`1`, `true`, `yes`, `on`), delete cached audio for this exact text and regenerate

**Response:**

- `200 OK`: Returns MP3 audio file (from cache if available)
- `202 Accepted`: Generation started, returns job info for polling
- `400 Bad Request`: Missing or invalid text parameter

**Example:**

```bash
curl "http://localhost:8000/tts?text=Rojbaş" --output greeting.mp3
```

**Example (force regeneration):**

```bash
curl --get \
   --data-urlencode "text=dîtin (bîn- ; dît-)" \
   --data-urlencode "force_regen=true" \
   "http://localhost:8000/tts" \
   --output greeting-fresh.mp3
```

**Long-running requests:**
For first-time generation of complex text, the service may take time. The endpoint will:

1. Return immediately with a 202 status and job_id if processing takes longer than the configured timeout (default 7 seconds)
2. Continue processing in the background even if the client disconnects
3. Cache the result for future requests

**How it works:**

- Texts up to 150 characters go to the free Kurdish TTS endpoint (https://www.kurdishtts.com/api/tts-demo) in a single request
- Longer texts are split on sentence boundaries and packed back into chunks of at most 150 characters, which are generated in parallel and stitched into one seamless track
- Only a single sentence longer than 150 characters goes to the authenticated endpoint (https://www.kurdishtts.com/api/tts-proxy), which needs `KURDISH_TTS_API_KEY`
- If the Kurdish TTS API fails, the service automatically falls back to the local model

**Example with polling:**

```bash
# Initial request (may return 202 with job_id)
curl "http://localhost:8000/tts?text=Very%20long%20text..."

# Response:
# {
#   "message": "Audio generation started...",
#   "job_id": "abc123...",
#   "status_url": "/status/abc123..."
# }

# Poll for completion
curl "http://localhost:8000/status/abc123..." --output result.mp3
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
  "tts_model_loaded": true
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
- `PORT`: Port number for the Flask service (default: `8000`)
- `TTS_WAIT_TIMEOUT`: Maximum time in seconds to wait for TTS generation before returning a job ID for polling (default: `7.0`)
- `HOST_PORT`: Port exposed on the host machine for local development (default: `8000`)
- `CORS_ORIGINS`: Comma-separated list of origins allowed to call the API from a browser, or `*` for any (default: `*`). Needed for web clients such as Expo web on `http://localhost:8081`; native apps are not subject to CORS.

### Volume Persistence

The Docker Compose configuration includes a named volume (`tts-cache`) that persists generated audio files between container restarts. This means:

- Cached audio files survive container restarts
- No need to regenerate previously requested text
- Faster response times for repeated requests

## Architecture

### TTS Engine Selection

The service intelligently selects between two TTS engines:

1. **Kurdish TTS API** — used for all text, with two endpoints
   - Free (`/api/tts-demo`): every chunk of 150 characters or fewer. No key required.
   - Authenticated (`/api/tts-proxy`): only sentences that are themselves longer
     than 150 characters and so cannot be split any further. Requires
     `KURDISH_TTS_API_KEY` (`x-api-key` header); see https://www.kurdishtts.com/docs/api
   - Voice: `kurmanji_236`, model `v4`
   - Chunks are generated in parallel (up to 4 at a time) and merged seamlessly:
     each chunk is trimmed to its speech, faded to zero at both edges,
     level-matched, and joined across a short silence, so there is no click at
     any boundary
   - Automatic fallback to local model if the API fails

2. **Local Facebook MMS Model** (`facebook/mms-tts-kmr-script_latin`)
   - Fallback option when the Kurdish TTS API is unavailable, or when a long
     sentence needs the authenticated endpoint and no API key is configured
   - Loaded into memory at startup for fast inference

### Text Preprocessing Pipeline

The service applies several preprocessing steps before TTS generation:

1. **Text Normalization**: Standardizes quotation marks and apostrophes
2. **Number Expansion**: Converts digits to Kurmanji words (e.g., "1" → "yek")
3. **Abbreviation Expansion**: Expands common abbreviations (e.g., "NATO" → "Nato")

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

The service will be available at `http://localhost:8000`.

### Project Structure

```
TimToSpeech/
├── app.py                   # Main Flask application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Base Docker Compose configuration (production)
├── docker-compose.dev.yml  # Development override (exposes host port)
├── Makefile                # Convenience commands
├── .env.example            # Environment variable template
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

### Docker Compose Configurations

The project uses Docker Compose with two configurations:

- **`docker-compose.yml`**: Base configuration for production
  - Uses `expose` directive (no host port mapping)
  - Suitable for deployment behind reverse proxies (Dokploy, Kubernetes, etc.)
- **`docker-compose.dev.yml`**: Development override
  - Adds `ports` directive to expose the service on the host
  - Combined with base config for local testing: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up`
  - Accessed via `make dev` commands

## Models Used

- **TTS Model**: `facebook/mms-tts-kmr-script_latin` - Facebook's Massively Multilingual Speech model for Kurmanji

The TTS model is automatically downloaded on first run and loaded into memory for fast inference.

## Troubleshooting

### Models not loading

If you see errors about models not loading:

1. Check internet connection (the model downloads from Hugging Face on first run)
2. Ensure sufficient disk space (the model is ~500MB)
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

If port 8000 is already in use on your host, you can change it:

1. **For development mode**: Create or edit `.env` file and set:

   ```bash
   HOST_PORT=8001  # Use a different host port
   PORT=8000       # Keep container port the same
   ```

2. **Or modify `docker-compose.dev.yml`**:
   ```yaml
   ports:
     - "8001:8000" # Use port 8001 on host instead
   ```

## License

This project uses models from Hugging Face. Please check individual model licenses:

- [facebook/mms-tts-kmr-script_latin](https://huggingface.co/facebook/mms-tts-kmr-script_latin)
- [facebook/mms-tts-kmr-script_latin](https://huggingface.co/facebook/mms-tts-kmr-script_latin)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

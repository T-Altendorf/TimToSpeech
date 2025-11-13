FROM python:3.11-slim

# Install ffmpeg for audio conversion and curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create cache directory
RUN mkdir -p /app/cache

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

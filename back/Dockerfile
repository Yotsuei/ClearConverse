# back/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PyTorch and audio processing libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p models processed_audio temp_uploads
VOLUME ["/app/models", "/app/processed_audio", "/app/temp_uploads"]

# Add a healthcheck endpoint to api.py if it doesn't exist
RUN echo '\n@app.get("/health")\ndef health_check():\n    return {"status": "ok"}\n' >> api.py

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
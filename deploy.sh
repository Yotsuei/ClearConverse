#!/bin/bash
# deploy.sh - Helper script for deploying the application

# Default configuration
MODE=${1:-production}
ENV_FILE=.env.production
COMPOSE_FILE=docker-compose.yml

# Check environment
if [ "$MODE" = "development" ]; then
    ENV_FILE=.env.development
    COMPOSE_FILE=docker-compose.dev.yml
    echo "Using development environment: $ENV_FILE with $COMPOSE_FILE"
else
    echo "Using production environment: $ENV_FILE with $COMPOSE_FILE"
fi

# Load environment variables from .env file
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: $ENV_FILE not found. Make sure to create this file!"
    exit 1
fi

# Print configuration
echo "Deploying ClearConverse in $MODE mode"
echo "API Base URL: $VITE_API_BASE_URL"
echo "WebSocket Base URL: $VITE_WS_BASE_URL"

# Create necessary directories if they don't exist
mkdir -p back/models back/processed_audio back/temp_uploads

# Set proper permissions
chmod -R 755 back/models back/processed_audio back/temp_uploads

# Check if Hugging Face token is set
if [ -z "$HF_AUTH_TOKEN" ]; then
    echo "Warning: HF_AUTH_TOKEN is not set. The application may not function properly."
    echo "Please add your Hugging Face token to your $ENV_FILE file."
fi

# Export environment variables for Docker Compose
export ENV_FILE
export API_HOST
export API_PORT
export CORS_ORIGINS
export MODEL_CACHE_DIR
export HF_AUTH_TOKEN
export VITE_API_BASE_URL
export VITE_WS_BASE_URL
export FRONTEND_PORT

# Build and start the services
echo "Building Docker containers using $COMPOSE_FILE..."
docker-compose -f $COMPOSE_FILE build

echo "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# Provide different messages based on the mode
if [ "$MODE" = "development" ]; then
    echo "ClearConverse development environment is now running!"
    echo "Frontend available at: http://localhost:${FRONTEND_PORT:-3000}"
    echo "Backend API available at: http://localhost:${API_PORT:-8000}"
    echo ""
    echo "The code is mounted as volumes, so any changes you make will be reflected immediately."
    echo "Check the logs with: docker-compose -f $COMPOSE_FILE logs -f"
else
    echo "ClearConverse production deployment completed!"
    echo "Frontend available at: http://localhost:${FRONTEND_PORT:-80}"
    echo "Backend API available at: http://localhost:${API_PORT:-8000}"
fi

# Create a cleanup script
cat > cleanup.sh << EOF
#!/bin/bash
# cleanup.sh - Script to clean up deployment

# Stop and remove containers
docker-compose -f $COMPOSE_FILE down

# Optional cleanup commands (uncomment if needed)
# echo "Do you want to clean up temporary files? (y/n)"
# read response
# if [ "\$response" = "y" ]; then
#   echo "Cleaning up temporary files..."
#   rm -rf back/temp_uploads/*
#   rm -rf back/processed_audio/*
#   echo "Temporary files cleaned up"
# fi

echo "Cleanup completed"
EOF

chmod +x cleanup.sh
echo "Created cleanup script: cleanup.sh"
#!/bin/bash
# deploy.sh - Helper script for deploying the application

# Load environment variables from .env file
set -a
source .env.production
set +a

# Default configuration
ENV_FILE=.env.production
MODE=${1:-production}

# Print configuration
echo "Deploying ClearConverse in $MODE mode"
echo "API Base URL: $REACT_APP_API_BASE_URL"
echo "WebSocket Base URL: $REACT_APP_WS_BASE_URL"

# Create necessary directories if they don't exist
mkdir -p models processed_audio temp_uploads

# Check environment
if [ "$MODE" = "development" ]; then
    ENV_FILE=.env.development
    export ENV_FILE
    echo "Using development environment: $ENV_FILE"
else
    echo "Using production environment: $ENV_FILE"
fi

# Build and start the services
docker-compose build --no-cache
docker-compose up -d

echo "ClearConverse deployment completed!"
echo "Frontend available at: http://localhost:${FRONTEND_PORT:-3000}"
echo "Backend API available at: http://localhost:${API_PORT:-8000}"

# Create a cleanup script
cat > cleanup.sh << 'EOF'
#!/bin/bash
# cleanup.sh - Script to clean up deployment

# Stop and remove containers
docker-compose down

# Remove temporary files (uncomment if needed)
# rm -rf temp_uploads/*
# rm -rf processed_audio/*

echo "Cleanup completed"
EOF

chmod +x cleanup.sh
echo "Created cleanup script: cleanup.sh"
#!/bin/bash
# cleanup.sh - Script to clean up deployment

# Stop and remove containers
docker-compose -f docker-compose.dev.yml down

# Optional cleanup commands (uncomment if needed)
# echo "Do you want to clean up temporary files? (y/n)"
# read response
# if [ "$response" = "y" ]; then
#   echo "Cleaning up temporary files..."
#   rm -rf back/temp_uploads/*
#   rm -rf back/processed_audio/*
#   echo "Temporary files cleaned up"
# fi

echo "Cleanup completed"

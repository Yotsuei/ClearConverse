#!/bin/bash
# dev-setup.sh - Helper script for setting up the development environment

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js 16.x or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install npm."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p back/models back/processed_audio back/temp_uploads
chmod -R 755 back/models back/processed_audio back/temp_uploads

# Create environment files if they don't exist
if [ ! -f ".env.development" ]; then
    echo "Creating .env.development file..."
    cat > .env.development << EOF
# Backend configuration
API_HOST=http://localhost
API_PORT=8000
CORS_ORIGINS=*
MODEL_CACHE_DIR=models
HF_AUTH_TOKEN=your_huggingface_auth_token_here

# Frontend configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
FRONTEND_PORT=3000
EOF
    echo "Please update the HF_AUTH_TOKEN in .env.development with your Hugging Face token."
fi

if [ ! -f "back/.env.development" ]; then
    echo "Creating back/.env.development file..."
    cat > back/.env.development << EOF
API_HOST=http://localhost
API_PORT=8000
CORS_ORIGINS=*
MODEL_CACHE_DIR=models
HF_AUTH_TOKEN=your_huggingface_auth_token_here
EOF
    echo "Please update the HF_AUTH_TOKEN in back/.env.development with your Hugging Face token."
fi

if [ ! -f "front/.env.development" ]; then
    echo "Creating front/.env.development file..."
    cat > front/.env.development << EOF
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
EOF
fi

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
cd back
python3 -m venv venv

# Activate the virtual environment based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Go back to project root
cd ..

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
cd front
npm install
cd ..

echo "Development environment setup complete!"
echo ""
echo "To start the backend:"
echo "  cd back"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  source venv/Scripts/activate"
    echo "  set ENV_FILE=.env.development"
else
    echo "  source venv/bin/activate"
    echo "  export ENV_FILE=.env.development"
fi
echo "  python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To start the frontend:"
echo "  cd front"
echo "  npm run dev"
echo ""
echo "Don't forget to update your Hugging Face token in the .env.development files!"
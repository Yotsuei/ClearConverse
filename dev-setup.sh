#!/bin/bash
# dev-setup.sh - Helper script for setting up the ClearConverse development environment

# Set up error handling
set -e  # Exit on any error
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Output formatting
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}     ClearConverse Development Environment Setup     ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Check prerequisites
echo -e "\n${BLUE}Checking prerequisites...${NC}"

# Check Python version - first try python3, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # On Windows, often only "python" is available
    PYTHON_CMD="python"
    
    # Verify it's Python 3
    PY_VERSION_CHECK=$(python --version 2>&1)
    if [[ ! $PY_VERSION_CHECK == Python\ 3* ]]; then
        echo -e "${RED}❌ Python command exists but it's not Python 3. Please install Python 3.${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.12 or higher.${NC}"
    exit 1
fi

# Get Python version
PY_VERSION=$($PYTHON_CMD --version | awk '{print $2}')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ $PY_MAJOR -lt 3 ] || [ $PY_MAJOR -eq 3 -a $PY_MINOR -lt 10 ]; then
    echo -e "${YELLOW}WARNING: Python version $PY_VERSION detected. ClearConverse recommends Python 3.12.${NC}"
    echo -e "Would you like to continue anyway? (y/n)"
    read response
    if [ "$response" != "y" ]; then
        echo "Setup aborted. Please install Python 3.12 or higher."
        exit 1
    fi
else
    echo -e "✅ Python $PY_VERSION"
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d 'v' -f 2)
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
    
    if [ $NODE_MAJOR -lt 18 ]; then
        echo -e "${YELLOW}WARNING: Node.js version $NODE_VERSION detected. ClearConverse recommends Node.js 18.x or higher.${NC}"
        echo -e "Would you like to continue anyway? (y/n)"
        read response
        if [ "$response" != "y" ]; then
            echo "Setup aborted. Please upgrade Node.js."
            exit 1
        fi
    else
        echo -e "✅ Node.js $NODE_VERSION"
    fi
else
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 18.x or higher.${NC}"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "✅ npm $NPM_VERSION"
else
    echo -e "${RED}❌ npm is not installed. Please install npm.${NC}"
    exit 1
fi

# Check FFmpeg (essential for audio processing)
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n 1 | awk '{print $3}')
    echo -e "✅ FFmpeg $FFMPEG_VERSION"
else
    echo -e "${YELLOW}WARNING: FFmpeg is not installed, but it's required for audio processing.${NC}"
    echo -e "Would you like to continue anyway? (y/n)"
    read response
    if [ "$response" != "y" ]; then
        echo "Setup aborted. Please install FFmpeg."
        exit 1
    fi
fi

# Create project directories
echo -e "\n${BLUE}Creating necessary directories...${NC}"
mkdir -p back/models back/processed_audio back/temp_uploads
chmod -R 755 back/models back/processed_audio back/temp_uploads
echo -e "✅ Directories created and permissions set"

# Create environment files
echo -e "\n${BLUE}Setting up environment files...${NC}"

# Ask for Hugging Face token
echo -e "${YELLOW}A Hugging Face token is required for downloading AI models.${NC}"
read -p "Enter your Hugging Face token (press Enter to skip for now): " HF_TOKEN
HF_TOKEN=${HF_TOKEN:-your_huggingface_auth_token_here}

# Root .env.development file
if [ ! -f ".env.development" ]; then
    echo -e "Creating .env.development file..."
    cat > .env.development << EOF
# Backend configuration
API_HOST=http://localhost
API_PORT=8000
CORS_ORIGINS=*
MODEL_CACHE_DIR=models
HF_AUTH_TOKEN=$HF_TOKEN

# Frontend configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
FRONTEND_PORT=3000
EOF
    echo -e "✅ .env.development created"
else
    echo -e "✅ .env.development already exists"
fi

# Backend .env.development file
if [ ! -f "back/.env.development" ]; then
    echo -e "Creating back/.env.development file..."
    cat > back/.env.development << EOF
API_HOST=http://localhost
API_PORT=8000
CORS_ORIGINS=*
MODEL_CACHE_DIR=models
HF_AUTH_TOKEN=$HF_TOKEN
EOF
    echo -e "✅ back/.env.development created"
else
    echo -e "✅ back/.env.development already exists"
fi

# Frontend .env.development file
if [ ! -f "front/.env.development" ]; then
    echo -e "Creating front/.env.development file..."
    cat > front/.env.development << EOF
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
EOF
    echo -e "✅ front/.env.development created"
else
    echo -e "✅ front/.env.development already exists"
fi

# Set up Python virtual environment
echo -e "\n${BLUE}Setting up Python virtual environment...${NC}"
cd back || { echo -e "${RED}❌ Could not find 'back' directory${NC}"; exit 1; }

# Check if venv already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Do you want to recreate it? (y/n)${NC}"
    read response
    if [ "$response" = "y" ]; then
        echo -e "Removing existing virtual environment..."
        rm -rf venv
        echo -e "Creating new virtual environment..."
        $PYTHON_CMD -m venv venv || { 
            echo -e "${RED}❌ Failed to create virtual environment with $PYTHON_CMD${NC}"
            echo -e "${YELLOW}Trying alternative methods...${NC}"
            
            # Try with virtualenv if venv fails
            if command -v pip &> /dev/null; then
                echo -e "Installing virtualenv..."
                pip install virtualenv
                echo -e "Creating virtual environment with virtualenv..."
                virtualenv venv || {
                    echo -e "${RED}❌ Failed to create virtual environment with virtualenv${NC}"
                    echo -e "${YELLOW}Please create virtual environment manually:${NC}"
                    echo -e "  cd back"
                    echo -e "  python -m venv venv"
                    exit 1
                }
            else
                echo -e "${RED}❌ Could not find pip to install virtualenv${NC}"
                exit 1
            fi
        }
    else
        echo -e "Using existing virtual environment."
    fi
else
    echo -e "Creating virtual environment..."
    $PYTHON_CMD -m venv venv || { 
        echo -e "${RED}❌ Failed to create virtual environment with $PYTHON_CMD${NC}"
        echo -e "${YELLOW}Trying alternative methods...${NC}"
        
        # Try with virtualenv if venv fails
        if command -v pip &> /dev/null; then
            echo -e "Installing virtualenv..."
            pip install virtualenv
            echo -e "Creating virtual environment with virtualenv..."
            virtualenv venv || {
                echo -e "${RED}❌ Failed to create virtual environment with virtualenv${NC}"
                echo -e "${YELLOW}Please create virtual environment manually:${NC}"
                echo -e "  cd back"
                echo -e "  python -m venv venv"
                exit 1
            }
        else
            echo -e "${RED}❌ Could not find pip to install virtualenv${NC}"
            exit 1
        fi
    }
fi

# Activate the virtual environment
echo -e "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash, MSYS2, Command Prompt, etc.)
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate || { 
            echo -e "${RED}❌ Failed to activate virtual environment with source command${NC}"
            echo -e "${YELLOW}This is common in some Windows environments.${NC}"
            echo -e "${YELLOW}Please run these commands manually after script completion:${NC}"
            echo -e "  cd back"
            echo -e "  .\\venv\\Scripts\\activate"
            VENV_FAILED=true
        }
    else
        echo -e "${RED}❌ Activation script not found at expected location${NC}"
        echo -e "${YELLOW}Please activate the virtual environment manually:${NC}"
        echo -e "  cd back"
        echo -e "  .\\venv\\Scripts\\activate"
        VENV_FAILED=true
    fi
else
    # Linux, macOS, etc.
    source venv/bin/activate || { 
        echo -e "${RED}❌ Failed to activate virtual environment${NC}"
        echo -e "${YELLOW}Please activate the virtual environment manually:${NC}"
        echo -e "  cd back"
        echo -e "  source venv/bin/activate"
        VENV_FAILED=true
    }
fi

# Only proceed with pip upgrade if activation was successful
if [ "$VENV_FAILED" != "true" ]; then
    # Upgrade pip if needed
    echo -e "Upgrading pip..."
    pip install --upgrade pip || {
        echo -e "${YELLOW}WARNING: Failed to upgrade pip, continuing with existing version${NC}"
    }
else
    echo -e "${YELLOW}Skipping pip upgrade due to virtual environment activation failure${NC}"
fi

# Install Python dependencies with timeout and retries
echo -e "\n${BLUE}Installing Python dependencies...${NC}"
echo -e "${YELLOW}This might take some time as it downloads several AI models and libraries.${NC}"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found in the 'back' directory${NC}"
    exit 1
fi

# Only proceed with dependency installation if venv activation was successful
if [ "$VENV_FAILED" != "true" ]; then
    MAX_RETRIES=3
    RETRY=0
    SUCCESS=false

    while [ $RETRY -lt $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
        echo -e "Installation attempt $(($RETRY + 1))/$MAX_RETRIES..."
        if pip install -r requirements.txt; then
            SUCCESS=true
            echo -e "${GREEN}✅ Python dependencies installed successfully!${NC}"
        else
            RETRY=$((RETRY + 1))
            if [ $RETRY -lt $MAX_RETRIES ]; then
                echo -e "${YELLOW}Installation failed. Retrying in 5 seconds...${NC}"
                sleep 5
            else
                echo -e "${RED}❌ Failed to install Python dependencies after $MAX_RETRIES attempts.${NC}"
                echo -e "${YELLOW}You may need to manually install them later:${NC}"
                echo -e "  cd back"
                if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
                    echo -e "  .\\venv\\Scripts\\activate"
                else
                    echo -e "  source venv/bin/activate" 
                fi
                echo -e "  pip install -r requirements.txt"
            fi
        fi
    done
else
    echo -e "${YELLOW}Skipping Python dependency installation due to virtual environment activation failure${NC}"
    echo -e "${YELLOW}Please install dependencies manually after activating the virtual environment:${NC}"
    echo -e "  cd back"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo -e "  .\\venv\\Scripts\\activate"
    else
        echo -e "  source venv/bin/activate" 
    fi
    echo -e "  pip install -r requirements.txt"
fi

# Go back to project root
cd ..

# Install Node.js dependencies
echo -e "\n${BLUE}Installing Node.js dependencies...${NC}"
cd front || { echo -e "${RED}❌ Could not find 'front' directory${NC}"; exit 1; }

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json not found in the 'front' directory${NC}"
    exit 1
fi

echo -e "Installing frontend dependencies..."
if npm install; then
    echo -e "${GREEN}✅ Node.js dependencies installed successfully!${NC}"
else
    echo -e "${RED}❌ Failed to install Node.js dependencies.${NC}"
    echo -e "${YELLOW}You may need to manually install them later:${NC}"
    echo -e "  cd front"
    echo -e "  npm install"
fi

# Go back to project root
cd ..

# Print setup summary
echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}ClearConverse development environment setup complete!${NC}"
echo -e "${BLUE}====================================================${NC}"

echo -e "\n${BLUE}NEXT STEPS:${NC}"
echo -e "\n1. To start the backend:"
echo -e "   cd back"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo -e "   .\\venv\\Scripts\\activate"
    echo -e "   set ENV_FILE=.env.development"
else
    echo -e "   source venv/bin/activate"
    echo -e "   export ENV_FILE=.env.development"
fi
echo -e "   python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000"

echo -e "\n2. To start the frontend (in a different terminal):"
echo -e "   cd front"
echo -e "   npm run dev"

echo -e "\n3. Access the application at http://localhost:5173"

if [ "$HF_TOKEN" == "your_huggingface_auth_token_here" ]; then
    echo -e "\n${YELLOW}IMPORTANT:${NC} You need to update your Hugging Face token in the .env.development files!"
    echo -e "Get your token from: https://huggingface.co/settings/tokens"
fi

# Special notes for Windows users
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo -e "\n${YELLOW}WINDOWS USERS NOTE:${NC}"
    echo -e "If you encounter issues with the 'python' command not being found:"
    echo -e "1. Ensure Python is in your PATH environment variable"
    echo -e "2. Try using 'py' instead of 'python' in commands"
    echo -e "3. You might need to use the full path to Python, e.g.:"
    echo -e "   C:\\Python312\\python.exe -m uvicorn api:app --reload --host 0.0.0.0 --port 8000"
fi

echo -e "\n${BLUE}====================================================${NC}"
echo -e "${BLUE}               Happy Development!                     ${NC}"
echo -e "${BLUE}====================================================${NC}"
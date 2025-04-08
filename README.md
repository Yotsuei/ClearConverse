# ClearConverse

ClearConverse is a speech transcription tool powered by Whisper-RESepFormer solution to offer quality transcription even in overlapping speech scenarios.

![ClearConverse Screenshot](https://via.placeholder.com/800x450.png?text=ClearConverse+Screenshot)

## Features

- **Multiple Audio Source Options**:
  - Upload audio files for transcription
  - Process audio from Google Drive URLs
- **Advanced Speech Processing**:
  - Accurate speaker diarization (speaker identification)
  - Overlapping speech detection and separation
  - Enhanced audio processing with noise reduction
- **User-Friendly Interface**:
  - Audio preview with playback controls
  - Real-time processing status via WebSockets
  - Speaker-separated transcript display
- **Versatile Output Options**:
  - Downloadable transcription results
  - Copy-to-clipboard functionality
  - Detailed audio statistics

## Prerequisites

Before you begin, ensure you have the following installed:
- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) (3.12 or higher)
- [Node.js](https://nodejs.org/) (18.x or higher)
- [npm](https://www.npmjs.com/get-npm) (usually included with Node.js)
- [Docker](https://www.docker.com/products/docker-desktop/) (optional, for containerized deployment)
- [FFmpeg](https://ffmpeg.org/download.html) (required for audio processing)

You'll also need a Hugging Face account and API token for accessing the required AI models.

## Models

This project uses multiple AI models to process audio:

### Speech Processing Models

- **Whisper**: For speech recognition and transcription
- **RESepFormer**: For speech separation in overlapping speech segments
- **PyAnnote**: For speaker diarization, voice activity detection, and speaker embeddings

### Local Fine-tuned Models (Optional)

For our specific research implementation, you can use custom fine-tuned models:
- Place your fine-tuned Whisper model in `back/models/whisper-ft/`
- Place your fine-tuned RESepFormer model in `back/models/resepformer-ft/`

### Fallback to Pre-trained Models

If no custom models are provided, the application will automatically download and use the pre-trained versions from:
- OpenAI (Whisper small.en model)
- SpeechBrain (RESepFormer base model)
- PyAnnote (Diarization and VAD models)

Note that performance may vary depending on which models are used.

## Getting Started

### Quick Setup (Using Docker)

For a quick setup using Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/clearconverse.git
cd clearconverse

# Create necessary directories
mkdir -p back/models back/processed_audio back/temp_uploads

# Create environment files
# Edit these files to set your Hugging Face token
cp .env.example .env.development
cp .env.example .env.production

# Start the development environment
./deploy.sh development
```

For more details about the Docker setup, refer to the [Docker Setup Documentation](Docker-Setup.md).

### Manual Development Setup

If you prefer to set up the environment manually:

```bash
# Make the script executable
chmod +x dev-setup.sh

# Run the setup script
./dev-setup.sh
```

This script will:
1. Create necessary directories
2. Set up environment files
3. Create a Python virtual environment
4. Install backend dependencies
5. Install frontend dependencies

After running the script, follow the displayed instructions to start the backend and frontend servers.

### Backend Setup (Manual)

1. **Create Environment File**

Create a file named `.env.development` in the `back` directory with the following content:

```
API_HOST=http://localhost
API_PORT=8000
CORS_ORIGINS=*
MODEL_CACHE_DIR=models
HF_AUTH_TOKEN=your_huggingface_auth_token_here
```

Replace `your_huggingface_auth_token_here` with your actual Hugging Face API token. You can get this from your [Hugging Face settings](https://huggingface.co/settings/tokens).

2. **Set Up Virtual Environment**

```bash
cd back
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

This may take a while as it downloads several AI models and libraries.

4. **Create Required Directories**

```bash
mkdir -p models processed_audio temp_uploads
```

### Frontend Setup (Manual)

1. **Create Environment File**

Create a file named `.env.development` in the `front` directory with the following content:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

2. **Install Dependencies**

```bash
cd front
npm install
```

## Running the Application (Manual)

### Start the Backend Server

Make sure your virtual environment is activated, then:

```bash
cd back
export ENV_FILE=.env.development  # On Windows, use: set ENV_FILE=.env.development
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The backend will start and load the necessary models on first run (which may take time).

### Start the Frontend Development Server

In a new terminal:

```bash
cd front
npm run dev
```

This will start the development server, typically on port 5173.

Open your browser and navigate to [http://localhost:5173](http://localhost:5173) to use the application.

## Using ClearConverse

1. **Select Audio Source**
   - Choose between File Upload or Google Drive URL
   - For File Upload: Upload local audio/video files
   - For Google Drive: Provide a shareable link to your audio file

2. **File Formats**
   - Supported formats: WAV, MP3, MP4, OGG, FLAC, M4A, AAC
   - For best results, use WAV or MP3 files

3. **Start Transcription**
   - Once your audio is uploaded, click "Start Transcription"
   - The system will process the audio using WebSocket for real-time progress updates
   - Processing steps include:
     - Building speaker profiles
     - Detecting speech segments
     - Processing overlapping speech
     - Generating transcription

4. **View and Download Results**
   - When processing is complete, you'll see the transcription with speakers labeled
   - Speakers are distinguished by color (SPEAKER_A, SPEAKER_B)
   - Statistics show word count, duration, and speaker turns
   - You can:
     - Download the transcript as a text file
     - Copy to clipboard
     - Clear transcription to process another file

## Troubleshooting

### Common Issues

1. **Models fail to download**
   - Check your Hugging Face token is correct in your environment files
   - Ensure you have internet connectivity
   - Try running with elevated permissions if necessary

2. **Audio processing fails**
   - Verify ffmpeg is installed and in your PATH
   - Check that your audio file is not corrupted
   - Try a different audio format (WAV or MP3 recommended)

3. **WebSocket connection fails**
   - Ensure your backend is running
   - Check browser console for CORS errors
   - Verify the VITE_WS_BASE_URL in your frontend .env file
   - The system will automatically fall back to polling if WebSockets fail

4. **Backend fails to start**
   - Check if port 8000 is already in use
   - Ensure all dependencies are installed
   - Verify your Python version (3.12 or higher required)
   - Check for error messages in the console

5. **Frontend displays only background**
   - Check browser console for errors
   - Verify all dependencies are installed
   - Make sure environment variables are correctly set

### Getting Help

If you encounter issues not covered here, please:
1. Check the console logs in both frontend and backend
2. Look for error messages in the browser developer tools
3. Use the health endpoint to verify backend is running: http://localhost:8000/health
4. File an issue in the project repository with detailed information

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses several open-source technologies:
- [Whisper](https://github.com/openai/whisper) for speech recognition
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [SpeechBrain](https://speechbrain.github.io/) for speech separation
- [React](https://reactjs.org/) and [Vite](https://vitejs.dev/) for the frontend
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
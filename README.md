# ClearConverse

ClearConverse is a speech transcription tool powered by Whisper-RESepFormer solution to offer quality transcription even in overlapping speech scenarios.

![ClearConverse Screenshot](https://via.placeholder.com/800x450.png?text=ClearConverse+Screenshot)

## Features

- Upload audio files for transcription
- Process audio from Google Drive URLs
- Accurate speaker diarization (speaker identification)
- Handle overlapping speech
- Clean audio preview
- Downloadable and copyable transcription results
- Speaker-separated transcripts

## Prerequisites

Before you begin, ensure you have the following installed:
- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) (3.9 or higher)
- [Node.js](https://nodejs.org/) (16.x or higher)
- [npm](https://www.npmjs.com/get-npm) (usually included with Node.js)
- [ffmpeg](https://ffmpeg.org/download.html) (required for audio processing)

You'll also need a Hugging Face account and API token for accessing the required AI models.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/clearconverse.git
cd clearconverse
```

### Backend Setup

1. **Create Environment Files**

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

### Frontend Setup

1. **Create Environment Files**

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

## Running the Application

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

This will start the development server, typically on port 3000.

Open your browser and navigate to [http://localhost:3000](http://localhost:3000) to use the application.

## Using ClearConverse

1. **Upload Audio**
   - Use the File Upload option to upload local audio/video files
   - Supported formats: WAV, MP3, MP4, OGG, FLAC, M4A, AAC
   - For best results, use WAV or MP3 files

2. **Process Google Drive URL**
   - Alternatively, use the Google Drive URL option to process audio from a shared link
   - Make sure the Google Drive link is publicly accessible

3. **Start Transcription**
   - Once your audio is uploaded, click "Start Transcription"
   - The system will process the audio, separating speakers and transcribing content
   - This may take some time depending on the length of the audio

4. **View and Download Results**
   - When processing is complete, you'll see the transcription with speakers labeled
   - You can download the transcript as a text file
   - Copy to clipboard option is also available

## Development

### Environment Configuration

The application uses `.env.development` files for configuration:

- Backend configuration in `back/.env.development`
- Frontend configuration in `front/.env.development`

For production deployment, create corresponding `.env.production` files.

### Project Structure

```
clearconverse/
├── back/                  # Backend (Python/FastAPI)
│   ├── api.py             # Main API endpoints
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Backend Docker configuration
│
├── front/                 # Frontend (React/Vite)
│   ├── src/               # Source code
│   ├── package.json       # NPM configuration
│   └── Dockerfile         # Frontend Docker configuration
│
├── docker-compose.yml     # Docker Compose configuration
└── deploy.sh              # Deployment helper script
```

## Troubleshooting

### Common Issues

1. **Models fail to download**
   - Check your Hugging Face token is correct
   - Ensure you have internet connectivity
   - Try running with elevated permissions if necessary

2. **Audio processing fails**
   - Verify ffmpeg is installed and in your PATH
   - Check that your audio file is not corrupted
   - Try a different audio format

3. **WebSocket connection fails**
   - Ensure your backend is running
   - Check browser console for CORS errors
   - Verify the VITE_WS_BASE_URL in your frontend .env file

4. **Frontend displays only background**
   - Check browser console for errors
   - Verify all dependencies are installed
   - Make sure environment variables are correctly set

### Getting Help

If you encounter issues not covered here, please:
1. Check the console logs in both frontend and backend
2. Look for error messages in the browser developer tools
3. File an issue in the project repository with detailed information

## License

[MIT License](LICENSE)

## Acknowledgements

This project uses several open-source technologies:
- [Whisper](https://github.com/openai/whisper) for speech recognition
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [SpeechBrain](https://speechbrain.github.io/) for speech separation
- [React](https://reactjs.org/) and [Vite](https://vitejs.dev/) for the frontend
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
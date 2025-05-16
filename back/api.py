# back/api.py

# System
import os
import json
import shutil
import logging
import traceback
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import multiprocessing
import psutil

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
from starlette.websockets import WebSocketDisconnect

# Environment and scheduling
from dotenv import load_dotenv

# Audio processing libs
import torch
import torchaudio
import whisper
import noisereduce as nr
import numpy as np
from collections import Counter, defaultdict

# External libraries for diarization, separation and VAD
from pyannote.audio import Pipeline, Inference
from speechbrain.inference import SepformerSeparation

# URL handling
import requests
import tempfile
import platform
import subprocess
from urllib.parse import urlparse
import validators
import re

import warnings

# PDF generation libraries
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Filter specific warnings
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Adjust logging level to ignore warnings
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("whisper").setLevel(logging.ERROR)

# =============================================================================
# Logging setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =============================================================================
# Load Environment
# =============================================================================
def load_environment():
    """Load appropriate environment variables based on deployment mode"""
    env_file = os.getenv('ENV_FILE', '.env.development')
    load_dotenv(env_file)
    logging.info(f"Loaded environment from {env_file}")
    
    # Return a config dictionary with all needed environment variables
    return {
        'api_host': os.getenv('API_HOST', 'http://localhost'),
        'api_port': int(os.getenv('API_PORT', 8000)),
        'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
        'model_cache_dir': os.getenv('MODEL_CACHE_DIR', 'models'),
        'hf_auth_token': os.getenv('HF_AUTH_TOKEN', '')
    }

env_config = load_environment()

# Dictionary to track running transcription processes
active_processes = {}

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AudioSegment:
    start: float
    end: float
    speaker_id: str
    audio_tensor: torch.Tensor
    is_overlap: bool = False
    transcription: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    auth_token: str
    target_sample_rate: int = 16000
    min_segment_duration: float = 0.45  
    overlap_threshold: float = 0.50  
    condition_on_previous_text: bool = True
    merge_gap_threshold: float = 0.50
    min_overlap_duration_for_separation: float = 0.50  
    max_embedding_segments: int = 100  
    enhance_separated_audio: bool = True
    use_vad_refinement: bool = True
    speaker_embedding_threshold: float = 0.40  
    noise_reduction_amount: float = 0.50  
    transcription_batch_size: int = 8
    use_speaker_embeddings: bool = True
    temperature: float = 0.1
    max_speakers: int = 2
    min_speakers: int = 1
    whisper_model_size: str = "small.en"
    transcribe_overlaps_individually: bool = True
    sliding_window_size: float = 0.80  
    sliding_window_step: float = 0.40  
    secondary_diarization_threshold: float = 0.30

# =============================================================================
# PDF Generation Functions
# =============================================================================

def generate_transcript_pdf(transcript_text, output_path, original_filename=None):
    """Generate a PDF from transcript text with script-like formatting, including filename and timestamp."""
    # Create a PDF document with page templates for headers/footers
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles for different parts of the script
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_LEFT,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Italic'],
        fontSize=12,
        alignment=TA_LEFT,
        spaceAfter=24,
        textColor=colors.darkgrey
    )
    
    speaker_style = ParagraphStyle(
        'Speaker',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leftIndent=0,
        spaceAfter=6
    )
    
    dialogue_style = ParagraphStyle(
        'Dialogue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leftIndent=20,
        spaceAfter=12,
        leading=14
    )
    
    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.gray,
        leftIndent=20,
        spaceAfter=2
    )
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=TA_RIGHT
    )
    
    # Create the story elements
    story = []
    
    # Add title with original filename if available
    if original_filename:
        story.append(Paragraph(f"[{original_filename}] Transcript", title_style))
    else:
        story.append(Paragraph("Transcript", title_style))
    
    # Add generation timestamp in subtitle
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on {generation_time}", subtitle_style))
    
    story.append(Spacer(1, 12))
    
    # Parse transcript text
    # Split by double newlines which typically separate utterances
    segments = re.split(r'\n\n|\r\n\r\n', transcript_text)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
            
        # Check if this segment starts with a speaker tag
        speaker_match = re.match(r'(\[SPEAKER_[A-Z]\])(?:\s+(\d+\.\d+s\s+-\s+\d+\.\d+s))?', segment)
        
        if speaker_match:
            # Extract speaker and timestamp if present
            speaker_tag = speaker_match.group(1)
            timestamp = speaker_match.group(2) if speaker_match.group(2) else ""
            
            # Get the actual dialogue (everything after the pattern)
            dialogue_start = speaker_match.end()
            dialogue = segment[dialogue_start:].strip()
            
            # Format speaker tag nicely (SPEAKER_A -> Speaker A)
            formatted_speaker = speaker_tag.replace('[SPEAKER_', 'Speaker ').replace(']', ':')
            
            # Add to story
            story.append(Paragraph(formatted_speaker, speaker_style))
            if timestamp:
                story.append(Paragraph(f"({timestamp})", timestamp_style))
            
            # Handle multi-line dialogue
            for line in dialogue.split('\n'):
                if line.strip():
                    story.append(Paragraph(line.strip(), dialogue_style))
            
            story.append(Spacer(1, 6))
        else:
            # If no speaker tag, just add as normal paragraph
            for line in segment.split('\n'):
                if line.strip():
                    story.append(Paragraph(line.strip(), dialogue_style))
            story.append(Spacer(1, 6))
    
    # Create the page template with footer
    def add_footer(canvas, doc):
        canvas.saveState()
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        p = Paragraph(footer_text, footer_style)
        w, h = p.wrap(doc.width, doc.bottomMargin)
        p.drawOn(canvas, doc.leftMargin, 0.5 * inch)
        canvas.restoreState()
    
    # Build PDF with footer template
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return output_path

# =============================================================================
# Utility Functions 
# =============================================================================

def merge_diarization_segments(segments: List[Tuple[float, float, str]], gap_threshold: float) -> List[Tuple[float, float, str]]:
    if not segments:
        return []
    segments.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end, current_speaker = segments[0]
    for start, end, speaker in segments[1:]:
        if speaker == current_speaker and (start - current_end) <= gap_threshold:
            current_end = end
        else:
            merged.append((current_start, current_end, current_speaker))
            current_start, current_end, current_speaker = start, end, speaker
    merged.append((current_start, current_end, current_speaker))
    return merged

def get_pyannote_vad_intervals(vad_annotation) -> List[Tuple[float, float]]:
    return [(segment.start, segment.end) for segment, _, _ in vad_annotation.itertracks(yield_label=True)]

def refine_segment_with_vad(segment: Tuple[float, float], vad_intervals: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    seg_start, seg_end = segment
    intersections = [
        (max(seg_start, vad_start), min(seg_end, vad_end))
        for vad_start, vad_end in vad_intervals
        if max(seg_start, vad_start) < min(seg_end, vad_end)
    ]
    if not intersections:
        return None
    return (min(s for s, _ in intersections), max(e for _, e in intersections))

def find_segment_overlaps(segments: List[Tuple[float, float, str]]) -> Dict[Tuple[float, float], List[str]]:
    events = []
    for start, end, speaker in segments:
        events.append((start, 1, speaker))
        events.append((end, -1, speaker))
    events.sort(key=lambda x: (x[0], x[1]))
    active_speakers = set()
    overlap_regions = []
    overlap_start = None
    for time, event_type, speaker in events:
        if event_type == 1:
            active_speakers.add(speaker)
            if len(active_speakers) > 1 and overlap_start is None:
                overlap_start = time
        else:
            if len(active_speakers) > 1 and overlap_start is not None:
                overlap_regions.append((overlap_start, time, active_speakers.copy()))
            active_speakers.discard(speaker)
            if len(active_speakers) <= 1:
                overlap_start = None
    return {(start, end): list(speakers) for start, end, speakers in overlap_regions}

def enhance_audio(audio: torch.Tensor, sample_rate: int, stationary: bool = True, prop_decrease: float = 0.75) -> torch.Tensor:
    audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    audio_np = nr.reduce_noise(y=audio_np, sr=sample_rate, stationary=stationary, prop_decrease=prop_decrease)
    if np.max(np.abs(audio_np)) > 0:
        audio_np = audio_np / np.max(np.abs(audio_np))
    return torch.tensor(audio_np, dtype=torch.float32)

# =============================================================================
# URL Handling Utilities
# =============================================================================
        
def download_file_from_url(url, output_path=None):
    """Download a file from a URL and save it to a temporary file if output_path is not provided."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        
        if not output_path:
            # Determine file extension from content type or URL
            file_ext = '.mp3'  # Default extension
            if 'audio/wav' in content_type:
                file_ext = '.wav'
            elif 'audio/mpeg' in content_type or 'audio/mp3' in content_type:
                file_ext = '.mp3'
            elif 'audio/ogg' in content_type:
                file_ext = '.ogg'
            elif 'video/mp4' in content_type:
                file_ext = '.mp4'
            else:
                # Try to get extension from URL
                parsed_url = urlparse(url)
                path = parsed_url.path
                if '.' in path:
                    url_ext = path.split('.')[-1].lower()
                    if url_ext in ['mp3', 'wav', 'ogg', 'mp4']:
                        file_ext = f'.{url_ext}'
            
            # Create a temporary file with the determined extension
            temp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Save the file content
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logging.info(f"Downloaded file from {url} to {output_path}")
        return output_path
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file from URL {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error downloading file from URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error processing URL: {str(e)}")

def download_file_from_google_drive(file_id, output_path=None):
    """
    Download a file from Google Drive using a more robust approach that handles confirmation tokens.
    """
    URL = "https://drive.google.com/uc?export=download"
    
    if not output_path:
        # Create a temporary file with an appropriate extension
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    # Use a session to handle cookies
    session = requests.Session()
    
    # Initial request to get confirmation token if needed
    response = session.get(URL, params={'id': file_id}, stream=True, 
                          headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    
    # Look for the download warning cookie that contains the token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    # If a token is found, a second request with the token is needed
    if token:
        logging.info(f"Obtained confirmation token for Google Drive file {file_id}")
        params = {'id': file_id, 'confirm': token}
    else:
        logging.info(f"No confirmation token needed for Google Drive file {file_id}")
        params = {'id': file_id}
    
    # Download the file with appropriate parameters
    response = session.get(URL, params=params, stream=True, 
                          headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    
    # Check if the response is valid
    if response.status_code != 200:
        error_msg = f"Failed to download file from Google Drive. Status code: {response.status_code}"
        logging.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Check content type to ensure it's getting a file, not an HTML page
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        logging.warning(f"Received HTML content instead of file. This might indicate access restrictions.")
    
    # Write the file to disk
    with open(output_path, 'wb') as f:
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
    
    logging.info(f"Successfully downloaded Google Drive file {file_id} ({total_size} bytes) to {output_path}")
    return output_path

def validate_url(url):
    """Validate if the URL is well-formed and accessible."""
    # Check if URL is well-formed
    if not validators.url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    # Special handling for Google Drive URLs
    if 'drive.google.com' in url:
        # Basic validation for Google Drive URLs
        file_id = None
        file_match = re.search(r'/file/d/([^/]+)', url)
        if file_match:
            file_id = file_match.group(1)
        else:
            open_match = re.search(r'[?&]id=([^&]+)', url)
            if open_match:
                file_id = open_match.group(1)
        
        if not file_id:
            raise HTTPException(status_code=400, detail="Invalid Google Drive URL format. Could not extract file ID.")
        
        return True
    
    # Normal validation for non-Google Drive URLs
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, headers=headers, timeout=10)
        
        # Check if response indicates success (status code 2xx)
        if not response.ok:
            raise HTTPException(status_code=400, 
                              detail=f"URL returned status code {response.status_code}. Make sure the URL is publicly accessible.")
        
        # Check if content type suggests audio or video
        content_type = response.headers.get('Content-Type', '').lower()
        valid_types = ['audio/', 'video/']
        
        is_valid_content = any(t in content_type for t in valid_types)
        
        # Check URL extension content-type can't be determined
        if not is_valid_content:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            valid_extensions = ['.mp3', '.wav', '.ogg', '.mp4', '.flac', '.m4a', '.aac']
            is_valid_content = any(path.endswith(ext) for ext in valid_extensions)
        
        if not is_valid_content:
            logging.warning(f"URL may not point to audio/video content: {content_type}")
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=400, detail="URL request timed out. Server might be slow or unreachable.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=400, detail="Failed to connect to the URL. Please check if the URL is correct and the server is running.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error validating URL: {str(e)}")
    
    return True

def ensure_wav_format(file_path: str) -> str:
    """
    Ensures the audio file is in WAV format.
    If it's an MP3, converts it to WAV and returns the new path.
    Otherwise, returns the original path.
    """
    if not file_path.lower().endswith('.mp3'):
        return file_path
        
    wav_path = file_path.replace('.mp3', '.wav')
    logging.info(f"Converting MP3 to WAV: {file_path} -> {wav_path}")
    
    try:
        # Use FFmpeg directly with subprocess
        cmd = ['ffmpeg', '-y', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path]
        
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"FFmpeg conversion failed: {stderr.decode()}")
            return file_path  # Return original file path if conversion fails
        
        # Verify the WAV file exists and has content
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            logging.info(f"Successfully converted MP3 to WAV: {wav_path}")
            return wav_path
        else:
            logging.error(f"Conversion produced empty or missing file: {wav_path}")
            return file_path
    
    except Exception as e:
        logging.error(f"Error during MP3 conversion: {str(e)}")
        return file_path  # Return original file path if exception occurs

# =============================================================================
# Task Tracking System
# =============================================================================

# State storage
uploaded_files = {}
progress_store = {}
result_store = {}
original_filenames = {}

# =============================================================================
# Enhanced Audio Processor Class
# =============================================================================

class EnhancedAudioProcessor:
    def __init__(self, config: Config, load_models_immediately: bool = False):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler = None
        
        # Model state tracking
        self.models_loaded = {
            'whisper': False,
            'resepformer': False,
            'pyannote': False
        }
        
        # Only load models if specified
        if load_models_immediately:
            self._initialize_models()

    def _initialize_models(self):
        logging.info(f"Initializing models on {self.device}...")
        cache_dir = env_config['model_cache_dir']
        
        # 1. Load fine-tuned RESepFormer model
        self._load_resepformer_model(cache_dir)
        
        # 2. Load fine-tuned Whisper model
        self._load_whisper_model(cache_dir)
        
        # 3. Initialize PyAnnote models
        self._initialize_pyannote_models(cache_dir)
        
        logging.info("Models initialized successfully!")

    # Add more granular loading methods that update progress
    def load_models_with_progress(self, progress_callback=None):
        """Load models with progress updates"""
        try:
            if progress_callback:
                progress_callback(5, "Initializing model environment...")
                
            # Load RESepFormer (25% of loading time)
            if not self.models_loaded['resepformer']:
                if progress_callback:
                    progress_callback(10, "Loading RESepFormer...")
                self._load_resepformer_model(env_config['model_cache_dir'])
                self.models_loaded['resepformer'] = True
            
            # Load Whisper (25% of loading time)
            if not self.models_loaded['whisper']:
                if progress_callback:
                    progress_callback(35, "Loading Whisper...")
                self._load_whisper_model(env_config['model_cache_dir'])
                self.models_loaded['whisper'] = True
            
            # Load PyAnnote (50% of loading time - these are larger)
            if not self.models_loaded['pyannote']:
                if progress_callback:
                    progress_callback(60, "Loading speaker diarization tool...")
                self._initialize_pyannote_models(env_config['model_cache_dir'])
                self.models_loaded['pyannote'] = True
            
            if progress_callback:
                progress_callback(90, "Models loaded, preparing for processing...")
            
            return True
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            if progress_callback:
                progress_callback(100, f"Error loading models: {str(e)}")
            return False
    
    def models_are_loaded(self):
        return all(self.models_loaded.values())

    def _load_whisper_model(self, cache_dir):
        """Load the fine-tuned Whisper model if available, otherwise use the base model"""
        logging.info("Loading Whisper model...")
        whisper_path = os.path.join(cache_dir, "whisper")
        model_dir = os.path.join(whisper_path, self.config.whisper_model_size)
        
        try:
            # Load base model first
            base_model = whisper.load_model(
                self.config.whisper_model_size, 
                download_root=whisper_path
            )
            
            # Check for fine-tuned models
            whisper_ft_path = os.path.join(cache_dir, "whisper-ft")
            if os.path.exists(whisper_ft_path):
                # Try loading safetensors first
                if os.path.exists(os.path.join(whisper_ft_path, "model.safetensors")):
                    logging.info("Loading fine-tuned Whisper model from safetensors...")
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(os.path.join(whisper_ft_path, "model.safetensors"), device="cpu")
                        base_model.load_state_dict(state_dict, strict=False)
                        logging.info("Fine-tuned Whisper model loaded successfully from safetensors!")
                    except Exception as e:
                        logging.error(f"Failed to load Whisper model from safetensors: {str(e)}")
                
                # Try loading PT file if safetensors failed or doesn't exist
                elif os.path.exists(os.path.join(whisper_ft_path, "model.pt")):
                    logging.info("Loading fine-tuned Whisper model from PT file...")
                    try:
                        checkpoint = torch.load(os.path.join(whisper_ft_path, "model.pt"), map_location="cpu")
                        base_model.load_state_dict(checkpoint, strict=False)
                        logging.info("Fine-tuned Whisper model loaded successfully from PT file!")
                    except Exception as e:
                        logging.error(f"Failed to load Whisper model from PT file: {str(e)}")
            
            self.whisper_model = base_model.to(self.device)
        except Exception as e:
            logging.error(f"Failed to initialize Whisper model: {str(e)}")
            # Fallback to smaller model if available
            try:
                logging.warning(f"Falling back to small.en Whisper model")
                self.whisper_model = whisper.load_model("small.en", download_root=whisper_path).to(self.device)
            except Exception as fallback_error:
                logging.error(f"Critical failure loading any Whisper model: {str(fallback_error)}")
                raise RuntimeError("Could not initialize Whisper model")

    def _load_resepformer_model(self, cache_dir):
        """Load the fine-tuned RESepFormer model if available, otherwise use the base model"""
        logging.info("Loading RESepFormer model...")
        resepformer_path = os.path.join(cache_dir, "resepformer")
        ft_model_path = os.path.join(cache_dir, "resepformer-ft")
        
        try:
            # First, load the base model to ensure a fallback is possible
            self.separator = SepformerSeparation.from_hparams(
                source="speechbrain/resepformer-wsj02mix",
                savedir=resepformer_path,
                run_opts={"device": self.device}
            )
            
            # If fine-tuned model exists, try to load it
            if os.path.exists(ft_model_path):
                logging.info("Found fine-tuned RESepFormer model. Attempting to load...")
                
                # Create a temporary working directory
                import tempfile
                import shutil
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Check required files
                    required_files = ['hyperparams.yaml', 'masknet.ckpt', 'encoder.ckpt', 'decoder.ckpt']
                    all_files_exist = all(os.path.exists(os.path.join(ft_model_path, f)) for f in required_files)
                    
                    if all_files_exist:
                        # Copy required files to temp dir
                        for f in required_files:
                            shutil.copy(os.path.join(ft_model_path, f), os.path.join(tmpdir, f))
                        
                        # Load state dict components with map_location to handle CUDA->CPU conversion
                        state_dict = {
                            'masknet': torch.load(os.path.join(ft_model_path, 'masknet.ckpt'), map_location=self.device),
                            'encoder': torch.load(os.path.join(ft_model_path, 'encoder.ckpt'), map_location=self.device),
                            'decoder': torch.load(os.path.join(ft_model_path, 'decoder.ckpt'), map_location=self.device)
                        }
                        
                        # Apply the component weights to the model
                        self.separator.load_state_dict(state_dict, strict=False)
                        logging.info("Fine-tuned RESepFormer model loaded successfully!")
                    else:
                        logging.warning(f"Missing required files for fine-tuned RESepFormer. Using base model instead.")
            else:
                logging.info("No fine-tuned RESepFormer model found. Using base model.")
        
        except Exception as e:
            logging.error(f"Failed to load RESepFormer model: {str(e)}")
            # Attempt fallback to base model if fine-tuned model hasn't loaded it yet
            try:
                logging.warning(f"Falling back to base RESepFormer model")
                self.separator = SepformerSeparation.from_hparams(
                    source="speechbrain/resepformer-wsj02mix",
                    savedir=resepformer_path,
                    run_opts={"device": self.device}
                )
            except Exception as fallback_error:
                logging.error(f"Critical failure loading any RESepFormer model: {str(fallback_error)}")
                raise RuntimeError("Could not initialize RESepFormer model")

    def _initialize_pyannote_models(self, cache_dir):
        """Initialize PyAnnote models for diarization, VAD, and speaker embedding"""
        try:
            logging.info("Initializing PyAnnote models...")
            
            # Set up cache directories
            diarization_cache = os.path.join(cache_dir, "speaker-diarization")
            vad_cache = os.path.join(cache_dir, "vad")
            embedding_cache = os.path.join(cache_dir, "embedding")
            
            self.embedding_model = Inference(
                "pyannote/embedding",
                window="whole",
                use_auth_token=self.config.auth_token,
            ).to(self.device)

            self.vad_pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=self.config.auth_token,
                cache_dir=vad_cache
            ).to(self.device)

            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.auth_token,
                cache_dir=diarization_cache
            ).to(self.device)
            
            logging.info("PyAnnote models initialized successfully!")
        except Exception as e:
            logging.error(f"Failed to initialize PyAnnote models: {str(e)}")
            raise RuntimeError("Could not initialize PyAnnote models")

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        logging.info(f"Loading audio from {file_path}")
        
        # Check if file is MP3 and convert to WAV first if needed
        if file_path.lower().endswith('.mp3'):
            logging.info(f"Converting MP3 to WAV before processing: {file_path}")
            temp_wav_path = file_path.replace('.mp3', '.wav')
            if not os.path.exists(temp_wav_path):
                try:
                    # Convert to WAV using ffmpeg
                    subprocess.check_call([
                        'ffmpeg', '-y', '-i', file_path, 
                        '-acodec', 'pcm_s16le', '-ar', str(self.config.target_sample_rate), 
                        temp_wav_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logging.info(f"Successfully converted MP3 to WAV: {temp_wav_path}")
                    file_path = temp_wav_path
                except Exception as e:
                    logging.error(f"Failed to convert MP3 to WAV: {str(e)}")
        
        # Continue with regular loading process
        signal, sample_rate = torchaudio.load(file_path)
        signal = signal.to(self.device)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if sample_rate != self.config.target_sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sample_rate:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.target_sample_rate
                ).to(self.device)
            signal = self.resampler(signal)
        signal_np = signal.cpu().squeeze().numpy()
        signal_np = nr.reduce_noise(y=signal_np, sr=self.config.target_sample_rate,
                                    stationary=True, prop_decrease=self.config.noise_reduction_amount)
        signal_np = signal_np / (np.max(np.abs(signal_np)) + 1e-8)
        signal = torch.tensor(signal_np, device=self.device).unsqueeze(0)
        duration = signal.shape[-1] / self.config.target_sample_rate
        logging.info(f"Audio loaded: {duration:.2f}s at {self.config.target_sample_rate}Hz")
        return signal, self.config.target_sample_rate

    def _extract_segment(self, audio: torch.Tensor, start: float, end: float, sample_rate: Optional[int] = None) -> torch.Tensor:
        sr = sample_rate or self.config.target_sample_rate
        audio_duration = audio.shape[-1] / sr
        logging.debug(f"Extracting segment: start={start:.2f}s, end={end:.2f}s, audio duration={audio_duration:.2f}s")
        
        if start < 0:
            logging.warning(f"Start time {start:.2f}s is negative; setting to 0.")
            start = 0.0
        if end > audio_duration:
            logging.warning(f"End time {end:.2f}s exceeds audio duration ({audio_duration:.2f}s); clipping to audio duration.")
            end = audio_duration

        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        if start_idx >= end_idx:
            logging.warning(f"Invalid segment indices: {start_idx}-{end_idx} for audio length {audio.shape[-1]} "
                            f"(start_time={start:.2f}s, end_time={end:.2f}s, sample_rate={sr}).")
            return torch.zeros((1, 100), device=self.device)
        
        return audio[:, start_idx:end_idx]

    def _extract_embedding(self, audio_segment: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if audio_segment.shape[-1] < self.config.target_sample_rate / 2:
                logging.warning("Segment too short for reliable embedding extraction")
                return None
            audio_np = (audio_segment.cpu().numpy() if len(audio_segment.shape) > 1
                        else audio_segment.cpu().unsqueeze(0).numpy())
            embedding = self.embedding_model({
                "waveform": torch.from_numpy(audio_np),
                "sample_rate": self.config.target_sample_rate
            })
            return embedding.to(self.device) if isinstance(embedding, torch.Tensor) else torch.tensor(embedding, device=self.device)
        except Exception as e:
            logging.error(f"Error in embedding extraction: {e}")
            return None

    def _calculate_embedding_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(embed1, embed2, dim=0).item()

    def _detect_overlap_regions(self, diarization_result) -> List[Tuple[float, float, List[str]]]:
        segments = [(segment.start, segment.end, speaker)
                    for segment, _, speaker in diarization_result.itertracks(yield_label=True)]
        overlap_dict = find_segment_overlaps(segments)
        overlap_regions = [
            (start, end, speakers)
            for (start, end), speakers in overlap_dict.items()
            if (end - start) >= self.config.overlap_threshold and len(speakers) > 1
        ]
        logging.info(f"Detected {len(overlap_regions)} overlap regions")
        return overlap_regions

    def _build_speaker_profiles(self, audio: torch.Tensor, diarization_result) -> Dict[str, torch.Tensor]:
        if not self.config.use_speaker_embeddings:
            return {}
        all_segments = [(segment.start, segment.end, speaker)
                        for segment, _, speaker in diarization_result.itertracks(yield_label=True)]
        speaker_segments = defaultdict(list)
        
        # Group segments by speaker
        for start, end, speaker in all_segments:
            if (end - start) >= 0.75:  # Lower minimum duration from 1.0 to 0.75
                speaker_segments[speaker].append((start, end, end-start))
        
        speaker_embeddings = {}
        for speaker, segments in speaker_segments.items():
            logging.info(f"Building profile for speaker {speaker} using {len(segments)} segments")
            
            # Sort by duration
            segments_by_duration = sorted(segments, key=lambda x: x[2], reverse=True)
            
            # Take top segments by duration (half of max_embedding_segments)
            top_segments = segments_by_duration[:self.config.max_embedding_segments // 2]
            
            # Sort the remaining segments by start time
            temporal_segments = sorted([s for s in segments if s not in top_segments], key=lambda x: x[0])
            
            # Take segments spaced throughout the recording (the other half)
            step = max(1, len(temporal_segments) // (self.config.max_embedding_segments // 2))
            diverse_segments = temporal_segments[::step][:self.config.max_embedding_segments // 2]
            
            # Combine the two sets of segments
            selected_segments = top_segments + diverse_segments
            
            # Extract embeddings from these segments
            embeddings = []
            embedding_quality = []
            for start, end, _ in selected_segments:
                segment_audio = self._extract_segment(audio, start, end)
                
                # Apply noise reduction for cleaner embedding
                if segment_audio.shape[-1] > self.config.target_sample_rate * 0.5:  # At least 0.5s
                    segment_audio_clean = enhance_audio(segment_audio, 
                                                    self.config.target_sample_rate,
                                                    prop_decrease=self.config.noise_reduction_amount)
                    embedding = self._extract_embedding(segment_audio_clean)
                    if embedding is not None:
                        # Estimate quality based on audio characteristics (e.g., signal variance)
                        signal_var = torch.var(segment_audio).item()
                        embeddings.append(embedding)
                        embedding_quality.append(signal_var)
            
            if embeddings:
                # If we have quality metrics, use weighted average based on quality
                if embedding_quality:
                    # Normalize quality scores
                    total_quality = sum(embedding_quality)
                    if total_quality > 0:
                        weights = [q / total_quality for q in embedding_quality]
                        weighted_sum = sum(e * w for e, w in zip(embeddings, weights))
                        speaker_embeddings[speaker] = weighted_sum
                    else:
                        speaker_embeddings[speaker] = torch.stack(embeddings).mean(dim=0)
                else:
                    speaker_embeddings[speaker] = torch.stack(embeddings).mean(dim=0)
                
                logging.info(f"Created embedding for speaker {speaker} from {len(embeddings)} segments")
            
        return speaker_embeddings

    def _resegment_overlap(self, audio_segment: torch.Tensor, seg_start: float, seg_end: float, speaker_profiles: Dict[str, torch.Tensor]) -> List[Tuple[float, float, str]]:
        window_size = self.config.sliding_window_size
        step = self.config.sliding_window_step
        
        # Use smaller steps for rapid exchanges
        segment_duration = seg_end - seg_start
        if segment_duration < 2.0:  # For short segments, use smaller step
            step = min(step, segment_duration / 4)
        
        window_results = []
        curr = seg_start
        prev_speaker = None
        
        while curr + window_size <= seg_end:
            segment = self._extract_segment(audio_segment, curr - seg_start, curr - seg_start + window_size)
            embedding = self._extract_embedding(segment)
            
            if embedding is not None:
                similarities = [(spk, self._calculate_embedding_similarity(embedding, profile))
                                for spk, profile in speaker_profiles.items()]
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Use more aggressive filtering for speaker changes
                top_speaker, top_confidence = similarities[0]
                
                # If we have multiple possible speakers, check if there's a clear winner
                if len(similarities) > 1:
                    second_speaker, second_confidence = similarities[1]
                    confidence_gap = top_confidence - second_confidence
                    
                    # If confidences are very close, this might be a transition point
                    if confidence_gap < 0.15 and prev_speaker and prev_speaker != top_speaker:
                        # Use previous speaker for continuity unless there's strong evidence
                        if second_speaker == prev_speaker and second_confidence > 0.65 * top_confidence:
                            top_speaker = prev_speaker
                            top_confidence = second_confidence
                
                dominant_speaker, confidence = top_speaker, top_confidence
                prev_speaker = dominant_speaker
            else:
                # If no embedding could be extracted, try to maintain continuity
                dominant_speaker = prev_speaker if prev_speaker else "UNKNOWN"
                confidence = 0.0
                
            window_results.append((curr, curr + window_size, dominant_speaker, confidence))
            curr += step
        
        if not window_results:
            return [(seg_start, seg_end, "UNKNOWN")]
        
        # Enhanced merging logic that's more sensitive to rapid exchanges
        merged = []
        cur_start, cur_end, cur_spk, cur_conf = window_results[0]
        
        for start, end, spk, conf in window_results[1:]:
            if spk == cur_spk and start - cur_end <= max(step * 1.5, 0.2):  # More permissive on gaps
                cur_end = end
                # Update confidence to the average
                cur_conf = (cur_conf + conf) / 2
            else:
                # Only include segment if it's long enough (avoid excessive fragmentation)
                if (cur_end - cur_start) >= min(0.3, segment_duration / 10):
                    merged.append((cur_start, cur_end, cur_spk))
                cur_start, cur_end, cur_spk, cur_conf = start, end, spk, conf
        
        # Add the last segment
        if (cur_end - cur_start) >= min(0.3, segment_duration / 10):
            merged.append((cur_start, cur_end, cur_spk))
        
        # Adjust boundaries to ensure no overlaps and correct time range
        final_segments = []
        for i, (start, end, spk) in enumerate(merged):
            adjusted_start = max(seg_start, start)
            adjusted_end = min(seg_end, end)
            
            # Ensure minimum segment duration where possible
            min_duration = min(0.3, segment_duration / 10)
            if adjusted_end - adjusted_start < min_duration and i > 0:
                # Try to extend by reducing previous segment
                prev_start, prev_end, prev_spk = final_segments[-1]
                if prev_end - prev_start > min_duration * 1.5:  # Only if previous segment is long enough
                    gap_to_fill = min_duration - (adjusted_end - adjusted_start)
                    prev_end -= min(gap_to_fill, prev_end - prev_start - min_duration)
                    adjusted_start = prev_end
                    final_segments[-1] = (prev_start, prev_end, prev_spk)
            
            if adjusted_end - adjusted_start >= min_duration:
                final_segments.append((adjusted_start, adjusted_end, spk))
        
        return [(max(seg_start, s), min(seg_end, e), spk) for s, e, spk in final_segments]
    
    def _diarize(self, file_path, min_speakers, max_speakers):
        """Run the speaker diarization process"""
        try:
            # Run the actual diarization
            result = self.diarization(
                file_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            return result
        except Exception as e:
            logging.error(f"Error in diarization: {str(e)}")
            raise

    def _process_overlap_segment(self, audio_segment: torch.Tensor, speaker_embeddings: Dict[str, torch.Tensor],
                               involved_speakers: List[str], seg_start: float, seg_end: float) -> List[Dict]:
        logging.info(f"Processing overlap segment: {seg_start:.2f}s-{seg_end:.2f}s")
        
        refined_regions = self._resegment_overlap(audio_segment, seg_start, seg_end, speaker_embeddings)
        
        results = []
        for new_start, new_end, spk in refined_regions:
            subsegment = self._extract_segment(audio_segment, new_start - seg_start, new_end - seg_start)
            
            try:
                separated = self.separator.separate_batch(subsegment)
                
                best_source, best_confidence = None, -1.0
                for idx in range(separated.shape[-1]):
                    source = separated[..., idx]
                    source = source / (torch.max(torch.abs(source)) + 1e-8)
                    embedding = self._extract_embedding(source)
                    if embedding is None:
                        continue
                    similarity = self._calculate_embedding_similarity(embedding, speaker_embeddings.get(spk, embedding))
                    if similarity > best_confidence:
                        best_confidence = similarity
                        best_source = source
                
                best_source = best_source if best_source is not None else subsegment
                source_np = best_source.squeeze().cpu().numpy()
                
                # Use transcription method
                transcription = self._transcribe(
                    source_np,
                    initial_prompt="This is a single speaker talking.",
                    temperature=self.config.temperature
                )
                
                results.append({
                    'audio': best_source,
                    'transcription': transcription['text'],
                    'speaker_id': spk,
                    'confidence': best_confidence
                })
            except Exception as e:
                logging.error(f"Error processing overlap subsegment: {e}")
                # Add a partial result with error information
                results.append({
                    'audio': subsegment,
                    'transcription': "[Processing error]",
                    'speaker_id': spk,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results

    def _secondary_diarization(self, audio_segment: torch.Tensor, seg_start: float, seg_end: float) -> List[Tuple[float, float, str]]:
        try:
            temp_path = "temp_segment.wav"
            torchaudio.save(temp_path, audio_segment.cpu(), self.config.target_sample_rate)
            diarization_result = self.diarization(
                temp_path,
                min_speakers=1,
                max_speakers=2
            )
            os.remove(temp_path)
            new_segments = [(segment.start, segment.end, speaker)
                            for segment, _, speaker in diarization_result.itertracks(yield_label=True)]
            if not new_segments:
                return [(seg_start, seg_end, "UNKNOWN")]
            return merge_diarization_segments(new_segments, self.config.merge_gap_threshold)
        except Exception as e:
            logging.error(f"Secondary diarization failed: {e}")
            return [(seg_start, seg_end, "UNKNOWN")]

    def save_segments(self, segments: List[AudioSegment], output_dir: str):
        output_dir = Path(output_dir)
        regular_dir = output_dir / "regular_segments"
        overlap_dir = output_dir / "overlap_segments"
        regular_dir.mkdir(parents=True, exist_ok=True)
        overlap_dir.mkdir(parents=True, exist_ok=True)
        
        for segment in segments:
            timestamp = f"{segment.start:.2f}-{segment.end:.2f}"
            if segment.is_overlap:
                filename = f"overlap_{timestamp}_{segment.speaker_id}.wav"
                save_path = overlap_dir / filename
            else:
                filename = f"{timestamp}_{segment.speaker_id}.wav"
                save_path = regular_dir / filename
            
            torchaudio.save(str(save_path), segment.audio_tensor.cpu(), self.config.target_sample_rate)
            logging.info(f"Saved segment: {save_path}")

    def save_debug_segments(self, segments: List[AudioSegment], output_dir: str):
        debug_dir = Path(output_dir) / "debug_segments"
        debug_dir.mkdir(parents=True, exist_ok=True)
        metadata = []
        
        for idx, segment in enumerate(segments):
            segment_id = f"segment_{idx:03d}"
            segment_type = "overlap" if segment.is_overlap else "regular"
            segment_dir = debug_dir / segment_type
            segment_dir.mkdir(exist_ok=True)
            
            audio_filename = f"{segment_id}.wav"
            torchaudio.save(str(segment_dir / audio_filename), segment.audio_tensor.cpu(), self.config.target_sample_rate)
            
            segment_metadata = {
                "segment_id": segment_id,
                "start_time": f"{segment.start:.3f}",
                "end_time": f"{segment.end:.3f}",
                "duration": f"{segment.end - segment.start:.3f}",
                "speaker_id": segment.speaker_id,
                "is_overlap": segment.is_overlap,
                "transcription": segment.transcription,
                "audio_file": str(segment_dir / audio_filename),
                "audio_stats": {
                    "max_amplitude": float(torch.max(torch.abs(segment.audio_tensor)).cpu()),
                    "mean_amplitude": float(torch.mean(torch.abs(segment.audio_tensor)).cpu()),
                    "samples": segment.audio_tensor.shape[-1]
                }
            }
            metadata.append(segment_metadata)
            
            with open(segment_dir / f"{segment_id}_info.txt", "w") as f:
                f.write(f"Segment ID: {segment_id}\n")
                f.write(f"Time: {segment.start:.3f}s - {segment.end:.3f}s\n")
                f.write(f"Speaker: {segment.speaker_id}\n")
                f.write(f"Overlap: {segment.is_overlap}\n")
                f.write(f"Transcription: {segment.transcription}\n")
        
        with open(debug_dir / "segments_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Debug segments saved to: {debug_dir}")
        logging.info(f"Total segments: {len(segments)}")
        logging.info(f"Overlap segments: {sum(1 for s in segments if s.is_overlap)}")
        logging.info(f"Regular segments: {sum(1 for s in segments if not s.is_overlap)}")

    def run(self, input_file, output_dir: str = "processed_audio", debug_mode: bool = False, 
            progress_callback=None):
        try:
            if progress_callback:
                progress_callback(5, "Starting processing")
            
            # Ensure models are loaded before processing
            if not self.models_are_loaded():
                if not self.load_models_with_progress(progress_callback):
                    return None, None, None
                    
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Processing file: {input_file}")
            
            if progress_callback:
                progress_callback(30, "Running file processing")
                    
            # Process the audio file
            results = self.process_file(input_file)
            if results is None:  # Processing failed for some reason
                return None, None, None
            
            if progress_callback:
                progress_callback(60, "Saving processed segments")
                
            # Check if we have valid segments with transcriptions
            if 'segments' not in results or not results['segments']:
                logging.error("No segments were generated during processing")
                return None, None, None
                
            # Check if any segments have transcriptions
            has_transcriptions = any(segment.transcription and segment.transcription.strip() 
                                    for segment in results['segments'])
                                    
            if not has_transcriptions:
                logging.error("No transcriptions were generated for any segments")
                return None, None, None
                    
            # Save the processed segments
            self.save_segments(results['segments'], output_dir)
            
            if debug_mode:
                self.save_debug_segments(results['segments'], output_dir)
                
            if progress_callback:
                progress_callback(80, "Saving transcript")
                
            # Generate and save transcript
            transcript_path = os.path.join(output_dir, "transcript.txt")
            transcript = ""
            for segment in results['segments']:
                transcript += f"[{segment.speaker_id}] {segment.start:.2f}s - {segment.end:.2f}s\n"
                transcript += f"{segment.transcription}\n\n"
                
            # Check if the transcript is not empty
            if not transcript.strip():
                logging.error("Generated transcript is empty")
                return None, None, None
                
            with open(transcript_path, "w", encoding='utf-8') as f:
                f.write(transcript)
                
            # Log the first few lines of transcript for debugging
            preview_lines = transcript.split('\n')[:6]
            logging.info(f"Transcript preview (first few lines):\n{''.join(preview_lines[:5])}")
            logging.info(f"Transcript saved to: {transcript_path}")
            logging.info(f"Transcript size: {os.path.getsize(transcript_path)} bytes")
            
            if progress_callback:
                progress_callback(100, "Processing completed")
                
            return input_file, transcript, transcript_path
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            traceback.print_exc()
            raise
    
    def _transcribe(self, audio_np, initial_prompt="", word_timestamps=False, 
                   condition_on_previous_text=True, temperature=0.0):
        """Simplified transcribe method"""
        try:
            result = self.whisper_model.transcribe(
                audio_np,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                condition_on_previous_text=condition_on_previous_text,
                temperature=temperature
            )
            return result
        except Exception as e:
            logging.error(f"Error in whisper transcription: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def process_file(self, file_path: str) -> Dict:
        try:
            # Convert MP3 to WAV if needed, for PyAnnote compatibility
            original_file_path = file_path
            file_path = ensure_wav_format(file_path)
                
            # Load the audio (this will use converted WAV if available)
            audio, sample_rate = self.load_audio(original_file_path)
            audio_duration = audio.shape[-1] / sample_rate
            logging.info(f"Processing audio file: {audio_duration:.2f} seconds")
                
            # Run Voice Activity Detection on the WAV file
            logging.info("Running Voice Activity Detection...")
            vad_result = self.vad_pipeline(file_path)
            vad_intervals = get_pyannote_vad_intervals(vad_result)
            logging.info(f"VAD detected {len(vad_intervals)} speech intervals")
            
            # Run Speaker Diarization on the WAV file
            logging.info("Running Speaker Diarization...")
            diarization_result = self._diarize(
                file_path,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
                
            # Process and merge segments
            raw_segments = [(segment.start, segment.end, speaker)
                            for segment, _, speaker in diarization_result.itertracks(yield_label=True)]
            logging.info(f"Diarization found {len(raw_segments)} raw segments")
                
            merged_segments = merge_diarization_segments(raw_segments, self.config.merge_gap_threshold)
            logging.info(f"After merging: {len(merged_segments)} segments")
                
            # Refine segments with VAD if enabled
            refined_segments = []
            if self.config.use_vad_refinement:
                for start, end, speaker in merged_segments:
                    refined = refine_segment_with_vad((start, end), vad_intervals)
                    if refined and (refined[1] - refined[0] >= self.config.min_segment_duration):
                        refined_segments.append((refined[0], refined[1], speaker))
                logging.info(f"After VAD refinement: {len(refined_segments)} segments")
            else:
                refined_segments = merged_segments
                
            # Build speaker profiles
            speaker_embeddings = self._build_speaker_profiles(audio, diarization_result)
            logging.info(f"Created embeddings for {len(speaker_embeddings)} speakers")
                
            # Map speaker IDs to friendly names
            speaker_counts = Counter(speaker for _, _, speaker in refined_segments)
            if len(speaker_counts) < 2:
                logging.warning("Not enough speakers detected, using default mapping")
                common_speakers = list(speaker_counts.keys())
                if len(common_speakers) == 0:
                    raise ValueError("No speakers detected in the audio file")
                speaker_mapping = {common_speakers[0]: "SPEAKER_A"}
            else:
                common_speakers = [spk for spk, _ in speaker_counts.most_common(2)]
                speaker_mapping = {common_speakers[0]: "SPEAKER_A", common_speakers[1]: "SPEAKER_B"}
            logging.info(f"Using speaker mapping: {speaker_mapping}")
            
            # Detect overlap regions
            overlap_regions = self._detect_overlap_regions(diarization_result)
                
            # Prepare for chronological processing
            # Sort segments by start time to ensure proper time ordering
            refined_segments.sort(key=lambda x: x[0])
            
            # Process each segment
            processed_segments = []
            meta_counts = {'SPEAKER_A': 0, 'SPEAKER_B': 0}
            
            # Variables to track context for better continuity
            previous_end = 0
            previous_speaker = None
            previous_transcript = ""
            
            # Process each segment
            segment_count = len(refined_segments)
            
            for i, (seg_start, seg_end, orig_speaker) in enumerate(refined_segments):
                # Skip segments that are too short
                if (seg_end - seg_start) < self.config.min_segment_duration:
                    continue
                
                # Check if segment contains overlap
                is_overlap = False
                involved_speakers = []
                
                for ov_start, ov_end, speakers in overlap_regions:
                    if max(seg_start, ov_start) < min(seg_end, ov_end):
                        is_overlap = True
                        involved_speakers = speakers
                        break
                
                # Extract the audio segment
                audio_segment = self._extract_segment(audio, seg_start, seg_end)
                spk_label = speaker_mapping.get(orig_speaker, "UNKNOWN")
                
                # Check for rapid exchange (current segment starts soon after previous)
                is_rapid_exchange = False
                if previous_speaker is not None and previous_speaker != orig_speaker:
                    # Consider it rapid exchange if gap between segments is small
                    if 0 < (seg_start - previous_end) < 0.5:  # 500ms threshold for rapid exchange
                        is_rapid_exchange = True
                        logging.info(f"Detected rapid exchange at {seg_start:.2f}s from {previous_speaker} to {spk_label}")
                
                # Process non-overlapping segments
                if not is_overlap:
                    embedding = self._extract_embedding(audio_segment)
                    
                    # Check embedding similarity and potentially re-analyze segment
                    if embedding is not None:
                        profile = speaker_embeddings.get(orig_speaker)
                        similarity = self._calculate_embedding_similarity(embedding, profile) if profile is not None else 0
                        
                        if similarity < self.config.secondary_diarization_threshold:
                            logging.info(f"Segment {seg_start:.2f}-{seg_end:.2f}s has low similarity ({similarity:.2f}); re-running secondary diarization.")
                            new_segments = self._secondary_diarization(audio_segment, seg_start, seg_end)
                            
                            for new_start, new_end, new_spk in new_segments:
                                sub_audio = self._extract_segment(audio_segment, new_start - seg_start, new_end - seg_start)
                                
                                # Adjust context for transcription continuity
                                initial_prompt = "This is a clear conversation with complete sentences."
                                
                                # Use previous transcript as context if it's the same speaker
                                if new_spk == previous_speaker and seg_start - previous_end < 1.0:
                                    initial_prompt = f"{previous_transcript.strip()} "
                                
                                # For rapid exchanges, use more context
                                if is_rapid_exchange:
                                    initial_prompt = "This is a fast-paced conversation with quick speaker changes. "
                                    
                                transcription = self.whisper_model.transcribe(
                                    sub_audio.squeeze().cpu().numpy(),
                                    initial_prompt=initial_prompt,
                                    word_timestamps=True,
                                    condition_on_previous_text=self.config.condition_on_previous_text,
                                    temperature=self.config.temperature
                                )
                                
                                final_label = speaker_mapping.get(new_spk, spk_label)
                                segment = AudioSegment(
                                    start=seg_start + new_start,
                                    end=seg_start + new_end,
                                    speaker_id=final_label,
                                    audio_tensor=sub_audio,
                                    is_overlap=False,
                                    transcription=transcription['text'],
                                    confidence=1.0,
                                    metadata={'rapid_exchange': is_rapid_exchange}
                                )
                                
                                processed_segments.append(segment)
                                meta_counts[final_label] = meta_counts.get(final_label, 0) + 1
                                
                                # Update context for next segment
                                previous_end = seg_start + new_end
                                previous_speaker = new_spk
                                previous_transcript = transcription['text']
                                
                            continue
                    
                    # Standard segment processing for non-overlap segments
                    # Choose appropriate initial prompt based on context
                    initial_prompt = "This is a conversation between two people."
                    
                    # Use previous transcript as context if same speaker and close in time
                    if orig_speaker == previous_speaker and seg_start - previous_end < 1.0:
                        initial_prompt = f"{previous_transcript.strip()} "
                    
                    # For rapid exchanges, use a different prompt
                    if is_rapid_exchange:
                        initial_prompt = "This is a fast-paced conversation with quick speaker changes. "
                    
                    transcription = self.whisper_model.transcribe(
                        audio_segment.squeeze().cpu().numpy(),
                        initial_prompt=initial_prompt,
                        word_timestamps=True,
                        condition_on_previous_text=self.config.condition_on_previous_text,
                        temperature=self.config.temperature
                    )
                    
                    segment = AudioSegment(
                        start=seg_start,
                        end=seg_end,
                        speaker_id=spk_label,
                        audio_tensor=audio_segment,
                        is_overlap=False,
                        transcription=transcription['text'],
                        confidence=1.0,
                        metadata={'rapid_exchange': is_rapid_exchange}
                    )
                    
                    processed_segments.append(segment)
                    meta_counts[spk_label] = meta_counts.get(spk_label, 0) + 1
                    
                    # Update context for next segment
                    previous_end = seg_end
                    previous_speaker = orig_speaker
                    previous_transcript = transcription['text']
                    
                # Process overlapping segments with special handling
                else:
                    # Reset context after overlap since it's a disruption
                    previous_speaker = None
                    previous_transcript = ""
                    
                    mapped_profiles = {speaker_mapping.get(k, k): v for k, v in speaker_embeddings.items()}
                    refined_results = self._process_overlap_segment(
                        audio_segment,
                        speaker_embeddings=mapped_profiles,
                        involved_speakers=[speaker_mapping.get(s, s) for s in involved_speakers],
                        seg_start=seg_start,
                        seg_end=seg_end
                    )
                    
                    for result in refined_results:
                        final_label = result['speaker_id']
                        processed_segments.append(AudioSegment(
                            start=seg_start,
                            end=seg_end,
                            speaker_id=final_label,
                            audio_tensor=result['audio'],
                            is_overlap=True,
                            transcription=result['transcription'],
                            confidence=result.get('confidence', 0.5),
                            metadata={'overlap_speakers': involved_speakers}
                        ))
                        
                    # Update previous_end regardless
                    previous_end = seg_end
                    
            # Sort segments by start time
            processed_segments.sort(key=lambda x: x.start)
            
            # Build metadata for the result
            metadata = {
                'duration': audio_duration,
                'speaker_a_segments': meta_counts.get('SPEAKER_A', 0),
                'speaker_b_segments': meta_counts.get('SPEAKER_B', 0),
                'total_segments': len(processed_segments),
                'speakers': list(speaker_mapping.values()),
                'rapid_exchanges': sum(1 for s in processed_segments if s.metadata.get('rapid_exchange', False))
            }
                
            return {'segments': processed_segments, 'metadata': metadata}
        except Exception as e:
            logging.error(f"Error in process_file: {e}")
            traceback.print_exc()
            return None

# =============================================================================
# FastAPI App Configuration
# =============================================================================

app = FastAPI(
    title="ClearConverse API",
    description="Transcription solution mainly powered by Whisper-RESepFormer solution. Supported by PyAnnote's Speaker Diarization, Voice Activity Detection, and Embeddings"
)

# Define a cleanup task that runs in the background
async def cleanup_old_files(max_age_hours=1):
    """
    Periodically clean up old temporary files and processed audio directories
    that are older than the specified age in hours.
    """
    while True:
        try:
            logging.info(f"Running scheduled cleanup of files older than {max_age_hours} hours")
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            files_removed = 0
            dirs_removed = 0
            
            # Clean temporary upload files
            for file_path in Path("temp_uploads").glob("*"):
                if file_path.is_file():
                    # Get the modification time of the file
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time < cutoff_time:
                        try:
                            task_id = file_path.stem.split(".")[0]  # Extract task ID from filename
                            logging.info(f"Removing old temp file for task {task_id}")
                            file_path.unlink()
                            files_removed += 1
                            
                            # Also clean up any task data in memory
                            if task_id in progress_store:
                                del progress_store[task_id]
                            if task_id in result_store:
                                del result_store[task_id]
                            if task_id in active_processes:
                                del active_processes[task_id]
                            if task_id in uploaded_files:
                                del uploaded_files[task_id]
                        except Exception as e:
                            logging.error(f"Failed to remove temp file {file_path}: {e}")
            
            # Clean processed audio directories
            for dir_path in Path(OUTPUT_DIR).glob("*"):
                if dir_path.is_dir():
                    try:
                        # Try to get task ID from directory name
                        task_id = dir_path.name
                        
                        # Use the most recent file in the directory to determine age
                        most_recent = max(
                            [f.stat().st_mtime for f in dir_path.glob("**/*") if f.is_file()], 
                            default=dir_path.stat().st_mtime
                        )
                        mod_time = datetime.fromtimestamp(most_recent)
                        
                        if mod_time < cutoff_time:
                            logging.info(f"Removing old processed directory for task {task_id}")
                            shutil.rmtree(dir_path)
                            dirs_removed += 1
                            
                            # Also clean up any task data in memory
                            if task_id in progress_store:
                                del progress_store[task_id]
                            if task_id in result_store:
                                del result_store[task_id]
                            if task_id in active_processes:
                                del active_processes[task_id]
                            if task_id in uploaded_files:
                                del uploaded_files[task_id]
                    except Exception as e:
                        logging.error(f"Failed to remove directory {dir_path}: {e}")
            
            logging.info(f"Cleanup completed: removed {files_removed} files and {dirs_removed} directories")
            
            # Wait for the next cleanup interval (1 hour)
            await asyncio.sleep(60 * 60)  # Sleep for 1 hour
            
        except Exception as e:
            logging.error(f"Error in cleanup task: {e}")
            # If there's an error, wait a bit and try again
            await asyncio.sleep(60)  # Sleep for 1 minute before retrying

# Add this function to start the background task
def start_cleanup_task(app):
    """Start the background cleanup task when the app starts"""
    @app.on_event("startup")
    async def start_scheduler():
        # Start the cleanup task in the background
        asyncio.create_task(cleanup_old_files())
        logging.info("Started scheduled cleanup task (running every hour)")

start_cleanup_task(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=env_config['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global directory setup
OUTPUT_DIR = "processed_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
temp_uploads = Path("temp_uploads")
temp_uploads.mkdir(exist_ok=True)

load_dotenv()  # this loads variables from .env into the environment
AUTH_TOKEN = env_config['hf_auth_token']
config = Config(auth_token=AUTH_TOKEN)
processor = None

def get_processor(load_models=False):
    """Get or create the processor instance (lazy loading)"""
    global processor
    if processor is None:
        AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN', '')
        config = Config(auth_token=AUTH_TOKEN)
        processor = EnhancedAudioProcessor(config, load_models_immediately=load_models)
    return processor


# =============================================================================
# Helper Functions for Endpoints
# =============================================================================

def update_progress(task_id: str, percent: int, message: str):
    """Update and log progress for a given task"""
    progress_store[task_id] = {"progress": percent, "message": message}
    logging.info(f"Task {task_id}: {percent}% - {message}")

def run_transcription_process(task_id, file_path, output_dir):
    """Function to run in a separate process for transcription"""
    # Set up logging for this process
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Starting transcription process for task {task_id}")
    
    task_output_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Check if already completed to prevent duplicate processing
    completed_marker = os.path.join(task_output_dir, "completed.txt")
    if os.path.exists(completed_marker):
        logging.info(f"Task {task_id} already completed, skipping")
        return
    
    # Function to write progress to file that the main process can read
    def progress_callback(percent, message):
        progress_file = os.path.join(task_output_dir, "progress.json")
        progress_data = {"progress": percent, "message": message}
        with open(progress_file, "w") as f:
            json.dump(progress_data, f)
        logging.info(f"Task {task_id}: {percent}% - {message}")
    
    try:
        # Initialize processor with lazy loading
        from dotenv import load_dotenv
        load_dotenv()
        AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN', '')
        config = Config(auth_token=AUTH_TOKEN)
        processor = EnhancedAudioProcessor(config, load_models_immediately=False)
        
        # First update - models will start loading during run() call
        progress_callback(5, "Starting model initialization...")
        
        # Run the processing with progress callback
        input_file, transcript, transcript_path = processor.run(
            file_path, 
            output_dir=task_output_dir,
            debug_mode=False,
            progress_callback=progress_callback
        )
        
        # Delete the in_progress marker
        in_progress_marker = os.path.join(task_output_dir, "in_progress.txt")
        if os.path.exists(in_progress_marker):
            os.remove(in_progress_marker)
        
        # Final progress update
        progress_callback(100, "Transcription complete")
        
        # Write a completion flag
        with open(completed_marker, "w") as f:
            f.write(f"Transcription completed at {datetime.now().isoformat()}")
    except Exception as e:
        logging.error(f"Error in transcription process: {e}")
        # Write an error file
        with open(os.path.join(task_output_dir, "error.txt"), "w") as f:
            f.write(f"Error: {str(e)}")
        
        # Delete the in_progress marker
        in_progress_marker = os.path.join(task_output_dir, "in_progress.txt")
        if os.path.exists(in_progress_marker):
            os.remove(in_progress_marker)
        
        # Write final error progress
        progress_callback(100, f"Error: {str(e)}")

# =============================================================================
# API Endpoints
# =============================================================================

# Add this constant at the top of the file, after the imports
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload an audio file for processing"""
    if not file.filename.endswith((".mp3", ".wav",)):
        raise HTTPException(status_code=400, detail="Invalid file type provided.")
    
    # Check file size before processing
    # Read the content into memory to check size
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413, 
            detail=f"File size exceeds the maximum limit of 10MB. Your file is {file_size / (1024 * 1024):.2f}MB."
        )
    
    # Generate a task ID and save the file
    task_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    filename = f"{task_id}{extension}"
    file_path = temp_uploads / filename
    
    # Store original filename mapped to task_id
    original_filenames[task_id] = file.filename
    
    with open(file_path, "wb") as f:
        f.write(content)  # Use the content we already read
    
    update_progress(task_id, 0, "File uploaded")
    
    # Convert to WAV if it's an MP3
    if str(file_path).lower().endswith('.mp3'):
        update_progress(task_id, 5, "Converting MP3 to WAV")
        wav_path = ensure_wav_format(str(file_path))
        uploaded_files[task_id] = wav_path
        update_progress(task_id, 10, "Conversion complete")
    else:
        uploaded_files[task_id] = str(file_path)
        update_progress(task_id, 10, "File ready for processing")
    
    return JSONResponse(content={"task_id": task_id, "preview_url": f"/preview/{filename}"})

@app.post("/upload-url")
async def upload_url(url: str = Form(...)):
    """Endpoint to process an audio file from a URL"""
    validate_url(url)
    
    # Generate task ID and prepare for download
    task_id = str(uuid.uuid4())
    parsed_url = urlparse(url)
    extension = os.path.splitext(parsed_url.path)[1].lower()
    
    if extension.lower() not in ['.mp3', '.wav', '.ogg', '.mp4', '.flac', '.m4a', '.aac']:
        extension = ".mp3"  # Default to mp3 if extension is unknown
        
    filename = f"{task_id}{extension}"
    file_path = temp_uploads / filename
    
    # Extract original filename from URL or use a default name
    original_filename = os.path.basename(parsed_url.path)
    if not original_filename or original_filename == '':
        original_filename = f"recording{extension}"
    
    # Store original filename or a derived name from URL
    original_filenames[task_id] = original_filename
    
    # Update progress to indicate download starting
    update_progress(task_id, 5, "Starting download from URL")
    
    try:
        # Handle Google Drive URLs specifically
        if 'drive.google.com' in url:
            logging.info(f"Detected Google Drive URL: {url}")
            
            # Extract file ID from URL
            file_id = None
            file_match = re.search(r'/file/d/([^/]+)', url)
            if file_match:
                file_id = file_match.group(1)
            else:
                open_match = re.search(r'[?&]id=([^&]+)', url)
                if open_match:
                    file_id = open_match.group(1)
            
            if not file_id:
                raise HTTPException(status_code=400, detail="Invalid Google Drive URL format. Could not extract file ID.")
            
            logging.info(f"Extracted Google Drive file ID: {file_id}")
            update_progress(task_id, 10, "Downloading from Google Drive")
            
            # For Google Drive, we'll check size during download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
            temp_file.close()
            
            # First, make a HEAD request to check content length if possible
            session = requests.Session()
            try:
                head_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                head_response = session.head(head_url, timeout=10)
                content_length = head_response.headers.get('Content-Length')
                
                if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                    size_mb = int(content_length) / (1024 * 1024)
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File size exceeds the maximum limit of 10MB. File size: {size_mb:.2f}MB"
                    )
            except requests.exceptions.RequestException:
                # If HEAD request fails, we'll check during download
                pass
            
            # Use the special Google Drive download function with size checking
            try:
                download_size = 0
                URL = "https://drive.google.com/uc?export=download"
                
                # Get token if needed
                session = requests.Session()
                response = session.get(URL, params={'id': file_id}, stream=True)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break
                
                # Download with token if needed
                params = {'id': file_id, 'confirm': token} if token else {'id': file_id}
                response = session.get(URL, params=params, stream=True)
                
                with open(temp_file.name, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            download_size += len(chunk)
                            if download_size > MAX_FILE_SIZE_BYTES:
                                # Close and delete the file
                                f.close()
                                os.unlink(temp_file.name)
                                size_mb = download_size / (1024 * 1024)
                                raise HTTPException(
                                    status_code=413, 
                                    detail=f"File size exceeds the maximum limit of 10MB. File size: {size_mb:.2f}MB"
                                )
                            f.write(chunk)
                
                # Move temp file to final location
                shutil.move(temp_file.name, file_path)
                logging.info(f"Successfully downloaded Google Drive file {file_id} to {file_path}")
                update_progress(task_id, 25, "Download complete")
            
            except HTTPException:
                # Re-raise HTTPExceptions
                raise
            except Exception as e:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                logging.error(f"Error downloading from Google Drive: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
        else:
            # For non-Google Drive URLs, use the regular download function with size checking
            update_progress(task_id, 5, "Downloading audio from URL")
            
            # First, make a HEAD request to check content length if possible
            try:
                head_response = requests.head(url, timeout=10, 
                                            headers={'User-Agent': 'Mozilla/5.0'})
                content_length = head_response.headers.get('Content-Length')
                
                if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                    size_mb = int(content_length) / (1024 * 1024)
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File size exceeds the maximum limit of 10MB. File size: {size_mb:.2f}MB"
                    )
            except requests.exceptions.RequestException:
                # If HEAD request fails, we'll check during download
                pass
            
            # Download with size checking
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
            temp_file.close()
            
            try:
                download_size = 0
                with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, 
                                stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(temp_file.name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                download_size += len(chunk)
                                if download_size > MAX_FILE_SIZE_BYTES:
                                    # Close and delete the file
                                    f.close()
                                    os.unlink(temp_file.name)
                                    size_mb = download_size / (1024 * 1024)
                                    raise HTTPException(
                                        status_code=413, 
                                        detail=f"File size exceeds the maximum limit of 10MB. File size: {size_mb:.2f}MB"
                                    )
                                f.write(chunk)
                
                # Move temp file to final location
                shutil.move(temp_file.name, file_path)
                update_progress(task_id, 25, "Download complete")
            
            except HTTPException:
                # Re-raise HTTPExceptions
                raise
            except Exception as e:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                logging.error(f"Error downloading from URL: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    
        # Convert to WAV if it's an MP3
        if str(file_path).lower().endswith('.mp3'):
            update_progress(task_id, 5, "Converting MP3 to WAV")
            wav_path = ensure_wav_format(str(file_path))
            uploaded_files[task_id] = wav_path
            update_progress(task_id, 10, "Conversion complete")
        else:
            uploaded_files[task_id] = str(file_path)
            
    except Exception as e:
        logging.error(f"Error downloading file from URL {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    
    return JSONResponse(content={
        "task_id": task_id,
        "preview_url": f"/preview/{filename}" 
    })

# Update the transcription endpoint
@app.post("/transcribe/{task_id}")
async def transcribe_task(task_id: str):
    """Start transcription process for a previously uploaded file"""
    if task_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Task ID not found. Please upload a file or URL first.")
        
    file_path = uploaded_files[task_id]
    
    # Check if this task is already processing or completed
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    completed_marker = os.path.join(task_dir, "completed.txt")
    in_progress_marker = os.path.join(task_dir, "in_progress.txt") 
    
    # If already completed, return immediately
    if os.path.exists(completed_marker):
        return JSONResponse(content={"task_id": task_id, "status": "already_completed"})
    
    # If already in progress, return immediately 
    if os.path.exists(in_progress_marker):
        return JSONResponse(content={"task_id": task_id, "status": "already_in_progress"})
    
    # Mark as in progress first
    os.makedirs(task_dir, exist_ok=True)
    with open(in_progress_marker, "w") as f:
        f.write(f"Started at {datetime.now().isoformat()}")
    
    update_progress(task_id, 0, "Task queued for transcription")
    
    # Cancel any existing process with the same task ID
    if task_id in active_processes:
        try:
            old_process_info = active_processes[task_id]
            parent = psutil.Process(old_process_info['pid'])
            
            # Kill child processes first
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except:
                    pass
                    
            # Kill parent
            parent.kill()
            logging.info(f"Terminated previous process for task {task_id}")
        except Exception as e:
            logging.error(f"Error terminating previous process: {e}")
    
    # Launch transcription as a separate process
    process = multiprocessing.Process(
        target=run_transcription_process,
        args=(task_id, file_path, OUTPUT_DIR)
    )
    process.start()
    
    # Store the process info for later cancellation
    active_processes[task_id] = {
        'process': process,
        'pid': process.pid,
        'start_time': time.time(),
        'file_path': file_path
    }
    
    update_progress(task_id, 5, "Starting transcription process")
    logging.info(f"Started transcription process for task {task_id} with PID {process.pid}")
    
    return JSONResponse(content={"task_id": task_id})

@app.get("/preview/{filename}")
async def preview_audio(filename: str):
    """Serve uploaded audio file for preview"""
    file_path = temp_uploads / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(str(file_path), media_type="audio/mpeg", filename=filename)

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Forcibly cancel a transcription process while preserving the uploaded file"""
    update_progress(task_id, 99, "Cancelling transcription...")
    
    if task_id in active_processes:
        try:
            # Get the process info
            process_info = active_processes[task_id]
            
            try:
                # First try to kill using psutil for better child process cleanup
                parent = psutil.Process(process_info['pid'])
                
                # Kill child processes first (recursive)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except:
                        pass
                
                # Then kill the parent
                parent.kill()
                logging.info(f"Killed process tree for task {task_id}")
            except:
                # Fallback to basic termination
                process = process_info['process']
                if process.is_alive():
                    process.terminate()
                    logging.info(f"Terminated process for task {task_id}")
            
            # Remove from active processes
            del active_processes[task_id]
            
            # Mark as cancelled in progress and result store
            update_progress(task_id, 100, "Transcription cancelled")
            result_store[task_id] = {"status": "cancelled", "message": "Transcription was cancelled"}
            
            # IMPORTANT: Only delete the processed_audio directory, keep the uploaded file!
            task_output_dir = os.path.join(OUTPUT_DIR, task_id)
            if os.path.exists(task_output_dir):
                try:
                    shutil.rmtree(task_output_dir)
                    logging.info(f"Removed processed audio directory for task {task_id}")
                except Exception as e:
                    logging.error(f"Failed to remove processed audio directory: {e}")
            
            # Create a cancelled marker file in the processed_audio directory
            os.makedirs(task_output_dir, exist_ok=True)
            with open(os.path.join(task_output_dir, "cancelled.txt"), "w") as f:
                f.write("Transcription was cancelled")
                
            # Keep the file reference in uploaded_files to allow re-transcription
            logging.info(f"Preserved uploaded file reference for task {task_id}")
            
            return JSONResponse(content={"status": "cancelled", "message": "Transcription cancelled successfully"})
        except Exception as e:
            logging.error(f"Error cancelling process: {e}")
            # Still report cancellation to the user
            update_progress(task_id, 100, "Transcription cancelled")
            result_store[task_id] = {"status": "cancelled", "message": "Transcription was cancelled"}
            return JSONResponse(content={"status": "cancelled", "error": str(e)})
    
    # Task not found in active processes, still acknowledge cancellation
    update_progress(task_id, 100, "Transcription cancelled")
    result_store[task_id] = {"status": "cancelled", "message": "Transcription was cancelled"}
    return JSONResponse(content={"status": "cancelled"})

@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Check if a task has completed in the separate process"""
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    
    # Check if task directory exists
    if not os.path.exists(task_dir):
        return JSONResponse(content={"status": "not_found"})
    
    # Check for cancellation file
    if os.path.exists(os.path.join(task_dir, "cancelled.txt")):
        return JSONResponse(content={
            "status": "cancelled",
            "message": "Transcription was cancelled"
        })
    
    # Check for completion file
    if os.path.exists(os.path.join(task_dir, "completed.txt")):
        # Check for transcript file
        transcript_path = os.path.join(task_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            return JSONResponse(content={
                "status": "completed",
                "download_url": f"/download/{task_id}/transcript.txt"
            })
    
    # Check for error file
    if os.path.exists(os.path.join(task_dir, "error.txt")):
        try:
            with open(os.path.join(task_dir, "error.txt"), "r") as f:
                error_message = f.read()
            return JSONResponse(content={"status": "error", "message": error_message})
        except:
            return JSONResponse(content={"status": "error", "message": "Unknown error occurred"})
    
    # Check for progress information
    progress_file = os.path.join(task_dir, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
            
            # Also update the in-memory progress store
            if task_id not in progress_store or progress_store[task_id] != progress_data:
                progress_store[task_id] = progress_data
                
            return JSONResponse(content=progress_data)
        except:
            pass
    
    # Check if process is still running
    if task_id in active_processes:
        process_info = active_processes[task_id]
        elapsed_time = time.time() - process_info['start_time']
        
        # Use in-memory progress if available
        if task_id in progress_store:
            return JSONResponse(content={
                **progress_store[task_id],
                "elapsed_seconds": elapsed_time
            })
        
        # Default progress
        return JSONResponse(content={
            "status": "processing",
            "progress": 5,
            "message": "Processing in progress...",
            "elapsed_seconds": elapsed_time
        })
    
    # Default to in-memory progress
    if task_id in progress_store:
        return JSONResponse(content=progress_store[task_id])
    
    # Default response if nothing else applies
    return JSONResponse(content={"status": "unknown", "progress": 0, "message": "Unknown status"})

@app.get("/download-pdf/{task_id}")
async def download_pdf_transcript(task_id: str):
    """Generate and download the transcript as a PDF file"""
    transcript_path = Path(OUTPUT_DIR) / task_id / "transcript.txt"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")
        
    try:
        # Read the text transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
            
        # Define the PDF output path
        pdf_path = Path(OUTPUT_DIR) / task_id / "transcript.pdf"
        
        # Get original filename if available
        original_filename = original_filenames.get(task_id, None)
        
        # Generate the PDF with filename
        generate_transcript_pdf(transcript_text, str(pdf_path), original_filename)
        
        # Serve the PDF file
        return FileResponse(
            path=str(pdf_path), 
            media_type="application/pdf",
            filename="transcript.pdf"
        )
    except Exception as e:
        logging.error(f"Error generating PDF for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/transcription/{task_id}")
async def get_transcription(task_id: str):
    """Get the transcription text for a completed task, with special handling for cancelled tasks"""
    transcript_file = Path(OUTPUT_DIR) / task_id / "transcript.txt"
    cancelled_file = Path(OUTPUT_DIR) / task_id / "cancelled.txt"
    
    logging.info(f"Transcription requested for task {task_id}")
    
    # First check if the task was cancelled
    if cancelled_file.exists():
        logging.info(f"Task {task_id} was cancelled, returning appropriate response")
        return JSONResponse(status_code=202, content={
            "status": "cancelled",
            "message": "Transcription was cancelled by the user"
        })
    
    # Then check if the transcript file exists
    if not transcript_file.exists():
        logging.error(f"Transcript file not found for task {task_id}: {transcript_file}")
        
        # Check if the task was processed but the file might have been cleaned up
        if task_id in result_store:
            # If we have a result but no file, try to provide a helpful error
            logging.error(f"Task {task_id} is in result_store but file not found")
            return JSONResponse(status_code=404, content={
                "error": "Transcript file not found",
                "detail": "The transcript file may have been deleted or the task was cancelled"
            })
        else:
            # General case when no data exists for this task
            logging.error(f"Task {task_id} not found in result_store")
            return JSONResponse(status_code=404, content={
                "error": "Transcription not found",
                "detail": "No transcription data found for this task ID"
            })
    
    # File exists, check if it has content
    file_size = transcript_file.stat().st_size
    logging.info(f"Found transcript file for task {task_id}, size: {file_size} bytes")
    
    if file_size == 0:
        logging.error(f"Transcript file for task {task_id} is empty")
        return JSONResponse(status_code=400, content={
            "error": "Empty transcript file",
            "detail": "The transcript file exists but contains no data"
        })
    
    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
        
        if not transcript.strip():
            logging.error(f"Transcript file for task {task_id} contains only whitespace")
            return JSONResponse(status_code=400, content={
                "error": "Empty transcript content",
                "detail": "The transcript file contains only whitespace"
            })
        
        # Log a preview of the transcript
        preview = transcript[:100] + "..." if len(transcript) > 100 else transcript
        logging.info(f"Returning transcript for task {task_id}: {preview}")
        
        return JSONResponse(content={"task_id": task_id, "transcription": transcript})
    except Exception as e:
        logging.error(f"Error reading transcript file {transcript_file}: {e}")
        return JSONResponse(status_code=500, content={
            "error": "Failed to read transcript",
            "detail": str(e)
        })

@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Check if a task has completed in the separate process, with improved handling for cancelled tasks"""
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    
    # Check if task directory exists
    if not os.path.exists(task_dir):
        # Even if the directory doesn't exist, the task might still be valid if it's in uploaded_files
        if task_id in uploaded_files:
            return JSONResponse(content={
                "status": "uploaded",
                "message": "Audio file uploaded but processing not started"
            })
        return JSONResponse(content={"status": "not_found"})
    
    # Check for cancellation file
    if os.path.exists(os.path.join(task_dir, "cancelled.txt")):
        # For cancelled tasks, also include the original audio preview URL if available
        if task_id in uploaded_files:
            file_ext = os.path.splitext(uploaded_files[task_id])[1]
            preview_filename = f"{task_id}{file_ext}"
            preview_url = f"/preview/{preview_filename}"
            
            return JSONResponse(content={
                "status": "cancelled",
                "message": "Transcription was cancelled",
                "preview_url": preview_url
            })
        return JSONResponse(content={
            "status": "cancelled",
            "message": "Transcription was cancelled"
        })
    
    # Check for completion file
    if os.path.exists(os.path.join(task_dir, "completed.txt")):
        # Check for transcript file
        transcript_path = os.path.join(task_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            return JSONResponse(content={
                "status": "completed",
                "download_url": f"/download/{task_id}/transcript.txt"
            })
    
    # Check for error file
    if os.path.exists(os.path.join(task_dir, "error.txt")):
        try:
            with open(os.path.join(task_dir, "error.txt"), "r") as f:
                error_message = f.read()
            return JSONResponse(content={"status": "error", "message": error_message})
        except:
            return JSONResponse(content={"status": "error", "message": "Unknown error occurred"})
    
    # Check for progress information
    progress_file = os.path.join(task_dir, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
            
            # Also update the in-memory progress store
            if task_id not in progress_store or progress_store[task_id] != progress_data:
                progress_store[task_id] = progress_data
                
            return JSONResponse(content=progress_data)
        except:
            pass
    
    # Check if process is still running
    if task_id in active_processes:
        process_info = active_processes[task_id]
        elapsed_time = time.time() - process_info['start_time']
        
        # Use in-memory progress if available
        if task_id in progress_store:
            return JSONResponse(content={
                **progress_store[task_id],
                "elapsed_seconds": elapsed_time
            })
        
        # Default progress
        return JSONResponse(content={
            "status": "processing",
            "progress": 5,
            "message": "Processing in progress...",
            "elapsed_seconds": elapsed_time
        })
    
    # Check for uploaded file status
    if task_id in uploaded_files:
        # If we have an uploaded file but no processing info, we're ready to transcribe
        file_ext = os.path.splitext(uploaded_files[task_id])[1]
        preview_filename = f"{task_id}{file_ext}"
        preview_url = f"/preview/{preview_filename}"
        
        return JSONResponse(content={
            "status": "uploaded",
            "message": "Audio file uploaded but processing not started",
            "preview_url": preview_url
        })
    
    # Default to in-memory progress
    if task_id in progress_store:
        return JSONResponse(content=progress_store[task_id])
    
    # Default response if nothing else applies
    return JSONResponse(content={"status": "unknown", "progress": 0, "message": "Unknown status"})

async def clean_up_task_files(task_id: str):
    """Clean up files associated with a cancelled task"""
    try:
        await asyncio.sleep(0.5)  # Small delay to ensure cancellation response is sent
        
        # Don't actually delete output files immediately
        # Instead, just create a cancelled.txt marker
        task_dir = os.path.join(OUTPUT_DIR, task_id)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)
            
        # Mark as cancelled by creating a file
        with open(os.path.join(task_dir, "cancelled.txt"), "w") as f:
            f.write(f"Task cancelled at {datetime.now().isoformat()}")
            
        logging.info(f"Created cancellation marker for task {task_id}")
    except Exception as e:
        logging.error(f"Error in clean_up_task_files for {task_id}: {e}")

@app.get("/transcription/{task_id}")
async def get_transcription(task_id: str):
    """Get the transcription text for a completed task"""
    transcript_file = Path(OUTPUT_DIR) / task_id / "transcript.txt"
    
    logging.info(f"Transcription requested for task {task_id}")
    
    if not transcript_file.exists():
        logging.error(f"Transcript file not found for task {task_id}: {transcript_file}")
        
        # Check if the task was processed but the file might have been cleaned up
        if task_id in result_store:
            # If we have a result but no file, try to provide a helpful error
            logging.error(f"Task {task_id} is in result_store but file not found")
            return JSONResponse(status_code=404, content={
                "error": "Transcript file not found",
                "detail": "The transcript file may have been deleted or the task was cancelled"
            })
        else:
            # General case when no data exists for this task
            logging.error(f"Task {task_id} not found in result_store")
            return JSONResponse(status_code=404, content={
                "error": "Transcription not found",
                "detail": "No transcription data found for this task ID"
            })
    
    # File exists, check if it has content
    file_size = transcript_file.stat().st_size
    logging.info(f"Found transcript file for task {task_id}, size: {file_size} bytes")
    
    if file_size == 0:
        logging.error(f"Transcript file for task {task_id} is empty")
        return JSONResponse(status_code=400, content={
            "error": "Empty transcript file",
            "detail": "The transcript file exists but contains no data"
        })
    
    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
        
        if not transcript.strip():
            logging.error(f"Transcript file for task {task_id} contains only whitespace")
            return JSONResponse(status_code=400, content={
                "error": "Empty transcript content",
                "detail": "The transcript file contains only whitespace"
            })
        
        # Log a preview of the transcript
        preview = transcript[:100] + "..." if len(transcript) > 100 else transcript
        logging.info(f"Returning transcript for task {task_id}: {preview}")
        
        return JSONResponse(content={"task_id": task_id, "transcription": transcript})
    except Exception as e:
        logging.error(f"Error reading transcript file {transcript_file}: {e}")
        return JSONResponse(status_code=500, content={
            "error": "Failed to read transcript",
            "detail": str(e)
        })

@app.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """Check the result status of a task and verify file existence"""
    # Check if task is marked as cancelled
    if task_id in result_store and result_store[task_id].get("status") == "cancelled":
        return JSONResponse(content={
            "status": "cancelled",
            "message": "Transcription was cancelled"
        })
    
    if task_id not in result_store:
        # Check if there's a cancellation marker
        cancelled_file = Path(OUTPUT_DIR) / task_id / "cancelled.txt"
        if cancelled_file.exists():
            return JSONResponse(content={
                "status": "cancelled",
                "message": "Transcription was cancelled"
            })
        
        return JSONResponse(status_code=404, content={"error": f"Task {task_id} not found"})
    
    # Get the stored result
    result = result_store[task_id]
    
    # If there's a download URL, verify the file exists
    if "download_url" in result:
        # Extract the file path from the download URL
        # The format is "/download/{task_id}/transcript.txt"
        file_path = result["download_url"].replace("/download/", "")
        transcript_file = Path(OUTPUT_DIR) / file_path
        
        if not transcript_file.exists():
            # File doesn't exist, update the result to reflect this
            logging.warning(f"Task {task_id} has download URL but file {transcript_file} doesn't exist")
            result = {
                "status": "error",
                "message": "Transcript file not found. It may have been deleted or cleanup occurred."
            }
            result_store[task_id] = result
    
    return JSONResponse(content=result)

from starlette.websockets import WebSocketDisconnect

@app.websocket("/ws/progress/{task_id}")
async def progress_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    try:
        # First check if task is already cancelled or completed
        task_dir = os.path.join(OUTPUT_DIR, task_id)
        
        # Check for cancellation, completion, or error files upfront
        is_cancelled = os.path.exists(os.path.join(task_dir, "cancelled.txt"))
        is_completed = os.path.exists(os.path.join(task_dir, "completed.txt"))
        is_error = os.path.exists(os.path.join(task_dir, "error.txt"))
        
        if is_cancelled:
            await websocket.send_json({"progress": 100, "message": "Transcription cancelled"})
            return
            
        if is_completed and os.path.exists(os.path.join(task_dir, "transcript.txt")):
            await websocket.send_json({"progress": 100, "message": "Transcription complete"})
            return
            
        if is_error:
            try:
                with open(os.path.join(task_dir, "error.txt"), "r") as f:
                    error_message = f.read()
                await websocket.send_json({"progress": 100, "message": f"Error: {error_message}"})
            except:
                await websocket.send_json({"progress": 100, "message": "Error occurred during processing"})
            return
        
        # Send initial progress from memory store or create default
        current_progress = progress_store.get(task_id, {"progress": 5, "message": "Processing in progress..."})
        await websocket.send_json(current_progress)
        
        # Poll for updates from progress.json file and check for completion/cancellation
        last_progress = current_progress
        poll_interval = 0.5  # seconds
        
        while True:
            await asyncio.sleep(poll_interval)
            
            # Check for cancellation file
            if os.path.exists(os.path.join(task_dir, "cancelled.txt")):
                await websocket.send_json({"progress": 100, "message": "Transcription cancelled"})
                break
                
            # Check for completion file
            if os.path.exists(os.path.join(task_dir, "completed.txt")):
                await websocket.send_json({"progress": 100, "message": "Transcription complete"})
                break
                
            # Check for error file
            if os.path.exists(os.path.join(task_dir, "error.txt")):
                try:
                    with open(os.path.join(task_dir, "error.txt"), "r") as f:
                        error_message = f.read()
                    await websocket.send_json({"progress": 100, "message": f"Error: {error_message}"})
                except:
                    await websocket.send_json({"progress": 100, "message": "Error occurred during processing"})
                break
            
            # Check for progress file
            progress_file = os.path.join(task_dir, "progress.json")
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, "r") as f:
                        progress_data = json.load(f)
                    
                    # Only send if different from last update
                    if progress_data != last_progress:
                        await websocket.send_json(progress_data)
                        last_progress = progress_data
                        progress_store[task_id] = progress_data
                except:
                    pass
            
            # Check if process is still alive, if not, might be stuck
            if task_id in active_processes:
                process_info = active_processes[task_id]
                process = process_info.get('process')
                
                if process and not process.is_alive():
                    # Process no longer running, but no completion/error files
                    # Either still finalizing or something went wrong
                    if not (os.path.exists(os.path.join(task_dir, "completed.txt")) or 
                            os.path.exists(os.path.join(task_dir, "error.txt")) or
                            os.path.exists(os.path.join(task_dir, "cancelled.txt"))):
                        # Wait a bit longer in case files are being written
                        await asyncio.sleep(2)
                        
                        # Check again for completion/error files
                        if not (os.path.exists(os.path.join(task_dir, "completed.txt")) or 
                                os.path.exists(os.path.join(task_dir, "error.txt")) or
                                os.path.exists(os.path.join(task_dir, "cancelled.txt"))):
                            # Process ended but no status files - likely an error
                            await websocket.send_json({
                                "progress": 100, 
                                "message": "Process ended unexpectedly"
                            })
                            # Create an error file
                            with open(os.path.join(task_dir, "error.txt"), "w") as f:
                                f.write("Process ended unexpectedly")
                            break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f"Error in WebSocket handler: {e}")
        try:
            await websocket.send_json({"progress": 100, "message": f"Error: {str(e)}"})
        except:
            pass

@app.get("/download/{file_path:path}")
async def download_transcript(file_path: str):
    """Download the transcript file"""
    transcript_path = Path(OUTPUT_DIR) / file_path
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")
        
    return FileResponse(path=str(transcript_path), media_type="text/plain", filename=transcript_path.name)

@app.delete("/cleanup/{task_id}")
async def cleanup(task_id: str, preserve_uploads: bool = False):
    """Clean up files and directories associated with a task ID,
    but preserve transcript files for completed tasks and optionally preserve uploads"""
    logging.info(f"Starting cleanup for task {task_id}")
    
    # Check if the task is marked as completed and has a transcript
    is_completed_with_transcript = False
    transcript_file = Path(OUTPUT_DIR) / task_id / "transcript.txt"
    completed_file = Path(OUTPUT_DIR) / task_id / "completed.txt"
    
    if completed_file.exists() and transcript_file.exists():
        is_completed_with_transcript = True
        logging.info(f"Task {task_id} is completed with transcript, preserving files")
    
    # First try to cancel ongoing task if it's still running
    if task_id in active_processes:
        try:
            process_info = active_processes[task_id]
            try:
                # Kill process if still running
                parent = psutil.Process(process_info['pid'])
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except:
                        pass
                parent.kill()
            except:
                process = process_info.get('process')
                if process and process.is_alive():
                    process.terminate()
                    
            # Remove from active processes
            del active_processes[task_id]
            logging.info(f"Cancelled ongoing task {task_id}")
        except Exception as e:
            logging.error(f"Error cancelling process: {e}")
    
    # Track all removed files and directories
    files_removed = 0
    dirs_removed = 0
    
    # Only remove directories and files if not a completed task with transcript
    if not is_completed_with_transcript:
        # 1. Remove processed audio directory for this task
        task_processed_folder = Path(OUTPUT_DIR) / task_id
        if task_processed_folder.exists() and task_processed_folder.is_dir():
            try:
                # Use shutil.rmtree to recursively delete the directory and all contents
                shutil.rmtree(task_processed_folder)
                dirs_removed += 1
                logging.info(f"Removed processed audio folder for task {task_id}")
            except Exception as e:
                logging.error(f"Failed to remove processed audio folder {task_processed_folder}: {e}")
        else:
            logging.info(f"No processed audio folder found for task {task_id}")
        
        # 2. Only remove temporary upload files if not preserving uploads
        if not preserve_uploads:
            temp_files_pattern = f"{task_id}*"  # Match all files starting with the task ID
            for temp_file in temp_uploads.glob(temp_files_pattern):
                try:
                    temp_file.unlink()
                    files_removed += 1
                    logging.info(f"Removed temp file: {temp_file}")
                except Exception as e:
                    logging.error(f"Failed to remove temp file {temp_file}: {e}")
            
            if files_removed == 0:
                logging.info(f"No temp files found for task {task_id}")
            
            # Remove from uploaded_files dict
            if task_id in uploaded_files:
                del uploaded_files[task_id]
                logging.info(f"Removed task {task_id} from uploaded files")
        else:
            logging.info(f"Preserving uploaded files for task {task_id}")
    else:
        logging.info(f"Preserving files for completed task {task_id} with transcript")
    
   # Always clear memory data for progress
    if task_id in progress_store:
        del progress_store[task_id]
        logging.info(f"Cleared progress data for task {task_id}")
    
    # Clear original filename data
    if task_id in original_filenames:
        del original_filenames[task_id]
        logging.info(f"Cleared original filename data for task {task_id}")
    
    # Only clear result store if not completed with transcript
    if task_id in result_store and not is_completed_with_transcript:
        del result_store[task_id]
        logging.info(f"Cleared result data for task {task_id}")
    
    return JSONResponse(content={
        "status": "success",
        "message": f"Cleaned up task {task_id}",
        "details": {
            "files_removed": files_removed,
            "directories_removed": dirs_removed,
            "preserved_transcript": is_completed_with_transcript,
            "preserved_uploads": preserve_uploads
        }
    })

@app.post("/admin/cleanup")
async def manual_cleanup(hours: int = 1):
    """Manually trigger cleanup of files older than the specified hours"""
    try:
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        files_removed = 0
        dirs_removed = 0
        
        # Clean temporary upload files
        for file_path in Path("temp_uploads").glob("*"):
            if file_path.is_file():
                # Get the modification time of the file
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mod_time < cutoff_time:
                    try:
                        file_path.unlink()
                        files_removed += 1
                    except Exception as e:
                        logging.error(f"Failed to remove temp file {file_path}: {e}")
        
        # Clean processed audio directories
        for dir_path in Path(OUTPUT_DIR).glob("*"):
            if dir_path.is_dir():
                # Get the modification time of the directory
                try:
                    # Use the most recent file in the directory to determine age
                    most_recent = max(
                        [f.stat().st_mtime for f in dir_path.glob("**/*") if f.is_file()], 
                        default=dir_path.stat().st_mtime
                    )
                    mod_time = datetime.fromtimestamp(most_recent)
                    
                    if mod_time < cutoff_time:
                        shutil.rmtree(dir_path)
                        dirs_removed += 1
                except Exception as e:
                    logging.error(f"Failed to remove directory {dir_path}: {e}")
        
        return JSONResponse(content={
            "status": "success", 
            "message": f"Removed {files_removed} files and {dirs_removed} directories older than {hours} hours"
        })
    except Exception as e:
        logging.error(f"Error in manual cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
@app.post("/cleanup/{task_id}")
async def cleanup_on_refresh(task_id: str):
    """Handle cleanup requests from navigator.sendBeacon (used on page refresh/close).
    This version preserves uploaded files."""
    logging.info(f"Received cleanup request via POST for task {task_id} (likely page refresh)")
    
    # Use the same cleanup logic as the DELETE endpoint, but with preserve_uploads=True
    response = await cleanup(task_id, preserve_uploads=True)  # Changed from False to True
    
    return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

# =============================================================================
# Main Entry Point
# =============================================================================

def setup_model_directories():
    """Create the model directory structure if it doesn't exist"""
    cache_dir = env_config['model_cache_dir']
    
    # Create main model directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create subdirectories for each model type
    model_dirs = [
        "whisper", "whisper-ft", "resepformer", "resepformer-ft",
        "speaker-diarization", "vad", "embedding"
    ]
    
    for dir_name in model_dirs:
        os.makedirs(os.path.join(cache_dir, dir_name), exist_ok=True)
    
    logging.info(f"Model directory structure created at {cache_dir}")

def setup_torch_optimizations():
    """Setup PyTorch optimizations for faster processing"""
    # Enable TF32 for faster processing if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logging.info("PyTorch optimizations enabled")

if __name__ == "__main__":
    # Setup optimizations
    setup_torch_optimizations()
    
    # Create model directories 
    setup_model_directories()
    
    uvicorn.run(app, host="0.0.0.0", port=env_config['api_port'])

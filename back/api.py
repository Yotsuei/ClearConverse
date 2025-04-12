# back/api.py

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

# =============================================================================
# Data Classes and Utility Functions (Same as your pipeline)
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
    min_segment_duration: float = 0.50  
    overlap_threshold: float = 0.50  
    condition_on_previous_text: bool = True
    merge_gap_threshold: float = 0.5  
    min_overlap_duration_for_separation: float = 0.5  
    max_embedding_segments: int = 100  
    enhance_separated_audio: bool = True
    use_vad_refinement: bool = True
    speaker_embedding_threshold: float = 0.50  
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
    secondary_diarization_threshold: float = 0.40

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
    
    Args:
        file_id: The Google Drive file ID
        output_path: Optional path to save the file, if None a temporary file will be created
        
    Returns:
        The path to the downloaded file
    """
    import requests
    import tempfile
    
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
    
    # If we found a token, we need a second request with the token
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
    
    # Check content type to ensure we're getting a file, not an HTML page
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        logging.warning(f"Received HTML content instead of file. This might indicate access restrictions.")
        # We could parse the HTML to extract a download link, but that's more complex
    
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
        # Just do a basic validation for Google Drive URLs
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
    
    # For non-Google Drive URLs, do the normal validation
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
        
        # If we can't determine from content-type, check URL extension
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

# =============================================================================
# Enhanced Audio Processor Class
# =============================================================================

class EnhancedAudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler = None
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
            # First, load the base model to ensure we have a fallback
            self.separator = SepformerSeparation.from_hparams(
                source="speechbrain/resepformer-wsj02mix",
                savedir=resepformer_path,
                run_opts={"device": self.device}
            )
            
            # If fine-tuned model exists, try to load it
            if os.path.exists(ft_model_path):
                logging.info("Found fine-tuned RESepFormer model. Attempting to load...")
                
                # Similar to the Google Colab approach, create a temporary working directory
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
                        
                        # Apply the component weights to our model
                        self.separator.load_state_dict(state_dict, strict=False)
                        logging.info("Fine-tuned RESepFormer model loaded successfully!")
                    else:
                        logging.warning(f"Missing required files for fine-tuned RESepFormer. Using base model instead.")
            else:
                logging.info("No fine-tuned RESepFormer model found. Using base model.")
        
        except Exception as e:
            logging.error(f"Failed to load RESepFormer model: {str(e)}")
            # Attempt fallback to base model if we haven't loaded it yet
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
        for start, end, speaker in all_segments:
            if (end - start) >= 1.0:
                speaker_segments[speaker].append((start, end))
        speaker_embeddings = {}
        for speaker, segments in speaker_segments.items():
            logging.info(f"Building profile for speaker {speaker} using {len(segments)} segments")
            segments.sort(key=lambda x: x[1]-x[0], reverse=True)
            selected_segments = segments[:self.config.max_embedding_segments]
            embeddings = []
            for start, end in selected_segments:
                segment_audio = self._extract_segment(audio, start, end)
                embedding = self._extract_embedding(segment_audio)
                if embedding is not None:
                    embeddings.append(embedding)
            if embeddings:
                speaker_embeddings[speaker] = torch.stack(embeddings).mean(dim=0)
                logging.info(f"Created embedding for speaker {speaker}")
        return speaker_embeddings

    def _resegment_overlap(self, audio_segment: torch.Tensor, seg_start: float, seg_end: float, speaker_profiles: Dict[str, torch.Tensor]) -> List[Tuple[float, float, str]]:
        window_size = self.config.sliding_window_size
        step = self.config.sliding_window_step
        refined_segments = []
        curr = seg_start
        window_results = []
        while curr + window_size <= seg_end:
            segment = self._extract_segment(audio_segment, curr - seg_start, curr - seg_start + window_size)
            embedding = self._extract_embedding(segment)
            if embedding is not None:
                similarities = [(spk, self._calculate_embedding_similarity(embedding, profile))
                                for spk, profile in speaker_profiles.items()]
                similarities.sort(key=lambda x: x[1], reverse=True)
                dominant_speaker, confidence = similarities[0]
            else:
                dominant_speaker, confidence = "UNKNOWN", 0.0
            window_results.append((curr, curr + window_size, dominant_speaker, confidence))
            curr += step
        if not window_results:
            return [(seg_start, seg_end, "UNKNOWN")]
        merged = []
        cur_start, cur_end, cur_spk, _ = window_results[0]
        for start, end, spk, conf in window_results[1:]:
            if spk == cur_spk and start - cur_end <= step:
                cur_end = end
            else:
                merged.append((cur_start, cur_end, cur_spk))
                cur_start, cur_end, cur_spk = start, end, spk
        merged.append((cur_start, cur_end, cur_spk))
        return [(max(seg_start, s), min(seg_end, e), spk) for s, e, spk in merged]

    def _process_overlap_segment(self, audio_segment: torch.Tensor, speaker_embeddings: Dict[str, torch.Tensor],
                                   involved_speakers: List[str], seg_start: float, seg_end: float) -> List[Dict]:
        logging.info(f"Processing overlap segment: {seg_start:.2f}s-{seg_end:.2f}s")
        refined_regions = self._resegment_overlap(audio_segment, seg_start, seg_end, speaker_embeddings)
        results = []
        for new_start, new_end, spk in refined_regions:
            subsegment = self._extract_segment(audio_segment, new_start - seg_start, new_end - seg_start)
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
            transcription = self.whisper_model.transcribe(
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

    def run(self, input_file, output_dir: str = "processed_audio", debug_mode: bool = False, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(5, "Starting processing")
                
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Processing file: {input_file}")
            
            if progress_callback:
                progress_callback(30, "Running file processing")
                
            # Process the audio file
            results = self.process_file(input_file)
            
            if progress_callback:
                progress_callback(60, "Saving processed segments")
                
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
                
            with open(transcript_path, "w", encoding='utf-8') as f:
                f.write(transcript)
                
            logging.info(f"Transcript saved to: {transcript_path}")
            
            if progress_callback:
                progress_callback(100, "Processing completed")
                
            return input_file, transcript, transcript_path
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            traceback.print_exc()
            raise

    def process_file(self, file_path: str) -> Dict:
        try:
            audio, sample_rate = self.load_audio(file_path)
            audio_duration = audio.shape[-1] / sample_rate
            logging.info(f"Processing audio file: {audio_duration:.2f} seconds")
            
            # Run Voice Activity Detection
            logging.info("Running Voice Activity Detection...")
            vad_result = self.vad_pipeline(file_path)
            vad_intervals = get_pyannote_vad_intervals(vad_result)
            logging.info(f"VAD detected {len(vad_intervals)} speech intervals")
            
            # Run Speaker Diarization
            logging.info("Running Speaker Diarization...")
            diarization_result = self.diarization(
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
            
            # Process each segment
            processed_segments = []
            meta_counts = {'SPEAKER_A': 0, 'SPEAKER_B': 0}
            
            for seg_start, seg_end, orig_speaker in refined_segments:
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
                
                audio_segment = self._extract_segment(audio, seg_start, seg_end)
                spk_label = speaker_mapping.get(orig_speaker, "UNKNOWN")
                
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
                                transcription = self.whisper_model.transcribe(
                                    sub_audio.squeeze().cpu().numpy(),
                                    initial_prompt="This is a clear conversation with complete sentences.",
                                    word_timestamps=True,
                                    condition_on_previous_text=self.config.condition_on_previous_text,
                                    temperature=self.config.temperature
                                )
                                final_label = speaker_mapping.get(new_spk, spk_label)
                                processed_segments.append(AudioSegment(
                                    start=seg_start + new_start,
                                    end=seg_start + new_end,
                                    speaker_id=final_label,
                                    audio_tensor=sub_audio,
                                    is_overlap=False,
                                    transcription=transcription['text'],
                                    confidence=1.0
                                ))
                                meta_counts[final_label] = meta_counts.get(final_label, 0) + 1
                            continue
                
                # Process overlapping segments with special handling
                if is_overlap:
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
                else:
                    # Standard segment processing
                    transcription = self.whisper_model.transcribe(
                        audio_segment.squeeze().cpu().numpy(),
                        initial_prompt="This is a conversation between two people.",
                        word_timestamps=True,
                        condition_on_previous_text=self.config.condition_on_previous_text,
                        temperature=self.config.temperature
                    )
                    processed_segments.append(AudioSegment(
                        start=seg_start,
                        end=seg_end,
                        speaker_id=spk_label,
                        audio_tensor=audio_segment,
                        is_overlap=False,
                        transcription=transcription['text'],
                        confidence=1.0
                    ))
                    meta_counts[spk_label] += 1
            
            # Sort segments by start time
            processed_segments.sort(key=lambda x: x.start)
            
            # Build metadata for the result
            metadata = {
                'duration': audio_duration,
                'speaker_a_segments': meta_counts.get('SPEAKER_A', 0),
                'speaker_b_segments': meta_counts.get('SPEAKER_B', 0),
                'total_segments': len(processed_segments),
                'speakers': list(speaker_mapping.values())
            }
            
            return {'segments': processed_segments, 'metadata': metadata}
        except Exception as e:
            logging.error(f"Error in process_file: {e}")
            traceback.print_exc()
            raise

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
    
    Args:
        max_age_hours: Maximum age of files in hours before they're deleted
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
processor = EnhancedAudioProcessor(config)

# State storage
uploaded_files = {}
progress_store = {}
result_store = {}

# =============================================================================
# Helper Functions for Endpoints
# =============================================================================

def update_progress(task_id: str, percent: int, message: str):
    """Update and log progress for a given task"""
    progress_store[task_id] = {"progress": percent, "message": message}
    logging.info(f"Task {task_id}: {percent}% - {message}")

async def process_audio_with_progress(task_id: str, file_path: str):
    """Process audio file with progress updates"""
    try:
        task_output_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Run the processing with progress callback
        input_file, transcript, transcript_path = processor.run(
            file_path, 
            output_dir=task_output_dir, 
            debug_mode=False, 
            progress_callback=lambda p, m: update_progress(task_id, 30 + int(p * 0.7), m)
        )
        
        # Store the result
        result_store[task_id] = {"download_url": f"/download/{task_id}/transcript.txt"}
        update_progress(task_id, 100, "Transcription complete")
    except Exception as e:
        # Handle errors
        update_progress(task_id, 100, f"Error: {str(e)}")
        result_store[task_id] = {"error": str(e)}
        logging.error(f"Error processing audio: {e}")
        traceback.print_exc()

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload an audio file for processing"""
    if not file.filename.endswith((".mp3", ".wav", ".ogg", ".mp4", ".flac", ".m4a", ".aac")):
        raise HTTPException(status_code=400, detail="Invalid file type provided.")
        
    # Generate a task ID and save the file
    task_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    filename = f"{task_id}{extension}"
    file_path = temp_uploads / filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Store the file path for later processing
    uploaded_files[task_id] = str(file_path)
    update_progress(task_id, 0, "File uploaded and saved")
    
    return JSONResponse(content={"task_id": task_id, "preview_url": f"/preview/{filename}"})

@app.post("/upload-url")
async def upload_url(url: str = Form(...)):
    """Endpoint to process an audio file from a URL"""
    validate_url(url)
    
    # Generate task ID and prepare for download
    task_id = str(uuid.uuid4())
    parsed_url = urlparse(url)
    extension = os.path.splitext(parsed_url.path)[1]
    
    if extension.lower() not in ['.mp3', '.wav', '.ogg', '.mp4', '.flac', '.m4a', '.aac']:
        extension = ".mp3"  # Default to mp3 if extension is unknown
        
    filename = f"{task_id}{extension}"
    file_path = temp_uploads / filename
    
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
                raise HTTPException(status_code=400, detail="Could not extract file ID from Google Drive URL")
            
            logging.info(f"Extracted Google Drive file ID: {file_id}")
            update_progress(task_id, 10, "Downloading from Google Drive")
            
            # Use the special Google Drive download function
            file_path = download_file_from_google_drive(file_id, str(file_path))
            update_progress(task_id, 25, "Download complete")
        else:
            # For non-Google Drive URLs, use the regular download function
            update_progress(task_id, 5, "Downloading audio from URL")
            with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            update_progress(task_id, 25, "Download complete")
    except Exception as e:
        logging.error(f"Error downloading file from URL {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
        
    # Store the file path for later processing
    uploaded_files[task_id] = str(file_path)
    
    return JSONResponse(content={
        "task_id": task_id,
        "preview_url": f"/preview/{filename}" 
    })

@app.post("/transcribe/{task_id}")
async def transcribe_task(task_id: str, background_tasks: BackgroundTasks):
    """Start transcription process for a previously uploaded file"""
    if task_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Task ID not found. Please upload a file or URL first.")
        
    file_path = uploaded_files[task_id]
    update_progress(task_id, 0, "Task queued for transcription")
    
    # Process audio in background
    background_tasks.add_task(process_audio_with_progress, task_id, file_path)
    
    return JSONResponse(content={"task_id": task_id})

@app.get("/preview/{filename}")
async def preview_audio(filename: str):
    """Serve uploaded audio file for preview"""
    file_path = temp_uploads / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(str(file_path), media_type="audio/mpeg", filename=filename)

@app.get("/transcription/{task_id}")
async def get_transcription(task_id: str):
    """Get the transcription text for a completed task"""
    transcript_file = Path(OUTPUT_DIR) / task_id / "transcript.txt"
    
    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    return JSONResponse(content={"task_id": task_id, "transcription": transcript})

@app.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """Check the result status of a task"""
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    return JSONResponse(content=result_store[task_id])

from starlette.websockets import WebSocketDisconnect

@app.websocket("/ws/progress/{task_id}")
async def progress_ws(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    
    try:
        # Get current progress data
        current_data = progress_store.get(task_id, {"progress": 0, "message": "Initializing..."})
        
        # Send initial data
        await websocket.send_json(current_data)
        
        # Exit early if task is already completed
        if current_data.get("progress", 0) >= 100:
            return
        
        # Poll for updates
        while True:
            await asyncio.sleep(0.5)
            
            new_data = progress_store.get(task_id, current_data)
            
            # Send update only if data changed
            if new_data != current_data:
                try:
                    await websocket.send_json(new_data)
                    current_data = new_data
                except RuntimeError:
                    # Connection was closed while we were sending
                    break
            
            # Stop polling once task is complete
            if new_data.get("progress", 0) >= 100:
                break
                
    except WebSocketDisconnect:
        # Client disconnected, just log at debug level since this is normal
        logging.debug(f"WebSocket client disconnected for task {task_id}")
    except Exception as e:
        # Log other errors but don't crash
        logging.error(f"WebSocket error for task {task_id}: {str(e)}")
    finally:
        pass

@app.get("/download/{file_path:path}")
async def download_transcript(file_path: str):
    """Download the transcript file"""
    transcript_path = Path(OUTPUT_DIR) / file_path
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")
        
    return FileResponse(path=str(transcript_path), media_type="text/plain", filename=transcript_path.name)

@app.delete("/cleanup/{task_id}")
async def cleanup(task_id: str):
    """Clean up temporary files for a task"""
    # Remove processed folder
    task_processed_folder = Path(OUTPUT_DIR) / task_id
    if task_processed_folder.exists() and task_processed_folder.is_dir():
        shutil.rmtree(task_processed_folder)
        logging.info(f"Cleared processed audio folder for task {task_id}")
    else:
        logging.info(f"No processed audio folder found for task {task_id}")
    
    # Remove temporary upload files
    files_removed = 0
    for temp_file in temp_uploads.iterdir():
        if temp_file.name.startswith(task_id):
            try:
                temp_file.unlink()
                files_removed += 1
                logging.info(f"Removed temp file: {temp_file}")
            except Exception as e:
                logging.error(f"Failed to remove temp file {temp_file}: {e}")
                
    if files_removed == 0:
        logging.info(f"No temp files found for task {task_id}")
        
    return JSONResponse(content={"status": f"Cleaned up files for task {task_id}"})

# Add this endpoint to your FastAPI app
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
    
    # You could add other optimizations here in the future
    logging.info("PyTorch optimizations enabled")

if __name__ == "__main__":
    # Setup optimizations
    setup_torch_optimizations()
    
    # Create model directories 
    setup_model_directories()
    
    uvicorn.run(app, host="0.0.0.0", port=env_config['api_port'])
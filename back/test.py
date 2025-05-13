# System imports
import os
import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import shutil
import csv
import argparse
from datetime import datetime
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

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Filter specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Adjust logging level to ignore warnings from libraries
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("whisper").setLevel(logging.ERROR)

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
    min_overlap_duration_for_separation: float = 0.60  
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
        import subprocess
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
        cache_dir = 'models'  # Default cache directory for models
        
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
                self._load_resepformer_model('models')
                self.models_loaded['resepformer'] = True
            
            # Load Whisper (25% of loading time)
            if not self.models_loaded['whisper']:
                if progress_callback:
                    progress_callback(35, "Loading Whisper...")
                self._load_whisper_model('models')
                self.models_loaded['whisper'] = True
            
            # Load PyAnnote (50% of loading time - these are larger)
            if not self.models_loaded['pyannote']:
                if progress_callback:
                    progress_callback(60, "Loading speaker diarization tool...")
                self._initialize_pyannote_models('models')
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
                    import subprocess
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
# Threshold Testing Functions
# =============================================================================

# =============================================================================
# Threshold Testing Functions
# =============================================================================

def test_diarization_configurations(audio_files, configurations, auth_token, output_dir="config_tests"):
    """
    Test various configuration combinations on audio files.
    
    Args:
        audio_files: List of paths to audio files to test
        configurations: List of configuration dictionaries with parameter values to test
        auth_token: HuggingFace auth token for PyAnnote models
        output_dir: Directory to store test results
    
    Returns:
        Dictionary with test results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test configuration
    config_path = os.path.join(output_dir, "test_config.json")
    test_config = {
        "audio_files": audio_files,
        "configurations": configurations,
        "timestamp": datetime.now().isoformat()
    }
    with open(config_path, "w") as f:
        json.dump(test_config, f, indent=2)
    
    # Create evaluation CSV template
    csv_path = os.path.join(output_dir, "evaluation.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Create header with all configuration parameters plus evaluation columns
        header = [
            "Audio File", "Config ID", 
            "min_segment_duration", "overlap_threshold", "merge_gap_threshold",
            "min_overlap_duration_for_separation", "speaker_embedding_threshold",
            "noise_reduction_amount", "sliding_window_size", "sliding_window_step",
            "secondary_diarization_threshold",
            "Speaker Attribution Score (1-5)", "Overlap Handling Score (1-5)",
            "False Speaker Changes", "Missed Speaker Changes", "Overall Quality (1-5)",
            "Notes"
        ]
        writer.writerow(header)
        
        # Add rows for each configuration and audio file
        for i, config in enumerate(configurations):
            config_id = f"config_{i+1:02d}"
            for audio_file in audio_files:
                base_name = os.path.basename(audio_file)
                row = [
                    base_name, config_id,
                    config.get("min_segment_duration", ""),
                    config.get("overlap_threshold", ""),
                    config.get("merge_gap_threshold", ""),
                    config.get("min_overlap_duration_for_separation", ""),
                    config.get("speaker_embedding_threshold", ""),
                    config.get("noise_reduction_amount", ""),
                    config.get("sliding_window_size", ""),
                    config.get("sliding_window_step", ""),
                    config.get("secondary_diarization_threshold", ""),
                    "", "", "", "", "", ""  # Empty fields for manual evaluation
                ]
                writer.writerow(row)
    
    # Process each file with each configuration
    results = {}
    
    # Create a configuration lookup with IDs
    config_lookup = {f"config_{i+1:02d}": config for i, config in enumerate(configurations)}
    
    for audio_file in audio_files:
        base_name = os.path.basename(audio_file)
        file_results = {}
        
        logging.info(f"Processing file: {base_name}")
        
        for config_id, config_params in config_lookup.items():
            logging.info(f"Testing configuration: {config_id}")
            
            # Create a formatted parameter string for directory naming
            params_str = "_".join([f"{k}{v}".replace(".", "p") for k, v in config_params.items()])
            test_name = f"{base_name}__{config_id}"
            test_output_dir = os.path.join(output_dir, test_name)
            os.makedirs(test_output_dir, exist_ok=True)
            
            # Save the configuration details to the test directory
            with open(os.path.join(test_output_dir, "config_details.json"), "w") as f:
                json.dump(config_params, f, indent=2)
            
            # Create base config with default values
            base_config = Config(auth_token=auth_token)
            
            # Update config with test parameters
            for param, value in config_params.items():
                if hasattr(base_config, param):
                    setattr(base_config, param, value)
                else:
                    logging.warning(f"Parameter {param} not found in Config class, skipping")
            
            # Process the file with these settings
            processor = EnhancedAudioProcessor(base_config, load_models_immediately=True)
            
            try:
                _, transcript, transcript_path = processor.run(
                    audio_file, 
                    output_dir=test_output_dir,
                    debug_mode=True
                )
                
                # Store result
                file_results[config_id] = {
                    "success": True,
                    "transcript_path": transcript_path,
                    "output_dir": test_output_dir,
                    "config": config_params
                }
                
                # Copy transcript to root with unique name for easier comparison
                if transcript_path and os.path.exists(transcript_path):
                    shutil.copy(
                        transcript_path,
                        os.path.join(output_dir, f"{test_name}_transcript.txt")
                    )
                
            except Exception as e:
                logging.error(f"Error processing with configuration {config_id}: {e}")
                file_results[config_id] = {
                    "success": False,
                    "error": str(e),
                    "output_dir": test_output_dir,
                    "config": config_params
                }
                
                # Save error information
                with open(os.path.join(test_output_dir, "error.txt"), "w") as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(traceback.format_exc())
        
        results[base_name] = file_results
    
    # Save overall results
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        # Create a serializable summary
        summary = {}
        for file_name, file_results in results.items():
            summary[file_name] = {}
            for config_id, result in file_results.items():
                summary[file_name][config_id] = {
                    "success": result["success"],
                    "output_dir": result.get("output_dir", ""),
                    "error": result.get("error", "") if not result["success"] else ""
                }
        json.dump(summary, f, indent=2)
    
    # Create a comparison HTML file for easy side-by-side transcript review
    create_comparison_html(audio_files, config_lookup, results, output_dir)
    
    print(f"\nTesting completed! Results saved to {output_dir}")
    print(f"Please open {csv_path} to manually evaluate the results.")
    print(f"You can also open {os.path.join(output_dir, 'transcript_comparison.html')} to compare transcripts side by side.")
    
    return results

def create_comparison_html(audio_files, config_lookup, results, output_dir):
    """Create an HTML file for side-by-side comparison of transcripts"""
    html_path = os.path.join(output_dir, "transcript_comparison.html")
    
    with open(html_path, "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Diarization Configuration Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .file-section { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }
                .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
                .config-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .config-card h3 { margin-top: 0; color: #0066cc; }
                .config-card pre { white-space: pre-wrap; background: #f5f5f5; padding: 10px; }
                .config-params { font-size: 12px; color: #666; margin-bottom: 10px; }
                .success { color: green; }
                .error { color: red; }
                .transcript { max-height: 400px; overflow-y: auto; }
                table { border-collapse: collapse; width: 100%; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Diarization Configuration Comparison</h1>
        """)
        
        # For each audio file
        for audio_file in audio_files:
            base_name = os.path.basename(audio_file)
            file_results = results.get(base_name, {})
            
            f.write(f"<div class='file-section'><h2>Audio File: {base_name}</h2>")
            
            # Add configuration table
            f.write("<h3>Configurations</h3>")
            f.write("<table><tr><th>Config ID</th>")
            
            # Get all parameter names from the first config
            if config_lookup:
                first_config = next(iter(config_lookup.values()))
                param_names = list(first_config.keys())
                for param in param_names:
                    f.write(f"<th>{param}</th>")
            
            f.write("<th>Status</th></tr>")
            
            # Add rows for each configuration
            for config_id, config_params in config_lookup.items():
                result = file_results.get(config_id, {})
                success = result.get("success", False)
                
                f.write(f"<tr><td>{config_id}</td>")
                
                for param in param_names:
                    f.write(f"<td>{config_params.get(param, '')}</td>")
                
                status_class = "success" if success else "error"
                status_text = "Success" if success else f"Error: {result.get('error', 'Unknown error')}"
                f.write(f"<td class='{status_class}'>{status_text}</td></tr>")
            
            f.write("</table>")
            
            # Add transcript comparison
            f.write("<h3>Transcripts</h3><div class='config-grid'>")
            
            for config_id, result in file_results.items():
                transcript_path = result.get("transcript_path", "")
                output_dir = result.get("output_dir", "")
                config = result.get("config", {})
                success = result.get("success", False)
                
                f.write(f"<div class='config-card'>")
                f.write(f"<h3>{config_id}</h3>")
                
                # Display configuration parameters
                f.write("<div class='config-params'>")
                for param, value in config.items():
                    f.write(f"{param}: {value}<br>")
                f.write("</div>")
                
                # Display status
                status_class = "success" if success else "error"
                status_text = "Success" if success else f"Error: {result.get('error', 'Unknown error')}"
                f.write(f"<p class='{status_class}'>{status_text}</p>")
                
                # Display transcript if available
                if success and transcript_path and os.path.exists(transcript_path):
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as t_file:
                            transcript = t_file.read()
                        f.write("<div class='transcript'><pre>")
                        f.write(transcript.replace("<", "&lt;").replace(">", "&gt;"))
                        f.write("</pre></div>")
                    except Exception as e:
                        f.write(f"<p class='error'>Error reading transcript: {str(e)}</p>")
                else:
                    f.write("<p>No transcript available</p>")
                
                f.write("</div>")
            
            f.write("</div></div>")
        
        f.write("""
        <script>
            // Add a function to highlight differences in speaker attribution
            function initComparison() {
                console.log("Initializing comparison tool...");
                // Add interactive comparison features if needed
            }
            document.addEventListener('DOMContentLoaded', initComparison);
        </script>
        </body>
        </html>
        """)

def progress_callback(percent, message):
    """Simple progress callback function"""
    print(f"Progress: {percent}% - {message}")

# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    # Load environment variables from .env.development file
    load_dotenv(dotenv_path='.env.development')
    
    parser = argparse.ArgumentParser(description='Test Speaker Diarization Configurations')
    
    # File selection - either a single file or a directory
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument('--audio_file', type=str, help='Path to a specific audio file to test')
    file_group.add_argument('--audio_dir', type=str, help='Directory containing audio files to test')
    
    parser.add_argument('--output_dir', type=str, default='config_tests', help='Directory to store test results')
    parser.add_argument('--auth_token', type=str, help='HuggingFace auth token for PyAnnote models (defaults to HF_AUTH_TOKEN env variable)')
    parser.add_argument('--file_extensions', type=str, default='.mp3,.wav', help='Comma-separated list of file extensions to process (only used with --audio_dir)')
    parser.add_argument('--limit', type=int, help='Limit the number of configurations to test (useful for testing the script)')
    
    args = parser.parse_args()
    
    # Use auth token from environment variable if not provided via command line
    if not args.auth_token:
        args.auth_token = os.getenv('HF_AUTH_TOKEN')
        if not args.auth_token:
            print("Error: HF_AUTH_TOKEN not found in environment variables and --auth_token not provided")
            exit(1)
    
    # Define configurations to test
    configurations = []
    
    # Base configuration
    base_config = {
        "min_segment_duration": 0.45,
        "overlap_threshold": 0.50,
        "merge_gap_threshold": 0.50,
        "min_overlap_duration_for_separation": 0.60,
        "speaker_embedding_threshold": 0.40,
        "noise_reduction_amount": 0.50,
        "sliding_window_size": 0.80,
        "sliding_window_step": 0.40,
        "secondary_diarization_threshold": 0.30
    }
    
    # Add base config as first configuration
    configurations.append({**base_config})
    
    # Add min_segment_duration variations
    for value in [0.35, 0.45, 0.55]:
        if value != base_config["min_segment_duration"]:
            configurations.append({
                **base_config,
                "min_segment_duration": value
            })
    
    # Add overlap_threshold variations
    for value in [0.40, 0.50, 0.60]:
        if value != base_config["overlap_threshold"]:
            configurations.append({
                **base_config,
                "overlap_threshold": value
            })
    
    # Add merge_gap_threshold variations
    for value in [0.40, 0.50, 0.60]:
        if value != base_config["merge_gap_threshold"]:
            configurations.append({
                **base_config,
                "merge_gap_threshold": value
            })
    
    # Add min_overlap_duration_for_separation variations
    for value in [0.50, 0.60, 0.70]:
        if value != base_config["min_overlap_duration_for_separation"]:
            configurations.append({
                **base_config,
                "min_overlap_duration_for_separation": value
            })
    
    # Add speaker_embedding_threshold variations
    for value in [0.40, 0.50, 0.80]:
        if value != base_config["speaker_embedding_threshold"]:
            configurations.append({
                **base_config,
                "speaker_embedding_threshold": value
            })
    
    # Add noise_reduction_amount variations
    for value in [0.30, 0.50, 0.70]:
        if value != base_config["noise_reduction_amount"]:
            configurations.append({
                **base_config,
                "noise_reduction_amount": value
            })
    
    # Add sliding_window_size variations
    for value in [0.60, 0.80, 1.00]:
        if value != base_config["sliding_window_size"]:
            configurations.append({
                **base_config,
                "sliding_window_size": value
            })
    
    # Add sliding_window_step variations
    for value in [0.30, 0.40, 0.50]:
        if value != base_config["sliding_window_step"]:
            configurations.append({
                **base_config,
                "sliding_window_step": value
            })
    
    # Add secondary_diarization_threshold variations
    for value in [0.30, 0.40, 0.70]:
        if value != base_config["secondary_diarization_threshold"]:
            configurations.append({
                **base_config,
                "secondary_diarization_threshold": value
            })
    
    # Add a few combined variations (these are just examples, you may want to customize)
    # Example: Configuration optimized for fast exchanges
    configurations.append({
        **base_config,
        "min_segment_duration": 0.35,
        "overlap_threshold": 0.40,
        "merge_gap_threshold": 0.40,
        "sliding_window_size": 0.60,
        "sliding_window_step": 0.30,
        "secondary_diarization_threshold": 0.25
    })
    
    # Example: Configuration optimized for cleaner speech
    configurations.append({
        **base_config,
        "min_segment_duration": 0.55,
        "noise_reduction_amount": 0.70,
        "speaker_embedding_threshold": 0.45,
        "secondary_diarization_threshold": 0.35
    })
    
    # Example: Configuration for high-overlap scenarios
    configurations.append({
        **base_config,
        "overlap_threshold": 0.40,
        "min_overlap_duration_for_separation": 0.50,
        "sliding_window_size": 0.60,
        "sliding_window_step": 0.30
    })
    
    # Limit the number of configurations if requested
    if args.limit and args.limit > 0 and args.limit < len(configurations):
        print(f"Limiting to first {args.limit} configurations (out of {len(configurations)} total)")
        configurations = configurations[:args.limit]
    
    # Determine which audio files to process
    audio_files = []
    
    if args.audio_file:
        # Process a single specified file
        audio_file = Path(args.audio_file)
        if not audio_file.exists():
            print(f"Error: Audio file '{args.audio_file}' not found")
            exit(1)
        
        # Check if it's a valid audio file
        if not any(str(audio_file).lower().endswith(ext) for ext in ['.mp3', '.wav']):
            print(f"Error: File '{args.audio_file}' is not a supported audio format (must be .mp3 or .wav)")
            exit(1)
            
        audio_files.append(str(audio_file))
        print(f"Using audio file: {audio_file}")
    else:
        # Process all files in the specified directory
        extensions = args.file_extensions.split(',')
        
        for ext in extensions:
            audio_files.extend([str(f) for f in Path(args.audio_dir).glob(f"*{ext}")])
        
        if not audio_files:
            print(f"No audio files found in {args.audio_dir} with extensions {extensions}")
            exit(1)
        
        print(f"Found {len(audio_files)} audio files to process:")
        for file in audio_files:
            print(f"  - {file}")
    
    print(f"Will test {len(configurations)} different configurations")
    for i, config in enumerate(configurations):
        print(f"  Config {i+1}:")
        for param, value in config.items():
            print(f"    {param}: {value}")
        print()
    
    # Confirm before starting
    confirm = input(f"\nThis will test {len(configurations)} configurations on {len(audio_files)} file(s). Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled")
        exit(0)
    
    # Run the tests
    results = test_diarization_configurations(
        audio_files,
        configurations,
        args.auth_token,
        args.output_dir
    )
    
    print("\nTesting completed! Please manually evaluate the transcripts in the output directory.")
    print(f"Use the evaluation.csv file in {args.output_dir} to record your findings.")
    print(f"You can also view transcripts side-by-side in {os.path.join(args.output_dir, 'transcript_comparison.html')}")
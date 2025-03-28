#!/usr/bin/env python
import os
import json
import shutil
import logging
import traceback
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

# Import necessary libraries for audio processing
import torch
import torchaudio
import whisper
import noisereduce as nr
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# External libraries for diarization, separation and VAD
from pyannote.audio import Pipeline, Inference
from speechbrain.inference import SepformerSeparation

# =============================================================================
# Logging setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    # Increase minimum segment duration to ensure segments are long enough for reliable embedding extraction.
    min_segment_duration: float = 0.75  
    # Slightly higher threshold to detect overlaps (adjust based on audio characteristics).
    overlap_threshold: float = 0.65  
    condition_on_previous_text: bool = True
    # Tighten gap threshold to avoid merging distinct speaker turns.
    merge_gap_threshold: float = 0.5  
    # Increase the minimum duration for overlap separation.
    min_overlap_duration_for_separation: float = 0.6  
    # Use more segments for a more robust speaker embedding average.
    max_embedding_segments: int = 100  
    enhance_separated_audio: bool = True
    use_vad_refinement: bool = True
    # Adjust the speaker embedding threshold if needed for more sensitive matching.
    speaker_embedding_threshold: float = 0.45  
    # Increase noise reduction to help cleaner input for diarization and embedding.
    noise_reduction_amount: float = 0.65  
    transcription_batch_size: int = 8
    use_speaker_embeddings: bool = True
    temperature: float = 0.1
    max_speakers: int = 2
    min_speakers: int = 1
    whisper_model_size: str = "small.en"
    transcribe_overlaps_individually: bool = True
    # Increase window size for overlap segmentation to capture more context.
    sliding_window_size: float = 1.0  
    # Use a finer step to get smoother segmentation in overlaps.
    sliding_window_step: float = 0.5  
    # Increase the secondary diarization threshold to catch more misclassifications.
    secondary_diarization_threshold: float = 0.35

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
# EnhancedAudioProcessor Class (Same as provided)
# =============================================================================

class EnhancedAudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler = None
        self._initialize_models()

    def _initialize_models(self):
        logging.info(f"Initializing models on {self.device}...")
        cache_dir = "models"  
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/resepformer-wsj02mix",
            savedir=os.path.join(cache_dir, "resepformer"),
            run_opts={"device": self.device}
        )
        self.whisper_model = whisper.load_model(self.config.whisper_model_size, download_root=os.path.join(cache_dir, "whisper")).to(self.device)
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.config.auth_token,
            cache_dir=os.path.join(cache_dir, "speaker-diarization")
        ).to(self.device)
        self.vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=self.config.auth_token,
            cache_dir=os.path.join(cache_dir, "vad")
        ).to(self.device)
        self.embedding_model = Inference(
            "pyannote/embedding",
            window="whole",
            use_auth_token=self.config.auth_token,
        ).to(self.device)
        logging.info("Models initialized successfully!")

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
        # Log the raw times for debugging
        audio_duration = audio.shape[-1] / sr
        logging.debug(f"Extracting segment: start={start:.2f}s, end={end:.2f}s, audio duration={audio_duration:.2f}s")
        
        # Ensure start and end are within audio duration bounds
        if start < 0:
            logging.warning(f"Start time {start:.2f}s is negative; setting to 0.")
            start = 0.0
        if end > audio_duration:
            logging.warning(f"End time {end:.2f}s exceeds audio duration ({audio_duration:.2f}s); clipping to audio duration.")
            end = audio_duration

        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        # Additional check: if indices are invalid, log detailed info and return a small non-empty tensor
        if start_idx >= end_idx:
            logging.warning(f"Invalid segment indices: {start_idx}-{end_idx} for audio length {audio.shape[-1]} "
                            f"(start_time={start:.2f}s, end_time={end:.2f}s, sample_rate={sr}).")
            # Return a small tensor of zeros (or consider skipping the segment)
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

    def process_file(self, file_path: str) -> Dict:
        try:
            audio, sample_rate = self.load_audio(file_path)
            audio_duration = audio.shape[-1] / sample_rate
            logging.info(f"Processing audio file: {audio_duration:.2f} seconds")
            logging.info("Running Voice Activity Detection...")
            vad_result = self.vad_pipeline(file_path)
            vad_intervals = get_pyannote_vad_intervals(vad_result)
            logging.info(f"VAD detected {len(vad_intervals)} speech intervals")
            logging.info("Running full-audio Speaker Diarization for profile building...")
            diarization_result = self.diarization(
                file_path,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            raw_segments = [(segment.start, segment.end, speaker)
                            for segment, _, speaker in diarization_result.itertracks(yield_label=True)]
            logging.info(f"Diarization found {len(raw_segments)} raw segments")
            merged_segments = merge_diarization_segments(raw_segments, self.config.merge_gap_threshold)
            logging.info(f"After merging: {len(merged_segments)} segments")
            refined_segments = []
            if self.config.use_vad_refinement:
                for start, end, speaker in merged_segments:
                    refined = refine_segment_with_vad((start, end), vad_intervals)
                    if refined and (refined[1] - refined[0] >= self.config.min_segment_duration):
                        refined_segments.append((refined[0], refined[1], speaker))
                logging.info(f"After VAD refinement: {len(refined_segments)} segments")
            else:
                refined_segments = merged_segments
            speaker_embeddings = self._build_speaker_profiles(audio, diarization_result)
            logging.info(f"Created embeddings for {len(speaker_embeddings)} speakers")
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
            overlap_regions = self._detect_overlap_regions(diarization_result)
            processed_segments = []
            meta_counts = {'SPEAKER_A': 0, 'SPEAKER_B': 0}
            for seg_start, seg_end, orig_speaker in refined_segments:
                is_overlap = False
                involved_speakers = []
                for ov_start, ov_end, speakers in overlap_regions:
                    if max(seg_start, ov_start) < min(seg_end, ov_end):
                        is_overlap = True
                        involved_speakers = speakers
                        break
                if (seg_end - seg_start) < self.config.min_segment_duration:
                    continue
                audio_segment = self._extract_segment(audio, seg_start, seg_end)
                spk_label = speaker_mapping.get(orig_speaker, "UNKNOWN")
                if not is_overlap:
                    embedding = self._extract_embedding(audio_segment)
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
            processed_segments.sort(key=lambda x: x.start)
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
            # In temporary mode, use the provided output_dir (which should be task-specific)
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Processing file: {input_file}")
            
            if progress_callback:
                progress_callback(30, "Running file processing")
            results = self.process_file(input_file)
            
            if progress_callback:
                progress_callback(60, "Saving processed segments")
            self.save_segments(results['segments'], output_dir)
            if debug_mode:
                self.save_debug_segments(results['segments'], output_dir)
            if progress_callback:
                progress_callback(80, "Saving transcript")
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

# =============================================================================
# FastAPI App and Endpoints
# =============================================================================

app = FastAPI(
    title="Enhanced Audio Transcription API",
    description="API to process audio files (MP3/WAV) and return a formatted transcript. The transcript can also be downloaded as a TXT file."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global output directory – but each task will have its own subfolder.
OUTPUT_DIR = "processed_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
config = Config(auth_token=AUTH_TOKEN)
processor = EnhancedAudioProcessor(config)

# Global dictionaries to track progress and results
progress_store: Dict[str, Dict] = {}
result_store: Dict[str, Dict] = {}

CLEANUP_THRESHOLD = timedelta(minutes=30)

def cleanup_old_tasks():
    """Delete task folders in OUTPUT_DIR older than CLEANUP_THRESHOLD."""
    now = datetime.now()
    for task_folder in Path(OUTPUT_DIR).iterdir():
        if task_folder.is_dir():
            # Get folder's last modified time
            folder_mtime = datetime.fromtimestamp(task_folder.stat().st_mtime)
            if now - folder_mtime > CLEANUP_THRESHOLD:
                logging.info(f"Cleaning up old task folder: {task_folder}")
                shutil.rmtree(task_folder)

# Start APScheduler when the app starts
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_tasks, 'interval', minutes=10)  # run every 10 minutes
scheduler.start()

# Make sure to shut down the scheduler when the app exits.
import atexit
atexit.register(lambda: scheduler.shutdown())

def update_progress(task_id: str, percent: int, message: str):
    progress_store[task_id] = {"progress": percent, "message": message}
    logging.info(f"Task {task_id}: {percent}% - {message}")

async def process_audio_with_progress(task_id: str, file_path: str):
    try:
        task_output_dir = os.path.join(OUTPUT_DIR, task_id)
        processor.run(
            file_path, 
            output_dir=task_output_dir, 
            debug_mode=False, 
            progress_callback=lambda p, m: update_progress(task_id, p, m)
        )
        result_store[task_id] = {"download_url": f"/download/{task_id}/transcript.txt"}
    except Exception as e:
        update_progress(task_id, 100, f"Error: {str(e)}")
        result_store[task_id] = {"error": str(e)}
    finally:
        # Ensure the temporary file is removed after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed temporary file: {file_path}")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP3 and WAV files are accepted.")
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / file.filename
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        task_id = str(uuid.uuid4())
        update_progress(task_id, 0, "Task queued")
        background_tasks.add_task(process_audio_with_progress, task_id, str(temp_file_path))
        return JSONResponse(content={"task_id": task_id})
    except Exception as e:
        logging.error(f"Error in /transcribe endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/progress/{task_id}")
async def progress_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            data = progress_store.get(task_id, {"progress": 0, "message": "Waiting..."})
            await websocket.send_json(data)
            if data.get("progress", 0) >= 100:
                break
            await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        await websocket.close()

@app.get("/download/{file_path:path}")
async def download_transcript(file_path: str):
    transcript_path = Path(OUTPUT_DIR) / file_path
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")
    return FileResponse(path=str(transcript_path), media_type="text/plain", filename=transcript_path.name)

# New cleanup endpoint to remove the temporary processed_audio folder for a given task.
@app.delete("/cleanup/{task_id}")
async def cleanup(task_id: str):
    task_folder = Path(OUTPUT_DIR) / task_id
    if task_folder.exists() and task_folder.is_dir():
        shutil.rmtree(task_folder)
        return JSONResponse(content={"status": f"Cleaned up files for task {task_id}"})
    else:
        raise HTTPException(status_code=404, detail="Task folder not found.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
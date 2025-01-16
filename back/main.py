from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioProcessor:
    def __init__(self):
        try:
            # Initialize RE-SepFormer model
            self.separator = separator.from_hparams(
                source="speechbrain/resepformer-wsj02mix",
                savedir="pretrained_models/resepformer-wsj02mix"
            )

            # Initialize Whisper model and processor
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

            # Move Whisper model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model.to(self.device)

            # Define sample rates
            self.separator_sample_rate = 8000
            self.whisper_sample_rate = 16000
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    async def process_audio(self, file: UploadFile) -> dict:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Create proper temp file path
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            print(f"Processing file: {temp_file_path}")

            # Load audio file using torchaudio first
            waveform, sample_rate = torchaudio.load(temp_file_path)
            
            # Resample if necessary
            if sample_rate != self.separator_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.separator_sample_rate
                )
                waveform = resampler(waveform)
            
            # Save resampled audio
            resampled_path = os.path.join(temp_dir, "resampled.wav")
            torchaudio.save(resampled_path, waveform, self.separator_sample_rate)

            # Separate audio sources
            est_sources = self.separator.separate_file(path=resampled_path)
            
            # Convert to numpy arrays
            source1 = est_sources[:, :, 0].cpu().numpy()
            source2 = est_sources[:, :, 1].cpu().numpy()

            # Transcribe both sources
            transcription1 = self._transcribe_audio(source1)
            transcription2 = self._transcribe_audio(source2)

            return {
                "source1_transcription": transcription1,
                "source2_transcription": transcription2
            }
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _transcribe_audio(self, audio_array: np.ndarray) -> str:
        try:
            # Resample if necessary
            if self.separator_sample_rate != self.whisper_sample_rate:
                audio_tensor = torch.tensor(audio_array)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.separator_sample_rate,
                    new_freq=self.whisper_sample_rate
                )
                audio_array = resampler(audio_tensor).numpy()

            # Ensure correct format
            if len(audio_array.shape) == 1:
                audio_array = audio_array.reshape(1, -1)

            # Process audio
            input_features = self.whisper_processor(
                audio_array[0],
                sampling_rate=self.whisper_sample_rate,
                return_tensors="pt"
            ).input_features

            # Generate transcription
            predicted_ids = self.whisper_model.generate(input_features.to(self.device))
            transcription = self.whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            raise

# Initialize audio processor
audio_processor = AudioProcessor()

@app.get("/")
async def read_root():
    return {"message": "API is working"}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type in ["audio/mpeg", "audio/wav", "audio/mp3"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Please upload an MP3 or WAV file."
            )
        
        print(f"Received file: {file.filename} ({file.content_type})")
        result = await audio_processor.process_audio(file)
        return result
    except Exception as e:
        print(f"Error in process_audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug"
    )
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import tempfile
import os
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioProcessor:
    def __init__(self):
        # Initialize RE-SepFormer model
        self.separator = separator.from_hparams(
            source="speechbrain/resepformer-wsj02mix",
            savedir="pretrained_models/resepformer-wsj02mix"
        )

        # Initialize Whisper model and processor
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        # Move Whisper model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.to(self.device)

        # Define sample rates
        self.separator_sample_rate = 8000
        self.whisper_sample_rate = 16000

    async def process_audio_file(self, file: UploadFile):
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Separate the audio
            est_sources = self.separator.separate_file(path=temp_path)
            
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

        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def _resample_audio(self, audio_array, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        audio_tensor = torch.tensor(audio_array)
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        resampled_audio = resampler(audio_tensor)
        return resampled_audio.numpy()

    def _transcribe_audio(self, audio_array, sample_rate=8000):
        """Transcribe audio using Whisper"""
        # Resample if necessary
        if sample_rate != self.whisper_sample_rate:
            audio_array = self._resample_audio(
                audio_array,
                orig_sr=sample_rate,
                target_sr=self.whisper_sample_rate
            )

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

# Initialize the audio processor
audio_processor = AudioProcessor()

@app.post("/process-audio")
async def process_audio(file: UploadFile):
    if not file.filename.endswith(('.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="File must be an MP3 or WAV audio file")
    
    try:
        result = await audio_processor.process_audio_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
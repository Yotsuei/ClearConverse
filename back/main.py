from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import tempfile
import os
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL
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
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        # Move Whisper model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.to(self.device)

        # Define sample rates
        self.separator_sample_rate = 8000
        self.whisper_sample_rate = 16000

    async def process_audio(self, file: UploadFile) -> dict:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Separate audio sources
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
            # Clean up temporary file
            os.unlink(temp_path)

    def _transcribe_audio(self, audio_array: np.ndarray) -> str:
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

# Initialize audio processor
audio_processor = AudioProcessor()

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        result = await audio_processor.process_audio(file)
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
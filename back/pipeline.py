import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.pretrained import SepformerSeparation as separator

class SpeechProcessor:
    def __init__(self):
        # Initialize RE-SepFormer model
        self.separator = separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix"
        )
        
        # Initialize Whisper model and processor
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model.to(self.device)

    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure audio is mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if necessary (Whisper's expected sample rate)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        return waveform

    def separate_speeches(self, waveform):
        """Separate overlapping speeches using RE-SepFormer."""
        # RE-SepFormer expects specific input format
        est_sources = self.separator.separate_batch(waveform)
        # Returns tensor of shape (batch, n_sources, time)
        return est_sources

    def transcribe_audio(self, waveform):
        """Transcribe audio using Whisper."""
        # Convert audio to numpy array
        input_features = self.whisper_processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        # Move input to same device as model
        input_features = input_features.to(self.device)
        
        # Generate transcription
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription

    def process_audio(self, audio_path):
        """Complete pipeline for processing audio file."""
        try:
            # Load audio
            waveform = self.load_audio(audio_path)
            
            # Separate speeches
            separated_sources = self.separate_speeches(waveform)
            
            # Transcribe each separated source
            transcriptions = []
            for source in separated_sources:
                # Ensure source is in correct format for Whisper
                source = source.unsqueeze(0)  # Add batch dimension
                transcription = self.transcribe_audio(source)
                transcriptions.append(transcription)
            
            return {
                'status': 'success',
                'transcriptions': transcriptions,
                'num_speakers': len(transcriptions)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

# Example usage
if __name__ == "__main__":
    processor = SpeechProcessor()
    result = processor.process_audio("path_to_your_audio.wav")
    
    if result['status'] == 'success':
        print(f"Number of speakers detected: {result['num_speakers']}")
        for i, transcription in enumerate(result['transcriptions'], 1):
            print(f"\nSpeaker {i} transcription:")
            print(transcription)
    else:
        print(f"Error: {result['message']}")
// App.tsx
import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import AudioRecorder from './components/AudioRecorder';
import AudioPlayer from './components/AudioPlayer';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import { Tab } from './components/Tab';

type AudioSource = {
  file: File | null;
  url: string | null;
};

type Module = 'upload' | 'record';

const App: React.FC = () => {
  const [activeModule, setActiveModule] = useState<Module>('upload');
  const [audioSource, setAudioSource] = useState<AudioSource>({ file: null, url: null });
  const [transcript, setTranscript] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);

  const handleUploadResponse = (transcript: string, downloadUrl: string) => {
    setTranscript(transcript);
    setDownloadUrl(downloadUrl);
    setIsProcessing(false);
    setProcessingProgress(100);
  };

  const handleFileSelected = (file: File) => {
    const url = URL.createObjectURL(file);
    setAudioSource({ file, url });
  };

  const handleRecordingComplete = (blob: Blob) => {
    const file = new File([blob], "recording.wav", { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    setAudioSource({ file, url });
  };

  const resetState = () => {
    setAudioSource({ file: null, url: null });
    setTranscript(null);
    setDownloadUrl(null);
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
  };

  // Simulate processing progress
  const startProcessingSimulation = () => {
    setIsProcessing(true);
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      setProcessingProgress(Math.min(progress, 95)); // Max at 95% to indicate waiting for server
      if (progress >= 95) {
        clearInterval(interval);
      }
    }, 500);
  };

  return (
    <div className="min-h-screen w-full bg-gray-50 flex flex-col items-center justify-center p-6 text-gray-800">
      <div className="w-full max-w-3xl bg-white shadow-xl rounded-xl p-8 border border-gray-200">
        <h1 className="text-4xl font-extrabold text-center mb-8 tracking-tight text-gray-900">
          Overlapping Speech Transcription
        </h1>
        
        {/* Module Selection Tabs */}
        <div className="flex rounded-lg mb-6 overflow-hidden">
          <Tab 
            active={activeModule === 'upload'} 
            onClick={() => setActiveModule('upload')}
            label="Upload File"
            icon="📁"
          />
          <Tab 
            active={activeModule === 'record'} 
            onClick={() => setActiveModule('record')}
            label="Record Audio"
            icon="🎙️"
          />
        </div>

        {/* Module Content */}
        <div className="bg-gray-50 rounded-lg p-6 mb-6 border border-gray-200">
          {activeModule === 'upload' ? (
            <FileUpload 
              onFileSelected={handleFileSelected}
              onUploadResponse={handleUploadResponse}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              setIsProcessing={setIsProcessing}
              startProcessing={startProcessingSimulation}
            />
          ) : (
            <AudioRecorder onRecordingComplete={handleRecordingComplete} />
          )}
        </div>

        {/* Audio Player */}
        {audioSource.url && (
          <div className="mt-6">
            <AudioPlayer audioUrl={audioSource.url} />
          </div>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <div className="mt-6">
            <p className="text-gray-700 mb-2 font-medium">Uploading...</p>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Processing Progress */}
        {isProcessing && (
          <div className="mt-6">
            <p className="text-gray-700 mb-2 font-medium">Processing audio...</p>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-green-500 h-2.5 rounded-full transition-all duration-300" 
                style={{ width: `${processingProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Transcription Display */}
        {transcript && downloadUrl && (
          <TranscriptionDisplay 
            transcript={transcript} 
            downloadUrl={downloadUrl}
            onClear={resetState}
          />
        )}
      </div>
      
      <footer className="mt-8 text-center text-gray-500 text-sm">
        © 2025 Speech Transcription Tool
      </footer>
    </div>
  );
};

export default App;
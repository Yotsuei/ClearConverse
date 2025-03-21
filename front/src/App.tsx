// App.tsx
import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import AudioRecorder from './components/AudioRecorder';
import AudioPlayer from './components/AudioPlayer';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import { Tab } from './components/Tab';
import MainMenu from './components/MainMenu';
import UrlUpload from './components/UrlUpload';
import ProgressBar from './components/ProgressBar';
import './index.css';

type AudioSource = {
  file: File | null;
  url: string | null;
};

type Module = 'upload' | 'record' | 'url';

const App: React.FC = () => {
  const [showMainMenu, setShowMainMenu] = useState<boolean>(true);
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
  
  const clearTranscription = () => {
    setTranscript(null);
    setDownloadUrl(null);
    setIsProcessing(false);
    setProcessingProgress(0);
  };
  
  const handleModuleSelect = (module: Module) => {
    setActiveModule(module);
    setShowMainMenu(false);
    resetState();
  };
  
  const goToMainMenu = () => {
    resetState();
    setShowMainMenu(true);
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

  // Handle transcription from either audio player or recorder
  const handleTranscribe = () => {
    if (!audioSource.file) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', audioSource.file);

    try {
      // Implement XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      xhr.open('POST', 'http://localhost:8000/transcribe');
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
        }
      });

      xhr.onload = function() {
        if (xhr.status === 200) {
          setIsUploading(false);
          startProcessingSimulation();
          
          // Parse response
          const response = JSON.parse(xhr.responseText);
          handleUploadResponse(response.transcript, response.download_url);
        } else {
          throw new Error(`Error: ${xhr.statusText}`);
        }
      };

      xhr.onerror = function() {
        throw new Error('Network error occurred');
      };

      xhr.send(formData);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('There was an error uploading your file.');
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col items-center justify-center p-6 text-gray-200">
      {showMainMenu ? (
        <MainMenu onSelectModule={handleModuleSelect} />
      ) : (
        <div className="w-full max-w-3xl bg-gray-800 shadow-xl rounded-xl p-8 border border-gray-700">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-extrabold tracking-tight text-gray-100">
              Speech Transcription
            </h1>
            <button 
              onClick={goToMainMenu}
              className="text-blue-400 hover:text-blue-300 flex items-center gap-1 text-sm font-medium"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
              </svg>
              Back to Main Menu
            </button>
          </div>
          
          {/* Module Selection Tabs */}
          <div className="flex rounded-lg mb-6 overflow-hidden border border-gray-700">
            <Tab 
              active={activeModule === 'upload'} 
              onClick={() => {
                setActiveModule('upload');
                resetState();
              }}
              label="Upload File"
              icon="📁"
            />
            <Tab 
              active={activeModule === 'url'} 
              onClick={() => {
                setActiveModule('url');
                resetState();
              }}
              label="Google Drive"
              icon="🔗"
            />
            <Tab 
              active={activeModule === 'record'} 
              onClick={() => {
                setActiveModule('record');
                resetState();
              }}
              label="Record Audio"
              icon="🎙️"
            />
          </div>

          {/* Module Content */}
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            {activeModule === 'upload' && (
              <FileUpload 
                onFileSelected={handleFileSelected}
                onUploadResponse={handleUploadResponse}
                setIsUploading={setIsUploading}
                setUploadProgress={setUploadProgress}
                setIsProcessing={setIsProcessing}
                startProcessing={startProcessingSimulation}
                clearTranscription={clearTranscription}
              />
            )}
            
            {activeModule === 'url' && (
              <UrlUpload
                onFileSelected={handleFileSelected}
                onUploadResponse={handleUploadResponse}
                setIsUploading={setIsUploading}
                setUploadProgress={setUploadProgress}
                setIsProcessing={setIsProcessing}
                startProcessing={startProcessingSimulation}
                clearTranscription={clearTranscription}
              />
            )}
            
            {activeModule === 'record' && (
              <AudioRecorder 
                onRecordingComplete={handleRecordingComplete} 
                onTranscribe={audioSource.file ? handleTranscribe : undefined}
              />
            )}
          </div>

        {/* Audio Player */}
        {audioSource.url && (
          <div className="mt-6">
            <AudioPlayer 
              audioUrl={audioSource.url} 
              onTranscribe={activeModule !== 'record' ? handleTranscribe : undefined}
            />
          </div>
        )}

        {/* Progress Bars */}
        {isUploading && (
          <div className="mt-6">
            <ProgressBar progress={uploadProgress} type="upload" />
          </div>
        )}

        {isProcessing && (
          <div className="mt-6">
            <ProgressBar progress={processingProgress} type="processing" />
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
      )}
      
      <footer className="mt-8 text-center text-gray-500 text-sm">
        © 2025 Speech Transcription Tool
      </footer>
    </div>
  );
};

export default App;
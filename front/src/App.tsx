// App.tsx
import React, { useState, useRef } from 'react';
import FileUpload from './components/FileUpload';
import UrlUpload from './components/UrlUpload';
import AudioPlayer from './components/AudioPlayer';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import MainMenu from './components/MainMenu';
import ProgressBar from './components/ProgressBar';
import ResetButton from './components/ResetButton';
import ClearButton from './components/ClearButton';
import './index.css';

type AudioSource = {
  file: File | null;
  url: string | null;
};

type Module = 'upload' | 'url';

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
  
  // Add XHR reference to allow cancellation of in-progress requests
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  // Add interval reference to clear processing simulation
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const handleUploadResponse = (transcript: string, downloadUrl: string) => {
    setTranscript(transcript);
    setDownloadUrl(downloadUrl);
    setIsProcessing(false);
    setProcessingProgress(100);
  };

  const handleFileSelected = (file: File) => {
    // If it's an empty file (used for clearing), don't create Object URL
    if (file.size === 0) {
      setAudioSource({ file: null, url: null });
      return;
    }
    
    const url = URL.createObjectURL(file);
    setAudioSource({ file, url });
  };

  const resetState = () => {
    // Cancel any ongoing requests first
    cancelTranscription();
    
    // Reset all states
    setAudioSource({ file: null, url: null });
    setTranscript(null);
    setDownloadUrl(null);
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
    
    // Also clear the file input if it exists
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };
  
  const clearTranscription = () => {
    // Only clear the transcription result, keep the audio file
    setTranscript(null);
    setDownloadUrl(null);
    setIsProcessing(false);
    setProcessingProgress(0);
  };
  
  const clearAll = () => {
    // Clear everything including audio and transcription
    resetState();
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
    
    // Clear any existing interval
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
    }
    
    // Start new interval
    const interval = setInterval(() => {
      progress += 5;
      setProcessingProgress(Math.min(progress, 95)); // Max at 95% to indicate waiting for server
      if (progress >= 95) {
        clearInterval(interval);
        processingIntervalRef.current = null;
      }
    }, 500);
    
    processingIntervalRef.current = interval;
  };

  // Cancel transcription function
  const cancelTranscription = () => {
    // Cancel any ongoing XHR request
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    
    // Clear processing simulation
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    
    // Reset progress states
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
  };

  // Handle transcription from either audio player or recorder
  const handleTranscribe = () => {
    if (!audioSource.file) return;
    
    // Log the file details for debugging
    console.log(`Sending file: ${audioSource.file.name}, type: ${audioSource.file.type}, size: ${audioSource.file.size} bytes`);
    
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', audioSource.file);

    try {
      // Implement XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      // Store XHR reference for cancellation
      xhrRef.current = xhr;
      
      xhr.open('POST', 'http://localhost:8000/transcribe');
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
        }
      });

      xhr.onload = function() {
        xhrRef.current = null;
        if (xhr.status === 200) {
          setIsUploading(false);
          startProcessingSimulation();
          
          // Parse response
          const response = JSON.parse(xhr.responseText);
          handleUploadResponse(response.transcript, response.download_url);
        } else {
          console.error(`Server error: ${xhr.status} ${xhr.statusText}`);
          console.error(`Response: ${xhr.responseText}`);
          setIsUploading(false);
          alert(`Error: ${xhr.statusText || 'Server error'}\n${xhr.responseText || ''}`);
        }
      };

      xhr.onerror = function() {
        xhrRef.current = null;
        setIsUploading(false);
        console.error('Network error occurred');
        alert('Network error occurred. Please check your connection and try again.');
      };

      xhr.onabort = function() {
        console.log('Transcription request aborted');
        setIsUploading(false);
      };

      xhr.send(formData);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('There was an error uploading your file.');
      setIsUploading(false);
      xhrRef.current = null;
    }
  };

  // Render different content based on the active module
  const renderModuleContent = () => {
    if (activeModule === 'upload') {
      return (
        <>
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            <FileUpload 
              onFileSelected={handleFileSelected}
              onUploadResponse={handleUploadResponse}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              setIsProcessing={setIsProcessing}
              startProcessing={startProcessingSimulation}
              clearTranscription={clearTranscription}
            />
          </div>

          {/* Audio Player - always show if audio is available */}
          {audioSource.url && (
            <div className="mt-6">
              <AudioPlayer 
                audioUrl={audioSource.url} 
                onTranscribe={handleTranscribe}
              />
            </div>
          )}
        </>
      );
    } else if (activeModule === 'url') {
      return (
        <>
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            <UrlUpload 
              onFileSelected={handleFileSelected}
              onUploadResponse={handleUploadResponse}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              setIsProcessing={setIsProcessing}
              startProcessing={startProcessingSimulation}
              clearTranscription={clearTranscription}
            />
          </div>

          {/* Audio Player - always show if audio is available and when the actual URL is available for URL uploads */}
          {audioSource.url && (
            <div className="mt-6">
              <AudioPlayer 
                audioUrl={audioSource.url} 
                onTranscribe={transcript ? undefined : handleTranscribe} // Only show transcribe button if no transcript yet
              />
            </div>
          )}
        </>
      );
    }
    
    return null;
  };

  // Determine if we should show the "Clear" button
  const shouldShowClearButton = audioSource.file && transcript;

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col items-center justify-center p-6 text-gray-200">
      {showMainMenu ? (
        <MainMenu onSelectModule={handleModuleSelect} />
      ) : (
        <div className="w-full max-w-3xl bg-gray-800 shadow-xl rounded-xl p-8 border border-gray-700">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-extrabold tracking-tight text-gray-100">
              {activeModule === 'upload' ? 'File Upload Transcription' : 'URL Upload Transcription'}
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
          
          {/* Module Content */}
          {renderModuleContent()}

          {/* Progress Bars with Cancel Button */}
          {isUploading && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-200">Upload Progress</h3>
              </div>
              <ProgressBar progress={uploadProgress} type="upload" onCancel={cancelTranscription} />
            </div>
          )}

          {isProcessing && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-200">Processing Progress</h3>
              </div>
              <ProgressBar progress={processingProgress} type="processing" onCancel={cancelTranscription} />
            </div>
          )}

          {/* Reset and Clear Buttons (when audio is loaded but no transcription yet) */}
          {audioSource.file && !transcript && !isUploading && !isProcessing && (
            <div className="mt-6 flex gap-4">
              <button
                onClick={resetState}
                className="flex-1 py-3 px-4 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors flex justify-center items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
                Reset
              </button>
            </div>
          )}

          {/* Transcription Display */}
          {transcript && downloadUrl && (
            <TranscriptionDisplay 
              transcript={transcript} 
              downloadUrl={downloadUrl}
              onClear={clearTranscription}
              onReset={clearAll} // Changed to clearAll to reset everything
            />
          )}
          
          {/* Clear Transcription Button (when both audio and transcript exist) */}
          {shouldShowClearButton && !isUploading && !isProcessing && (
            <div className="mt-4 flex justify-end">
              <button
                onClick={clearAll} // Changed to clearAll to reset everything including audio
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                </svg>
                Clear Everything
              </button>
            </div>
          )}
        </div>
      )}
      
      {/* Floating Reset and Clear Buttons - only show when not on main menu */}
      {!showMainMenu && (audioSource.file || isUploading || isProcessing || transcript) && (
        <ResetButton 
          onReset={resetState}
          onClear={clearTranscription}
          isProcessing={isUploading || isProcessing}
          hasTranscription={Boolean(transcript)}
        />
      )}
      
      {/* Adding ClearButton as a separate floating button on the left side */}
      {!showMainMenu && audioSource.file && transcript && !isUploading && !isProcessing && (
        <div className="fixed bottom-6 left-6 z-10">
          <ClearButton 
            onClear={clearAll} // Changed to clearAll to reset everything including audio
            isProcessing={false}
            includeAudio={true} // Explicitly setting to true to indicate clearing everything
          />
        </div>
      )}
      
      <footer className="mt-8 text-center text-gray-500 text-sm">
        © 2025 Speech Transcription Tool
      </footer>
    </div>
  );
};

export default App;
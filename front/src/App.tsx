import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import UrlUpload from './components/UrlUpload';
import AudioPlayer from './components/AudioPlayer';
import ProgressBar from './components/ProgressBar';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import MainMenu from './components/MainMenu';
import ResetButton from './components/ResetButton';

const App: React.FC = () => {
  // Main app states
  const [appState, setAppState] = useState<'menu' | 'upload' | 'url'>('menu');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [showAudioPlayer, setShowAudioPlayer] = useState(false);

  // Handle module selection from main menu
  const handleModuleSelect = (module: 'upload' | 'url') => {
    setAppState(module);
    // Reset states when changing modules
    setSelectedFile(null);
    setTranscript('');
    setDownloadUrl('');
    setAudioUrl('');
    setShowAudioPlayer(false);
    setIsUploading(false);
    setIsProcessing(false);
    setUploadProgress(0);
  };

  // Handle file selection - automatically show the audio player
  const handleFileSelected = (file: File) => {
    // Reset transcription but keep other states
    setTranscript('');
    setDownloadUrl('');
    setSelectedFile(file);
    
    // Create a URL for audio preview - show audio player immediately
    if (file && file.size > 0) { // Only create URL for non-empty files
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setShowAudioPlayer(true);
    } else if (file) {
      // For empty files (likely from URL component), keep the audio player
      // but don't create a new object URL
      setShowAudioPlayer(true);
    } else {
      setAudioUrl('');
      setShowAudioPlayer(false);
    }
  };

  // Clean up object URLs on unmount or when file changes
  useEffect(() => {
    return () => {
      if (audioUrl && !audioUrl.startsWith('http')) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const handleUploadResponse = (transcriptText: string, url: string) => {
    console.log("Setting transcript:", transcriptText?.substring(0, 50) + "...");
    console.log("Setting download URL:", url);
    setTranscript(transcriptText || ""); // Ensure we always set a string
    setDownloadUrl(url || "");
    setIsProcessing(false);
    setIsUploading(false);
  };

  const startProcessing = () => {
    setIsProcessing(true);
    setIsUploading(false);
  };

  const handleCancelUpload = () => {
    setIsUploading(false);
    setIsProcessing(false);
    setUploadProgress(0);
  };

  const handleReset = () => {
    if (selectedFile && audioUrl && !audioUrl.startsWith('http')) {
      URL.revokeObjectURL(audioUrl);
    }
    // Return to main menu
    setAppState('menu');
    setSelectedFile(null);
    setTranscript('');
    setDownloadUrl('');
    setAudioUrl('');
    setIsUploading(false);
    setIsProcessing(false);
    setUploadProgress(0);
    setShowAudioPlayer(false);
  };

  const clearTranscription = () => {
    setTranscript('');
    setDownloadUrl('');
    setIsProcessing(false);
  };

  // Create navigation component
  const Navigation = () => (
    <div className="flex mb-6 border-b border-gray-700">
      <button 
        onClick={() => handleReset()}
        className="px-4 py-2 text-gray-400 hover:text-blue-400 transition-colors flex items-center gap-1"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
        </svg>
        Back to Menu
      </button>
      <div className="ml-auto flex">
        <span className="px-4 py-2 text-gray-300">
          {appState === 'upload' ? 'File Upload' : 'URL Upload'}
        </span>
      </div>
    </div>
  );

  // Debug function to check if transcript exists
  const debugTranscript = () => {
    console.log("Transcript exists:", !!transcript);
    console.log("Transcript length:", transcript?.length || 0);
    console.log("First 50 chars:", transcript?.substring(0, 50) || "empty");
    console.log("Download URL:", downloadUrl);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 py-12 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          Audio Transcription Tool
        </h1>
        
        {/* Main content area */}
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700">
          {/* Main Menu */}
          {appState === 'menu' && (
            <MainMenu onSelectModule={handleModuleSelect} />
          )}

          {/* Navigation for Upload/URL screens */}
          {appState !== 'menu' && !transcript && (
            <Navigation />
          )}
          
          {/* Loading indicators */}
          {isUploading && (
            <ProgressBar 
              progress={uploadProgress} 
              type="upload" 
              onCancel={handleCancelUpload} 
            />
          )}
          
          {isProcessing && (
            <ProgressBar 
              progress={uploadProgress} 
              type="processing" 
              onCancel={handleCancelUpload} 
            />
          )}
          
          {/* File uploader */}
          {appState === 'upload' && !isUploading && !isProcessing && !transcript && (
            <FileUpload 
              onFileSelected={handleFileSelected}
              onUploadResponse={handleUploadResponse}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              setIsProcessing={setIsProcessing}
              startProcessing={startProcessing}
              clearTranscription={clearTranscription}
            />
          )}
          
          {/* URL uploader */}
          {appState === 'url' && !isUploading && !isProcessing && !transcript && (
            <UrlUpload 
              onFileSelected={handleFileSelected}
              onUploadResponse={handleUploadResponse}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              setIsProcessing={setIsProcessing}
              startProcessing={startProcessing}
              clearTranscription={clearTranscription}
            />
          )}
          
          {/* Audio preview */}
          {showAudioPlayer && audioUrl && (
            <div className="mt-6">
              <AudioPlayer 
                audioUrl={audioUrl} 
                onTranscribe={() => {
                  // You can add functionality here if needed
                  console.log("Transcribe clicked from AudioPlayer");
                }}
              />
            </div>
          )}
          
          {/* Debug button in development */}
          {process.env.NODE_ENV === 'development' && (
            <button 
              onClick={debugTranscript}
              className="mt-4 px-3 py-1 bg-purple-800 text-white text-xs rounded"
            >
              Debug Transcript
            </button>
          )}
        </div>
        
        {/* Transcription display - outside the main content box */}
        {transcript && (
          <div className="mt-6 bg-gray-800 rounded-lg shadow-lg border border-gray-700">
            <TranscriptionDisplay 
              transcript={transcript} 
              downloadUrl={downloadUrl}
              onClear={clearTranscription}
              onReset={handleReset}
            />
          </div>
        )}
        
        {/* Reset button only shown when not on main menu */}
        {appState !== 'menu' && (
          <ResetButton 
            onReset={handleReset} 
            onClear={clearTranscription}
            isProcessing={isProcessing || isUploading}
            hasTranscription={!!transcript}
          />
        )}
      </div>
    </div>
  );
};

export default App;
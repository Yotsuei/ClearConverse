// App.tsx - Main component with auto-loading audio player
import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import UrlUpload from './components/UrlUpload';
import AudioPlayer from './components/AudioPlayer';
import ProgressBar from './components/ProgressBar';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import { Tab } from './components/Tab';
import ResetButton from './components/ResetButton';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'upload' | 'url'>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [showAudioPlayer, setShowAudioPlayer] = useState(false);

  // Handle file selection - automatically show the audio player
  const handleFileSelected = (file: File) => {
    // Reset states
    setTranscript('');
    setDownloadUrl('');
    setSelectedFile(file);
    
    // Create a URL for audio preview - show audio player immediately
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setShowAudioPlayer(true);
    } else {
      setAudioUrl('');
      setShowAudioPlayer(false);
    }
  };

  // Clean up object URLs on unmount or when file changes
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const handleUploadResponse = (transcriptText: string, url: string) => {
    setTranscript(transcriptText);
    setDownloadUrl(url);
    setIsProcessing(false);
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
    if (selectedFile) {
      URL.revokeObjectURL(audioUrl);
    }
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
  
  // Determine if we should show the uploader
  const showUploader = !isUploading && !isProcessing && !transcript;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 py-12 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          Audio Transcription Tool
        </h1>
        
        {/* Tab selection */}
        {showUploader && (
          <div className="flex mb-6 border-b border-gray-700">
            <Tab 
              active={activeTab === 'upload'} 
              onClick={() => setActiveTab('upload')} 
              label="Upload File" 
              icon="📂" 
            />
            <Tab 
              active={activeTab === 'url'} 
              onClick={() => setActiveTab('url')} 
              label="From URL" 
              icon="🔗" 
            />
          </div>
        )}
        
        {/* Main content area */}
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700">
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
          {showUploader && activeTab === 'upload' && (
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
          {showUploader && activeTab === 'url' && (
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
                }}
              />
            </div>
          )}
        </div>
        
        {/* Transcription display */}
        {transcript && (
          <TranscriptionDisplay 
            transcript={transcript} 
            downloadUrl={downloadUrl}
            onClear={clearTranscription}
            onReset={handleReset}
          />
        )}
        
        {/* Reset button */}
        <ResetButton 
          onReset={handleReset} 
          onClear={clearTranscription}
          isProcessing={isProcessing || isUploading}
          hasTranscription={!!transcript}
        />
      </div>
    </div>
  );
};

export default App;
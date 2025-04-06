// App.tsx
import React, { useState, useRef, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import UrlUpload from './components/UrlUpload';
import AudioPlayer from './components/AudioPlayer';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import MainMenu from './components/MainMenu';
import ProgressBar from './components/ProgressBar';
import ResetButton from './components/ResetButton';
import ClearButton from './components/ClearButton';
import WebSocketProgressHandler from './components/WebSocketProgressHandler';
import './index.css';

type AudioSource = {
  previewUrl: string | null; // Changed from url to previewUrl
};

type Module = 'upload' | 'url';

const App: React.FC = () => {
  const [showMainMenu, setShowMainMenu] = useState<boolean>(true);
  const [activeModule, setActiveModule] = useState<Module>('upload');
  const [audioSource, setAudioSource] = useState<AudioSource>({ previewUrl: null });
  const [transcript, setTranscript] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [processingMessage, setProcessingMessage] = useState<string>('Processing...');
  
  // Add XHR reference to allow cancellation of in-progress requests
  const xhrRef = useRef<XMLHttpRequest | null>(null);

  // Effect to fetch transcription once processing is complete
  useEffect(() => {
    if (processingProgress === 100 && taskId) {
      fetchTranscription(taskId);
    }
  }, [processingProgress, taskId]);

  const fetchTranscription = async (taskId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/transcription/${taskId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch transcription: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.transcription) {
        setTranscript(data.transcription);
        setDownloadUrl(`/download/${taskId}/transcript.txt`);
      } else {
        throw new Error('No transcription data returned');
      }
    } catch (error) {
      console.error('Error fetching transcription:', error);
      alert(`Failed to get transcription: ${(error as Error).message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProgressUpdate = (progress: number, message: string) => {
    setProcessingProgress(progress);
    setProcessingMessage(message);
  };

  const handleProcessingComplete = (downloadUrl: string) => {
    if (!downloadUrl) {
      setDownloadUrl(downloadUrl);
    }
  };

  // Modified to handle backend preview URL
  const handleUploadSuccess = (previewUrl: string, newTaskId: string) => {
    setAudioSource({ previewUrl: `http://localhost:8000${previewUrl}` });
    setTaskId(newTaskId);
    setIsUploading(false);
  };

  const resetState = () => {
    cancelTranscription();
    
    // Reset all states
    setAudioSource({ previewUrl: null });
    setTranscript(null);
    setDownloadUrl(null);
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
    setTaskId(null);
    setProcessingMessage('Processing...');
  };
  
  const clearTranscription = () => {
    setTranscript(null);
    setDownloadUrl(null);
    setIsProcessing(false);
    setProcessingProgress(0);
  };
  
  const clearAll = () => {
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

  const cancelTranscription = () => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);

    if (taskId) {
      fetch(`http://localhost:8000/cleanup/${taskId}`, { method: 'DELETE' })
        .then(response => {
          if (!response.ok) {
            console.error('Failed to cleanup task on server');
          }
        })
        .catch(err => {
          console.error('Error cleaning up task:', err);
        });
    }
  };

  const startTranscription = async () => {
    if (!taskId) {
      alert('No task ID available. Please upload a file or URL first.');
      return;
    }
    
    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingMessage('Starting transcription...');

    try {
      const response = await fetch(`http://localhost:8000/transcribe/${taskId}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Transcription started:', data);
    } catch (error) {
      console.error('Error starting transcription:', error);
      alert(`Failed to start transcription: ${(error as Error).message}`);
      setIsProcessing(false);
    }
  };

  const handleTranscribe = () => {
    if (!taskId) {
      alert('Please upload a file or URL first.');
      return;
    }
    
    startTranscription();
  };

  const renderModuleContent = () => {
    if (activeModule === 'upload') {
      return (
        <>
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            <FileUpload 
              setTaskId={setTaskId}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              clearTranscription={clearTranscription}
              onUploadSuccess={(previewUrl, taskId) => handleUploadSuccess(previewUrl, taskId)}
            />
          </div>

          {/* Audio Preview and Transcribe Button Section */}
          {audioSource.previewUrl && (
            <div className="mt-6 space-y-4">
              <AudioPlayer audioUrl={audioSource.previewUrl} />
              
              {/* Simplified conditional */}
              {taskId && !transcript && (
                <button
                  onClick={handleTranscribe}
                  className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
                >
                  Start Transcription
                </button>
              )}
            </div>
          )}
        </>
      );
    } else if (activeModule === 'url') {
      return (
        <>
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            <UrlUpload 
              setTaskId={setTaskId}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              clearTranscription={clearTranscription}
              onUploadSuccess={handleUploadSuccess}
            />
          </div>

          {/* Audio Preview and Transcribe Button */}
          {audioSource.previewUrl && (
            <div className="mt-6 space-y-4">
              <AudioPlayer audioUrl={audioSource.previewUrl} />
              
              {taskId && !transcript && (
                <button
                  onClick={handleTranscribe}
                  className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
                >
                  Start Transcription
                </button>
              )}
            </div>
          )}
        </>
      );
    }
    
    return null;
  };

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col items-center justify-center p-6 text-gray-200">
      {taskId && isProcessing && (
        <WebSocketProgressHandler 
          taskId={taskId} 
          onProgressUpdate={handleProgressUpdate}
          onComplete={handleProcessingComplete}
        />
      )}
    
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
          
          {renderModuleContent()}

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
              <ProgressBar 
                progress={processingProgress} 
                type="processing" 
                onCancel={cancelTranscription}
                message={processingMessage}
              />
              <p className="text-sm text-gray-400 mt-2">{processingMessage}</p>
            </div>
          )}

          {audioSource.previewUrl && !transcript && !isUploading && !isProcessing && (
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

          {transcript && downloadUrl && (
            <TranscriptionDisplay 
              transcript={transcript} 
              downloadUrl={downloadUrl}
              onClear={clearTranscription}
              onReset={clearAll}
            />
          )}
        </div>
      )}
      
      {!showMainMenu && (audioSource.previewUrl || isUploading || isProcessing || transcript) && (
        <ResetButton 
          onReset={resetState}
          onClear={clearTranscription}
          isProcessing={isUploading || isProcessing}
          hasTranscription={Boolean(transcript)}
        />
      )}
      
      {!showMainMenu && audioSource.previewUrl && transcript && !isUploading && !isProcessing && (
        <div className="fixed bottom-6 left-6 z-10">
          <ClearButton 
            onClear={clearAll}
            isProcessing={false}
            includeAudio={true}
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
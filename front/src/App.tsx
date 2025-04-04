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
import './index.css';

type AudioSource = {
  file: File | null;
  url: string | null;
};

type Module = 'upload' | 'url';

const App: React.FC = () => {
  // State for module selection, audio file, transcription, etc.
  const [showMainMenu, setShowMainMenu] = useState<boolean>(true);
  const [activeModule, setActiveModule] = useState<Module>('upload');
  const [audioSource, setAudioSource] = useState<AudioSource>({ file: null, url: null });
  const [transcript, setTranscript] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  // References for cancellation and polling
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Handle file selection
  const handleFileSelected = (file: File) => {
    // If it's an empty file (used for clearing), don't create Object URL
    if (file.size === 0) {
      setAudioSource({ file: null, url: null });
      return;
    }
    
    const url = URL.createObjectURL(file);
    setAudioSource({ file, url });
  };

  // Reset the entire state (file, transcript, progress, etc.)
  const resetState = () => {
    cancelTranscription();
    setAudioSource({ file: null, url: null });
    setTranscript(null);
    setDownloadUrl(null);
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
    setErrorMessage(null);
    setTaskId(null);
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  // Clear only the transcription, keeping the audio file
  const clearTranscription = () => {
    setTranscript(null);
    setDownloadUrl(null);
    setIsProcessing(false);
    setProcessingProgress(0);
    setErrorMessage(null);
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

  // Cancel any ongoing transcription requests, polling, or WebSocket connections
  const cancelTranscription = () => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
  };

  // Poll the task result endpoint for transcription and download URL
  const pollTaskResult = (taskId: string) => {
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/task/${taskId}/result`);
        if (res.ok) {
          const data = await res.json();
          if (data.transcript && data.download_url) {
            setTranscript(data.transcript);
            setDownloadUrl(data.download_url);
            setIsProcessing(false);
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
            if (wsRef.current) {
              wsRef.current.close();
              wsRef.current = null;
            }
          } else if (data.error) {
            setErrorMessage(data.error);
            setIsProcessing(false);
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
          }
        } else {
          console.error('Polling error: ', res.statusText);
          setErrorMessage(`Polling error: ${res.statusText}`);
          setIsProcessing(false);
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }
      } catch (error) {
        console.error('Polling exception: ', error);
        setErrorMessage(`Polling exception: ${error}`);
        setIsProcessing(false);
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
    }, 2000); // poll every 2 seconds
  };

  // Start a WebSocket connection for progress updates
  const startWebSocket = (taskId: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProcessingProgress(data.progress);
      if (data.progress >= 100 && wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
    ws.onerror = (error) => {
      console.error('WebSocket error: ', error);
    };
    wsRef.current = ws;
  };

  // Handle the response from the initial /transcribe call which now returns a taskId
  const handleUploadResponse = (receivedTaskId: string) => {
    setTaskId(receivedTaskId);
    setIsProcessing(true);
    startWebSocket(receivedTaskId);
    pollTaskResult(receivedTaskId);
  };

  // Handle file transcription request and response
  const handleTranscribe = () => {
    if (!audioSource.file) return;
    console.log(`Sending file: ${audioSource.file.name}`);
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', audioSource.file);
    try {
      const xhr = new XMLHttpRequest();
      xhrRef.current = xhr;
      xhr.open('POST', 'http://localhost:8000/transcribe');
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
        }
      });
      xhr.onload = function () {
        xhrRef.current = null;
        if (xhr.status === 200) {
          setIsUploading(false);
          const response = JSON.parse(xhr.responseText);
          if (response.task_id) {
            handleUploadResponse(response.task_id);
          } else {
            setErrorMessage('No task id returned from server.');
          }
        } else {
          console.error(`Server error: ${xhr.status} ${xhr.statusText}`);
          setIsUploading(false);
          setErrorMessage(`Server error: ${xhr.statusText}`);
        }
      };
      xhr.onerror = function () {
        xhrRef.current = null;
        setIsUploading(false);
        setErrorMessage('Network error occurred.');
      };
      xhr.onabort = function () {
        console.log('Transcription request aborted');
        setIsUploading(false);
      };
      xhr.send(formData);
    } catch (error) {
      console.error('Upload failed:', error);
      setIsUploading(false);
      setErrorMessage('There was an error uploading your file.');
      xhrRef.current = null;
    }
  };

  // Clean up polling interval and WebSocket on component unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Render any error messages
  const renderError = () => {
    return errorMessage ? (
      <div className="bg-red-600 p-4 rounded-lg text-white my-4">
        {errorMessage}
      </div>
    ) : null;
  };

  // Render the main module content based on active module selection
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
              // startProcessing no longer needed here since we handle it with WebSocket and polling
              clearTranscription={clearTranscription}
            />
          </div>
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
              // startProcessing handled via polling/WebSocket
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

  // Determine if the "Clear" button should be shown
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
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
              <ProgressBar progress={processingProgress} type="processing" onCancel={cancelTranscription} />
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

          {audioSource.file && !transcript && !isUploading && !isProcessing && (
            <div className="mt-6 flex gap-4">
              <button
                onClick={resetState}
                className="flex-1 py-3 px-4 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors flex justify-center items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
                Reset
              </button>
            </div>
          )}

          {shouldShowClearButton && !isUploading && !isProcessing && (
            <div className="mt-4 flex justify-end">
              <button
                onClick={clearAll}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                </svg>
                Clear Everything
              </button>
            </div>
          )}
        </div>
      )}

      {!showMainMenu && (audioSource.file || isUploading || isProcessing || transcript) && (
        <ResetButton 
          onReset={resetState}
          onClear={clearTranscription}
          isProcessing={isUploading || isProcessing}
          hasTranscription={Boolean(transcript)}
        />
      )}

      {!showMainMenu && audioSource.file && transcript && !isUploading && !isProcessing && (
        <div className="fixed bottom-6 left-6 z-10">
          <ClearButton 
            onClear={clearAll}
            isProcessing={false}
            includeAudio={true}
          />
        </div>
      )}

      {renderError()}

      <footer className="mt-8 text-center text-gray-500 text-sm">
        © 202X Your Company. All rights reserved.
      </footer>
    </div>
  );
};

export default App;
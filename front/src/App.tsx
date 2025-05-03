import React, { useState, useEffect } from 'react';
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
import config from './config';

// Type definitions
type AudioSource = {
  previewUrl: string | null;
};

type Module = 'upload' | 'url';

// API configuration
const API_BASE_URL = config.api.baseUrl;

const App: React.FC = () => {
  // State management
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
  const [processingMessage, setProcessingMessage] = useState<string>('Preparing to process...');
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    const handleBeforeUnload = () => {
      // Clean up files when the page is refreshed or closed
      if (taskId) {
        // Use the sync version of fetch for beforeunload
        // Note: This may not complete in all browsers, but it's a best effort
        navigator.sendBeacon(`${API_BASE_URL}/cleanup/${taskId}`);
      }
    };
  
    // Add event listener for page unload
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    // Clean up the event listener when component unmounts
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [taskId, API_BASE_URL]);  

  // Fetch transcription when processing is complete
  useEffect(() => {
    if (processingProgress === 100 && taskId) {
      // Only fetch transcription if the message doesn't indicate cancellation
      const isCancelled = processingMessage.toLowerCase().includes('cancel');
      
      if (!isCancelled) {
        fetchTranscription(taskId);
      } else {
        // If cancelled, just update UI state without trying to fetch the transcript
        setIsProcessing(false);
        // Don't clear the audio source or taskId, keep them for potential retry
      }
    }
  }, [processingProgress, taskId, processingMessage]);  

  // Fallback to polling if WebSocket fails
  useEffect(() => {
    if (connectionAttempts > 0 && !isWsConnected && isProcessing) {
      const pollInterval = setInterval(() => {
        if (taskId) {
          pollTaskProgress(taskId);
        }
      }, 2000);
      
      return () => clearInterval(pollInterval);
    }
  }, [connectionAttempts, isWsConnected, isProcessing, taskId]);

  useEffect(() => {
    if (audioSource.previewUrl && !isProcessing) {
      // Check model loading status periodically
      const checkModelLoading = async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/model-loading-status`);
          if (response.ok) {
            const data = await response.json();
            
            if (data.status === "loading") {
              // Update UI to show model loading is happening in background
              setProcessingMessage(`Preparing models: ${data.message}`);
              setProcessingProgress(data.progress);
            }
          }
        } catch (error) {
          console.error("Error checking model loading status:", error);
        }
      };
      
      // Check immediately and then every 2 seconds
      checkModelLoading();
      const interval = setInterval(checkModelLoading, 2000);
      
      return () => clearInterval(interval);
    }
  }, [audioSource.previewUrl, isProcessing]);

  // Function to poll task progress as fallback
  const pollTaskProgress = async (taskId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/task/${taskId}/result`);
      if (response.ok) {
        const data = await response.json();
        
        if (data.download_url) {
          setProcessingProgress(100);
          setProcessingMessage("Processing complete!");
          setDownloadUrl(data.download_url);
          fetchTranscription(taskId);
        }
      }
    } catch (error) {
      console.error("Error polling for progress:", error);
    }
  };

  // WebSocket progress handling
  const handleProgressUpdate = (progress: number, message: string) => {
    if (progress > 0) {
      setIsWsConnected(true);
    }
    
    // Special handling for cancellation messages
    if (message.toLowerCase().includes('cancel')) {
      setProcessingProgress(progress);
      setProcessingMessage(message);
      
      // If progress is 100%, this is a completed cancellation
      if (progress >= 100) {
        setIsProcessing(false);
        
        // Reset the app after a short delay
        setTimeout(() => {
          resetState();
        }, 2000);
      }
      return;
    }
    
    // For completion (100%), we still want to set the progress
    if (progress === 100) {
      setProcessingProgress(100);
      setProcessingMessage(message || "Processing complete!");
      setIsWsConnected(true);
      return;
    }
    
    // For all other progress updates, just update the message without changing the numeric progress
    // This allows the animation to continue running while updating the text
    setProcessingMessage(message || "Processing in progress...");
  }  

  const handleProcessingComplete = (downloadUrl: string) => {
    if (downloadUrl) {
      setDownloadUrl(downloadUrl);
    }
  };

  const handleWebSocketConnectionFailed = () => {
    setConnectionAttempts(prev => prev + 1);
    if (connectionAttempts >= 3) {
      setIsWsConnected(false);
      setProcessingMessage("Processing in progress...");
    }
  };

  // Handle successful file/URL upload
  const handleUploadSuccess = (previewUrl: string, newTaskId: string) => {
    setAudioSource({ previewUrl: `${API_BASE_URL}${previewUrl}` });
    setTaskId(newTaskId);
    setIsUploading(false);
  };

  // Reset all state
  const resetState = () => {
    // Only attempt to clean up if we have a task ID
    if (taskId) {
      // Make a delete request to clean up files on the server
      fetch(`${API_BASE_URL}/cleanup/${taskId}`, { method: 'DELETE' })
        .then(response => {
          if (!response.ok) {
            console.error('Failed to cleanup task on server');
          } else {
            console.log(`Successfully cleaned up task ${taskId} on server`);
          }
        })
        .catch(err => {
          console.error('Error cleaning up task:', err);
        });
    }
    
    // Reset all client-side state
    setAudioSource({ previewUrl: null });
    setTranscript(null);
    setDownloadUrl(null);
    setIsUploading(false);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingProgress(0);
    setTaskId(null);
    setProcessingMessage('Preparing to process...');
    setIsWsConnected(false);
    setConnectionAttempts(0);
    setIsCancelling(false);
  };  
  
  // Clear just the transcription
  const clearTranscription = () => {
    if (taskId) {
      // Use the same full cleanup endpoint
      fetch(`${API_BASE_URL}/cleanup/${taskId}`, { method: 'DELETE' })
        .catch(err => {
          console.error('Error cleaning up task:', err);
        });
    }
    
    // Reset transcription-related state only (but keeps audio UI state)
    setTranscript(null);
    setDownloadUrl(null);
    setIsProcessing(false);
    setProcessingProgress(0);
  };  
  
  // Reset everything
  const clearAll = () => {
    resetState(); // This now includes server cleanup
  };
  
  // Select a module from the main menu
  const handleModuleSelect = (module: Module) => {
    setActiveModule(module);
    setShowMainMenu(false);
    resetState();
  };
  
  // Return to main menu
  const goToMainMenu = () => {
    resetState();
    setShowMainMenu(true);
  };

  // Cancel ongoing transcription
  const cancelTranscription = () => {
    if (taskId && (isUploading || isProcessing) && !isCancelling) {
      setIsCancelling(true);
      setProcessingMessage("Cancelling transcription...");
      setProcessingProgress(99);
      
      fetch(`${API_BASE_URL}/cancel/${taskId}`, { method: 'POST' })
        .then(response => {
          if (!response.ok) {
            console.error('Failed to cancel task on server');
          }
          return response.json();
        })
        .then(data => {
          console.log('Cancellation response:', data);
          
          // KEEP the audioSource
          setIsUploading(false);
          setUploadProgress(0);
          setIsProcessing(false);
          setProcessingProgress(100);
          setProcessingMessage('Transcription cancelled');
          
          setTimeout(() => {
            setIsCancelling(false);
            // Reset processing state only
            setIsProcessing(false); 
            setProcessingProgress(0);
            setProcessingMessage('Preparing to process...');
          }, 2000);
        })
        .catch(err => {
          console.error('Error cancelling task:', err);
          
          setIsUploading(false);
          setUploadProgress(0);
          setIsProcessing(false);
          setProcessingProgress(100);
          setProcessingMessage('Transcription cancelled');
          
          setTimeout(() => {
            setIsCancelling(false);
          }, 2000);
        });
    } else {
      setIsUploading(false);
      setUploadProgress(0);
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  };

  // Start the transcription process
  const startTranscription = async () => {
    if (!taskId) {
      alert('No task ID available. Please upload a file or URL first.');
      return;
    }
    
    setIsProcessing(true);
    setProcessingProgress(5);
    setProcessingMessage('Starting transcription...');
    setIsWsConnected(false);
    setConnectionAttempts(0);

    try {
      const response = await fetch(`${API_BASE_URL}/transcribe/${taskId}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Transcription started:', data);
      
      setProcessingMessage('Processing in progress...');
    } catch (error) {
      console.error('Error starting transcription:', error);
      alert(`Failed to start transcription: ${(error as Error).message}`);
      setIsProcessing(false);
    }
  };

  const handleApiError = (error: any) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      if (error.response.status === 413) {
        alert(`File size exceeds the ${config.upload.maxFileSizeMB}MB limit. Please upload a smaller file.`);
      } else if (error.response.data && error.response.data.detail) {
        alert(`Error: ${error.response.data.detail}`);
      } else {
        alert(`Server error: ${error.response.status}`);
      }
    } else if (error.request) {
      // The request was made but no response was received
      alert('No response received from server. Please check your connection and try again.');
    } else {
      // Something happened in setting up the request that triggered an Error
      alert(`Error: ${error.message}`);
    }
  };
  
  const fetchTranscription = async (taskId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/transcription/${taskId}`);
      
      if (!response.ok) {
        // Special case for cancelled tasks - 202 status code
        if (response.status === 202) {
          console.log("Task was cancelled, no transcription to fetch");
          setIsProcessing(false);
          return; // Exit without showing error
        }
        
        if (response.status === 404) {
          // For 404 errors, don't show an alert - this is handled elsewhere
          console.log('Transcription not found (likely cancelled or not yet created)');
          setIsProcessing(false);
          return;
        }
        
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
      // Only show alerts for errors other than 404s or cancelled tasks
      if (!error.message.includes('404') && !error.message.includes('202')) {
        alert(`Failed to get transcription: ${(error as Error).message}`);
      }
    } finally {
      setIsProcessing(false);
    }
  };  

  // Handle transcription button click
  const handleTranscribe = () => {
    if (!taskId) {
      alert('Please upload a file or URL first.');
      return;
    }
    
    startTranscription();
  };

  // Render the active module content
  const renderModuleContent = () => {
    if (activeModule === 'upload') {
      return renderFileUploadModule();
    } else if (activeModule === 'url') {
      return renderUrlUploadModule();
    }
    return null;
  };

  // Render file upload module
  const renderFileUploadModule = () => {
    return (
      <>
        {!audioSource.previewUrl ? (
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            <FileUpload 
              setTaskId={setTaskId}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              clearTranscription={clearTranscription}
              onUploadSuccess={handleUploadSuccess}
            />
          </div>
        ) : (
          renderAudioPreview()
        )}
      </>
    );
  };

  // Render URL upload module
  const renderUrlUploadModule = () => {
    return (
      <>
        {!audioSource.previewUrl ? (
          <div className="bg-gray-750 rounded-lg p-6 mb-6 border border-gray-700">
            <UrlUpload 
              setTaskId={setTaskId}
              setIsUploading={setIsUploading}
              setUploadProgress={setUploadProgress}
              clearTranscription={clearTranscription}
              onUploadSuccess={handleUploadSuccess}
            />
          </div>
        ) : (
          renderAudioPreview()
        )}
      </>
    );
  };

  // Render audio preview with transcribe button
  const renderAudioPreview = () => {
    return (
      <div className="space-y-4">
        <AudioPlayer audioUrl={audioSource.previewUrl!} />
        {taskId && !transcript && (
          isProcessing ? (
            <button
              onClick={cancelTranscription}
              disabled={isCancelling}
              className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
                ${isCancelling 
                  ? 'bg-gray-500 cursor-not-allowed' 
                  : 'bg-red-600 hover:bg-red-700 active:scale-98 shadow-lg'}`}
            >
              {isCancelling ? 'Cancelling...' : 'Cancel Transcription'}
            </button>
          ) : (
            <button
              onClick={handleTranscribe}
              className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
            >
              Start Transcription
            </button>
          )
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col items-center justify-center p-6 text-gray-200">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-5xl font-extrabold text-gray-100 mb-4">
          <span className="text-blue-400">Clear</span>Converse
        </h1>
        <p className="text-xl text-gray-400 max-w-3xl mx-auto">
          A speech transcription tool mainly powered by Whisper-RESepFormer solution to offer quality transcription even in overlapping speech scenarios.
        </p>
      </div>

      {/* WebSocket progress handler */}
      {taskId && isProcessing && (
        <WebSocketProgressHandler 
          taskId={taskId}
          apiBaseUrl={config.api.baseUrl}
          wsBaseUrl={config.api.wsBaseUrl}
          onProgressUpdate={handleProgressUpdate}
          onComplete={handleProcessingComplete}
          onConnectionFailed={handleWebSocketConnectionFailed}
          maxReconnectAttempts={config.ui.maxWebSocketReconnectAttempts}
        />
      )}
    
      {showMainMenu ? (
        <MainMenu onSelectModule={handleModuleSelect} />
      ) : (
        <div className="w-full max-w-3xl bg-gray-800 shadow-xl rounded-xl p-8 border border-gray-700">
          {/* Module header with back button */}
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-extrabold tracking-tight text-gray-100">
              {activeModule === 'upload' ? 'File Upload' : 'Google Drive URL'}
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
          
          {/* Module content */}
          {renderModuleContent()}

          {/* Upload progress */}
          {isUploading && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-200">Upload Progress</h3>
              </div>
              <ProgressBar 
                type="upload" 
                onCancel={cancelTranscription} 
                message={uploadProgress === 100 ? "Upload complete" : undefined}
                isComplete={uploadProgress === 100}
              />
            </div>
          )}

          {/* Processing progress */}
          {isProcessing && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-200">Processing Progress</h3>
              </div>
              <ProgressBar 
                type="processing" 
                onCancel={cancelTranscription}
                message={processingMessage}
                isComplete={processingProgress === 100}
              />
            </div>
          )}

          {/* Reset button when audio is loaded but not processing */}
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

          {/* Transcription display */}
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
      
      {/* Floating reset button */}
      {!showMainMenu && (audioSource.previewUrl || isUploading || isProcessing || transcript) && (
        <ResetButton 
          onReset={resetState}
          onClear={clearTranscription}
          isProcessing={isUploading || isProcessing}
          hasTranscription={Boolean(transcript)}
        />
      )}
      
      {/* Floating clear button */}
      {!showMainMenu && audioSource.previewUrl && transcript && !isUploading && !isProcessing && (
        <div className="fixed bottom-6 left-6 z-10">
          <ClearButton 
            onClear={clearAll}
            isProcessing={false}
            includeAudio={true}
          />
        </div>
      )}
      
      {/* Footer */}
      <footer className="mt-8 text-center text-gray-500 text-sm">
        © 2025 ClearConverse
      </footer>
    </div>
  );
};

export default App;
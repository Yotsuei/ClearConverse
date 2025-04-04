// components/UrlUpload.tsx
import React, { useState, useEffect } from 'react';

interface UrlUploadProps {
  onFileSelected: (file: File) => void;
  onUploadResponse: (transcript: string, downloadUrl: string) => void;
  setIsUploading: (isUploading: boolean) => void;
  setUploadProgress: (progress: number) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  startProcessing: () => void;
  clearTranscription: () => void;
}

const UrlUpload: React.FC<UrlUploadProps> = ({ 
  onFileSelected, 
  onUploadResponse, 
  setIsUploading, 
  setUploadProgress,
  setIsProcessing,
  startProcessing,
  clearTranscription
}) => {
  const [url, setUrl] = useState<string>('');
  const [isValidUrl, setIsValidUrl] = useState<boolean>(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [uploadXhr, setUploadXhr] = useState<XMLHttpRequest | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [audioLoaded, setAudioLoaded] = useState<boolean>(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Basic URL validation
  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputUrl = e.target.value;
    setUrl(inputUrl);
    
    // Clear previous errors and states
    setUrlError(null);
    setAudioLoaded(false);
    setAudioUrl(null);
    
    // Validate the URL (basic validation)
    if (inputUrl.trim() !== '') {
      try {
        // Check if it's a valid URL format
        new URL(inputUrl);
        
        // Basic URL validation passed
        setIsValidUrl(true);
        
        // Check for Google Drive links and provide more specific guidance
        if (inputUrl.includes('drive.google.com')) {
          // Not an error, just a note
          setUrlError('Note: Google Drive links need to be shared with "Anyone with the link" and direct download links.');
        } else {
          // Advanced validation for audio extensions
          const audioExtensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.mp4'];
          const hasAudioExtension = audioExtensions.some(ext => 
            inputUrl.toLowerCase().endsWith(ext)
          );
          
          if (!hasAudioExtension && 
              !inputUrl.includes('storage.googleapis.com') && 
              !inputUrl.includes('dropbox.com')) {
            // Just a warning, don't invalidate URL
            setUrlError('URL may not be a direct audio file link. Ensure it points directly to an audio file.');
          }
        }
      } catch (e) {
        setIsValidUrl(false);
        setUrlError('Please enter a valid URL');
      }
    } else {
      setIsValidUrl(false);
    }
  };

  const handleCancelUpload = () => {
    if (uploadXhr) {
      uploadXhr.abort(); // Abort the XHR request
      setUploadXhr(null);
    }
    
    // Close WebSocket connection if active
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
      wsConnection.close();
      setWsConnection(null);
    }
    
    // Reset states
    setTaskId(null);
    setIsUploading(false);
    setUploadProgress(0);
  };

  // Set up WebSocket connection for progress updates
  useEffect(() => {
    if (taskId) {
      const ws = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`);
      
      ws.onopen = () => {
        console.log(`WebSocket connection established for task ${taskId}`);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("Progress update:", data);
          
          // Update progress based on the server message
          if (data.progress <= 30) {
            // First 30% is considered upload
            setIsUploading(true);
            setUploadProgress(data.progress * (100/30)); // Scale to 0-100%
          } else {
            // After 30%, switch to processing mode
            setIsUploading(false);
            setIsProcessing(true);
            // Scale from 30-100 to 0-100
            const scaledProgress = ((data.progress - 30) / (100 - 30)) * 100;
            startProcessing(); // Ensure processing mode is active
            setUploadProgress(Math.min(Math.round(scaledProgress), 99)); // Cap at 99% until complete
          }
          
          // Add message display if you want to show server messages
          if (data.message) {
            console.log("Server message:", data.message);
            // Could display this message in the UI
          }
          
          // If processing is complete
          if (data.progress >= 100) {
            setIsUploading(false);
            setIsProcessing(false);
            setUploadProgress(100);
            
            // Fetch the result
            fetch(`http://localhost:8000/task/${taskId}/result`)
              .then(response => response.json())
              .then(result => {
                if (result.error) {
                  throw new Error(result.error);
                }
                
                // Fetch transcript content
                fetch(`http://localhost:8000${result.download_url}`)
                  .then(response => response.text())
                  .then(transcript => {
                    onUploadResponse(transcript, result.download_url);
                  });
              })
              .catch(error => {
                console.error("Error fetching results:", error);
                setUrlError(`Error processing audio: ${error.message}`);
              })
              .finally(() => {
                ws.close();
              });
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };
      
      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setUrlError("Connection error while monitoring progress");
      };
      
      ws.onclose = () => {
        console.log("WebSocket connection closed");
      };
      
      setWsConnection(ws);
      
      // Clean up WebSocket on unmount or when taskId changes
      return () => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      };
    }
  }, [taskId]);

  const handleLoadAudio = async () => {
    if (!url || !isValidUrl) {
      setUrlError('Please enter a valid URL');
      return;
    }
    
    setIsLoading(true);
    setAudioLoaded(false);
    setAudioUrl(null);
    setUrlError(null);
    
    try {
      // Directly use the URL without conversion since the API handles it
      // Optionally, you could directly set a temporary preview URL if available
      // Here we assume the preview will be available after transcription.
      const audio = new Audio();
      audio.onloadeddata = () => {
        // Set a temporary state or simply mark as loaded.
        setAudioLoaded(true);
        setIsLoading(false);
      };
      audio.onerror = (e) => {
        console.error("Audio loading error:", e);
        setUrlError('Error loading audio from URL. Ensure the URL points directly to an accessible audio file.');
        setIsLoading(false);
      };
      audio.crossOrigin = "anonymous";
      audio.src = url;
      audio.load();
    } catch (error) {
      console.error('URL loading failed:', error);
      setUrlError(`Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`);
      setIsLoading(false);
    }
  };

  // Second step: Transcribe the audio
  const handleTranscribe = async () => {
    clearTranscription();
    if (!url || !isValidUrl) {
      setUrlError('Please enter a valid audio URL');
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(5);
    setUrlError(null);
    
    try {
      // Directly use the provided URL; no need to convert it here.
      const response = await fetch('http://localhost:8000/transcribe-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `url=${encodeURIComponent(url)}`
      });
      
      if (!response.ok) {
        let errorMessage = await response.text();
        try {
          const errorJson = JSON.parse(errorMessage);
          errorMessage = errorJson.detail || `Error ${response.status}: ${response.statusText}`;
        } catch (e) {}
        throw new Error(errorMessage);
      }
      
      const result = await response.json();
      if (result.task_id && result.preview_url) {
        setTaskId(result.task_id);
        // Use the preview URL returned by the API to show the audio preview.
        setAudioUrl(result.preview_url);
      } else {
        throw new Error("Invalid response from server: missing task_id or preview_url");
      }
    } catch (error) {
      console.error('URL transcription failed:', error);
      setUrlError(`Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`);
      setIsUploading(false);
      setUploadProgress(0);
      setTaskId(null);
    }
  };

  const handleClearUrl = () => {
    setUrl('');
    setIsValidUrl(false);
    setUrlError(null);
    setAudioLoaded(false);
    setAudioUrl(null);
    // Also notify parent component to clear audio preview
    const emptyBlob = new Blob([], { type: 'audio/mp3' });
    const emptyFile = new File([emptyBlob], 'empty.mp3', { type: 'audio/mp3' });
    onFileSelected(emptyFile);
    clearTranscription();
  };

  return (
    <div className="flex flex-col">
      <h2 className="text-xl font-bold text-gray-200 mb-4">Audio from URL</h2>
      <p className="text-gray-400 mb-6 text-center">
        Enter a direct URL to an audio file for transcription.
      </p>
      
      <div className="mb-4">
        <label htmlFor="audio-url" className="block text-sm font-medium text-gray-300 mb-1">
          Audio URL
        </label>
        <div className="relative">
          <input
            id="audio-url"
            type="text"
            value={url}
            onChange={handleUrlChange}
            placeholder="https://example.com/audio.mp3"
            className={`w-full p-3 pr-10 bg-gray-700 text-gray-200 border rounded-lg focus:outline-none focus:ring-2 ${
              url && !isValidUrl 
                ? 'border-red-500 focus:ring-red-500 focus:border-red-500' 
                : 'border-gray-600 focus:ring-blue-500 focus:border-blue-500'
            }`}
          />
          {url && (
            <button
              type="button"
              onClick={handleClearUrl}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-gray-300"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          )}
        </div>
        {urlError && (
          <p className={`mt-1 text-sm ${urlError.startsWith('Note') ? 'text-blue-400' : urlError.startsWith('URL may') ? 'text-yellow-400' : 'text-red-400'}`}>
            {urlError}
          </p>
        )}
        <p className="mt-1 text-sm text-gray-400">
          Enter the direct URL to an audio file. The URL must be publicly accessible.
        </p>
        
        {/* Google Drive specific helper */}
        {url && url.includes('drive.google.com') && (
          <div className="mt-2 p-2 bg-blue-900/30 border border-blue-800 rounded-lg text-xs text-blue-300">
            <p className="font-medium mb-1">Google Drive Links:</p>
            <ol className="list-decimal list-inside space-y-1 ml-1">
              <li>Open your Google Drive file</li>
              <li>Click "Share" and set access to "Anyone with the link"</li>
              <li>For best results, use the direct download link format</li>
            </ol>
          </div>
        )}
      </div>
      
      {/* Step 1 - Load Audio Button (when not yet loaded) */}
      {!audioLoaded && !isLoading && !uploadXhr && (
        <button
          onClick={handleLoadAudio}
          disabled={!url || !isValidUrl}
          className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            ${!url || !isValidUrl
              ? 'bg-gray-600 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg'}`
          }
        >
          Load Audio from URL
        </button>
      )}
      
      {/* Loading indicator */}
      {isLoading && (
        <button
          disabled
          className="w-full py-3 px-5 text-white font-bold rounded-lg bg-blue-600 flex items-center justify-center"
        >
          <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Loading Audio...
        </button>
      )}
      
      {/* Step 2 - Transcribe Button (after audio is loaded) */}
      {audioLoaded && audioUrl && (
        <audio controls src={audioUrl} crossOrigin="anonymous">
          Your browser does not support the audio element.
        </audio>
      )}

      {audioLoaded && !uploadXhr && !taskId && (
        <button
          onClick={handleTranscribe}
          className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
        >
          Transcribe Audio
        </button>
      )}
      
      {/* Cancel button (during transcription) */}
      {(uploadXhr || taskId) && !(isUploading === false && isProcessing === false) && (
        <button
          onClick={handleCancelUpload}
          className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            bg-red-600 hover:bg-red-700 active:scale-98 shadow-lg"
        >
          Cancel Transcription
        </button>
      )}
      
      {/* Additional help information */}
      <div className="mt-6 bg-gray-700 p-3 rounded-lg text-sm border border-gray-600">
        <h3 className="font-semibold text-gray-200 mb-1 flex items-center">
          <svg className="w-4 h-4 mr-1 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          Supported URL Types
        </h3>
      </div>
    </div>
  );
};

export default UrlUpload;
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

  // Basic URL validation
  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputUrl = e.target.value;
    setUrl(inputUrl);
    
    // Clear previous errors
    setUrlError(null);
    
    // Validate the URL (basic validation)
    if (inputUrl.trim() !== '') {
      try {
        // Check if it's a valid URL format
        new URL(inputUrl);
        
        // Basic URL validation passed
        setIsValidUrl(true);
        
        // Advanced validation can be added here if needed
        const audioExtensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.mp4'];
        const hasAudioExtension = audioExtensions.some(ext => 
          inputUrl.toLowerCase().endsWith(ext)
        );
        
        if (!hasAudioExtension && 
            !inputUrl.includes('drive.google.com') && 
            !inputUrl.includes('dropbox.com') &&
            !inputUrl.includes('storage.googleapis.com')) {
          // Just a warning, don't invalidate URL
          setUrlError('Warning: URL doesn\'t appear to be an audio file or from a recognized service.');
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
          setUploadProgress(data.progress);
          
          // If processing is complete
          if (data.progress >= 100) {
            setIsUploading(false);
            
            // Fetch the result
            fetch(`http://localhost:8000/task/${taskId}/result`)
              .then(response => response.json())
              .then(result => {
                if (result.error) {
                  throw new Error(result.error);
                }
                
                // Create a dummy file for audio preview
                // This is just for UI continuity - the actual audio is processed directly from the URL
                const dummyAudioBlob = new Blob([], { type: 'audio/mp3' });
                const audioFile = new File([dummyAudioBlob], 'url_audio.mp3', { type: 'audio/mp3' });
                
                // Pass results to parent components
                onFileSelected(audioFile);
                
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

  const handleUpload = async () => {
    // Clear any previous transcription when uploading
    clearTranscription();
    
    if (!url || !isValidUrl) {
      setUrlError('Please enter a valid URL');
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      // Use XMLHttpRequest for request handling
      const xhr = new XMLHttpRequest();
      setUploadXhr(xhr);
      
      // Using the dedicated URL transcription endpoint
      xhr.open('POST', 'http://localhost:8000/transcribe-url');
      xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
      
      xhr.onload = function() {
        if (xhr.status === 200) {
          // Parse response to get task ID
          try {
            const response = JSON.parse(xhr.responseText);
            if (response.task_id) {
              setTaskId(response.task_id);
              startProcessing();
            } else {
              throw new Error("Invalid response from server: missing task_id");
            }
          } catch (err) {
            throw new Error(`Error parsing server response: ${xhr.responseText}`);
          }
        } else {
          let errorMsg = "Server error";
          try {
            const response = JSON.parse(xhr.responseText);
            errorMsg = response.detail || `Server error: ${xhr.status}`;
          } catch (e) {
            errorMsg = `Server error: ${xhr.status} ${xhr.statusText}`;
          }
          throw new Error(errorMsg);
        }
      };

      xhr.onerror = function() {
        throw new Error('Network error occurred. Please check your connection and try again.');
      };

      xhr.onabort = function() {
        console.log('URL upload aborted');
        setIsUploading(false);
        setUploadXhr(null);
      };

      // Send the URL as form data
      xhr.send(`url=${encodeURIComponent(url)}`);
    } catch (error) {
      console.error('URL upload failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setUrlError(`Error: ${errorMessage}`);
      setIsUploading(false);
      setUploadXhr(null);
    }
  };

  const handleClearUrl = () => {
    setUrl('');
    setIsValidUrl(false);
    setUrlError(null);
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
          <p className={`mt-1 text-sm ${urlError.startsWith('Warning') ? 'text-yellow-400' : 'text-red-400'}`}>
            {urlError}
          </p>
        )}
        <p className="mt-1 text-sm text-gray-400">
          Enter the direct URL to an audio file. The URL must be publicly accessible.
        </p>
      </div>
      
      <div className="bg-gray-700 border border-gray-600 rounded-lg p-4 mb-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-gray-200">Supported URL Sources</h3>
            <div className="mt-2 text-sm text-gray-400">
              <ul className="list-disc list-inside space-y-1">
                <li>Direct links to audio files (MP3, WAV, OGG, etc.)</li>
                <li>Google Drive links (must be publicly accessible)</li>
                <li>Dropbox shared links</li>
                <li>Other publicly accessible audio hosting services</li>
              </ul>
              <p className="mt-2">For best results, use direct links to MP3 or WAV files.</p>
            </div>
          </div>
        </div>
      </div>
      
      {!uploadXhr ? (
        <button
          onClick={handleUpload}
          disabled={!url || !isValidUrl}
          className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            ${!url || !isValidUrl
              ? 'bg-gray-600 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg'}`
          }
        >
          Transcribe from URL
        </button>
      ) : (
        <button
          onClick={handleCancelUpload}
          className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            bg-red-600 hover:bg-red-700 active:scale-98 shadow-lg"
        >
          Cancel Transcription
        </button>
      )}
    </div>
  );
};

export default UrlUpload;
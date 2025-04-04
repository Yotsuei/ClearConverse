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
            setUploadProgress(100); // Upload is complete
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

  // First step: Load the audio without transcribing
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
      // Create a URL object to validate
      new URL(url);
      
      // Use a proxy/cors approach or direct if allowed
      // For this example, we'll create a blob URL for the audio
      // In a real implementation, you might need to download via backend
      
      // Create a temporary audio element to test loading
      const audio = new Audio();
      
      // Set up event listeners
      audio.onloadeddata = () => {
        // Create a blob URL from the audio 
        // (In a real implementation, you should download the file)
        
        // Here we'll "fake" the audio loading by just passing the URL directly
        // which works for many direct audio URLs but not for services that require authentication
        
        // Create a blob with metadata for the file props
        const temporaryBlob = new Blob(['audio content'], { type: 'audio/mpeg' });
        const fileName = url.split('/').pop() || 'audio_from_url.mp3';
        const file = new File([temporaryBlob], fileName, { type: 'audio/mpeg' });
        
        // Pass file to parent component
        onFileSelected(file);
        
        // Set the audio URL for direct playing
        setAudioUrl(url);
        setAudioLoaded(true);
        setIsLoading(false);
      };
      
      audio.onerror = () => {
        setUrlError('Error loading audio from URL. Make sure it\'s a direct link to an audio file.');
        setIsLoading(false);
      };
      
      // Start loading the audio
      audio.src = url;
      audio.load();
      
    } catch (error) {
      console.error('URL loading failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setUrlError(`Error: ${errorMessage}`);
      setIsLoading(false);
    }
  };

  // Second step: Transcribe the audio (called after preview)
  const handleTranscribe = async () => {
    // Clear any previous transcription when uploading
    clearTranscription();
    
    if (!url || !isValidUrl || !audioLoaded) {
      setUrlError('Please load the audio first');
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(5); // Start with a small initial progress
    setUrlError(null);
    
    try {
      // Using the dedicated URL transcription endpoint
      const response = await fetch('http://localhost:8000/transcribe-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `url=${encodeURIComponent(url)}`
      });
      
      if (!response.ok) {
        let errorMessage = await response.text();
        try {
          const errorJson = JSON.parse(errorMessage);
          errorMessage = errorJson.detail || `Error ${response.status}: ${response.statusText}`;
        } catch (e) {
          // If not JSON, use the text directly
        }
        throw new Error(errorMessage);
      }
      
      const result = await response.json();
      
      if (result.task_id) {
        setTaskId(result.task_id);
        // WebSocket will now track progress
      } else {
        throw new Error("Invalid response from server: missing task_id");
      }
    } catch (error) {
      console.error('URL transcription failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setUrlError(`Error: ${errorMessage}`);
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
          <p className={`mt-1 text-sm ${urlError.startsWith('Warning') ? 'text-yellow-400' : 'text-red-400'}`}>
            {urlError}
          </p>
        )}
        <p className="mt-1 text-sm text-gray-400">
          Enter the direct URL to an audio file. The URL must be publicly accessible.
        </p>
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
      {audioLoaded && !uploadXhr && (
        <button
          onClick={handleTranscribe}
          className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
        >
          Transcribe Audio
        </button>
      )}
      
      {/* Cancel button (during transcription) */}
      {uploadXhr && (
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
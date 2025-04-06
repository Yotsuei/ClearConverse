// components/UrlUpload.tsx
import React, { useState } from 'react';

interface UrlUploadProps {
  setTaskId: (taskId: string) => void;
  setIsUploading: (isUploading: boolean) => void;
  setUploadProgress: (progress: number) => void;
  clearTranscription: () => void;
  onUploadSuccess: (previewUrl: string, taskId: string) => void; // Add this prop
}


const UrlUpload: React.FC<UrlUploadProps> = ({ 
  setTaskId,
  setIsUploading, 
  setUploadProgress,
  clearTranscription,
  onUploadSuccess // Add this prop
}) => {
  const [url, setUrl] = useState<string>('');
  const [isValidUrl, setIsValidUrl] = useState<boolean>(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [uploadXhr, setUploadXhr] = useState<XMLHttpRequest | null>(null);

  // Handle URL input change
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
        
        // Check if it's a valid audio/video URL
        const isAudioVideo = 
          inputUrl.includes('drive.google.com') || 
          inputUrl.includes('docs.google.com') || 
          inputUrl.includes('storage.googleapis.com') ||
          inputUrl.includes('youtube.com') ||
          inputUrl.includes('youtu.be') ||
          inputUrl.includes('soundcloud.com') ||
          inputUrl.endsWith('.mp3') ||
          inputUrl.endsWith('.wav') ||
          inputUrl.endsWith('.ogg') ||
          inputUrl.endsWith('.mp4');
          
        if (isAudioVideo) {
          setIsValidUrl(true);
        } else {
          setIsValidUrl(false);
          setUrlError('Please enter a valid audio/video URL');
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
    
    // Reset uploading state
    setIsUploading(false);
    setUploadProgress(0);
  };

  const handleUpload = async () => {
    
    if (!url || !isValidUrl) {
      setUrlError('Please enter a valid audio/video URL');
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    
    // Create form data with the URL
    const formData = new FormData();
    formData.append('url', url);

    try {
      // Implement XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      setUploadXhr(xhr);
      
      xhr.open('POST', 'http://localhost:8000/upload-url');
      
      // Since url uploads don't have reliable progress events,
      // simulate progress with a counter
      let simulatedProgress = 0;
      const progressInterval = setInterval(() => {
        simulatedProgress += 5;
        if (simulatedProgress <= 90) {
          setUploadProgress(simulatedProgress);
        } else {
          clearInterval(progressInterval);
        }
      }, 300);
      
      xhr.onload = function() {
        clearInterval(progressInterval);
        
        if (xhr.status === 200) {
          setUploadProgress(100);
          setIsUploading(false);
          setUploadXhr(null);
          
          const response = JSON.parse(xhr.responseText);
          
          if (response.task_id) {
            console.log('URL uploaded. Task ID:', response.task_id);  
            
            // Modified: Use the preview URL from backend response
            if (response.preview_url) {
              onUploadSuccess(response.preview_url, response.task_id); // This should handle both
            }  
          } else {
            throw new Error('No task ID returned from server');
          }
        } else {
          throw new Error(`Error: ${xhr.statusText || 'Server returned an error'}`);
        }
      };

      xhr.onerror = function() {
        clearInterval(progressInterval);
        setUploadXhr(null);
        throw new Error('Network error occurred');
      };

      xhr.onabort = function() {
        clearInterval(progressInterval);
        console.log('Upload aborted');
      };

      xhr.send(formData);
    } catch (error) {
      console.error('URL upload failed:', error);
      setUrlError(`There was an error processing your URL: ${(error as Error).message}`);
      setIsUploading(false);
      setUploadXhr(null);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    e.preventDefault();
    const pastedText = e.clipboardData.getData('text');
    setUrl(pastedText);
    
    // Validate pasted URL
    try {
      new URL(pastedText);
      const isAudioVideo = 
        pastedText.includes('drive.google.com') || 
        pastedText.includes('docs.google.com') || 
        pastedText.includes('storage.googleapis.com') ||
        pastedText.includes('youtube.com') ||
        pastedText.includes('youtu.be') ||
        pastedText.includes('soundcloud.com') ||
        pastedText.endsWith('.mp3') ||
        pastedText.endsWith('.wav') ||
        pastedText.endsWith('.ogg') ||
        pastedText.endsWith('.mp4');
        
      if (isAudioVideo) {
        setIsValidUrl(true);
        setUrlError(null);
      } else {
        setIsValidUrl(false);
        setUrlError('Please enter a valid audio/video URL');
      }
    } catch (e) {
      setIsValidUrl(false);
      setUrlError('Please enter a valid URL');
    }
  };

  const handleClearUrl = () => {
    setUrl('');
    setIsValidUrl(false);
    setUrlError(null);
  };

  return (
    <div className="flex flex-col">
      <div className="mb-4">
        <label htmlFor="audio-url" className="block text-sm font-medium text-gray-300 mb-1">
          Audio/Video URL
        </label>
        <div className="relative">
          <input
            id="audio-url"
            type="text"
            value={url}
            onChange={handleUrlChange}
            onPaste={handlePaste}
            placeholder="https://example.com/audio.mp3 or Google Drive URL"
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
          <p className="mt-1 text-sm text-red-400">{urlError}</p>
        )}
        <p className="mt-1 text-sm text-gray-400">
          Paste a link to an audio/video file. For Google Drive, make sure the file is publicly accessible.
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
            <h3 className="text-sm font-medium text-gray-200">Supported URLs</h3>
            <div className="mt-2 text-sm text-gray-400">
              <ul className="list-disc pl-5 space-y-1">
                <li>Direct links to MP3, WAV, OGG, or MP4 files</li>
                <li>Google Drive shared audio/video files</li>
                <li>YouTube videos (audio will be extracted)</li>
                <li>SoundCloud tracks (if publicly accessible)</li>
              </ul>
            </div>
            <p className="mt-2 text-xs text-gray-400">
              For Google Drive: Open your file, click "Share", then "Change to anyone with the link",
              and finally "Copy link".
            </p>
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
          Upload from URL
        </button>
      ) : (
        <button
          onClick={handleCancelUpload}
          className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            bg-red-600 hover:bg-red-700 active:scale-98 shadow-lg"
        >
          Cancel Upload
        </button>
      )}
    </div>
  );
};

export default UrlUpload;
// components/UrlUpload.tsx
import React, { useState } from 'react';

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
  const [isValidUrl, setIsValidUrl] = useState<boolean>(true);
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
        
        // Check if it's a Google Drive URL
        if (inputUrl.includes('drive.google.com') || 
            inputUrl.includes('docs.google.com') || 
            inputUrl.includes('storage.googleapis.com')) {
          setIsValidUrl(true);
        } else {
          setIsValidUrl(false);
          setUrlError('Please enter a valid Google Drive URL');
        }
      } catch (e) {
        setIsValidUrl(false);
        setUrlError('Please enter a valid URL');
      }
    } else {
      setIsValidUrl(false);
      setUrlError(null);
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
    // Clear any previous transcription
    clearTranscription();
    
    if (!url || !isValidUrl) {
      setUrlError('Please enter a valid Google Drive URL');
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    
    // Create form data with the URL
    const formData = new FormData();
    formData.append('url', url);

    try {
      // Simulate direct file download first, since the backend will do this
      setUploadProgress(10);
      
      // Simulate processing
      setTimeout(() => setUploadProgress(30), 300);
      setTimeout(() => setUploadProgress(50), 600);
      
      // Implement XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      setUploadXhr(xhr); // Store XHR reference for cancel functionality
      
      xhr.open('POST', 'http://localhost:8000/transcribe-url');
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          // Start from 50% since we're simulating the prior download
          const progress = 50 + Math.round((event.loaded / event.total) * 50);
          setUploadProgress(progress);
        }
      });

      xhr.onload = function() {
        if (xhr.status === 200) {
          setIsUploading(false);
          setUploadXhr(null);
          startProcessing();
          
          // Parse response
          const response = JSON.parse(xhr.responseText);
          
          // Create a File object from the URL for preview purposes
          fetch(response.temp_audio_url)
            .then(res => res.blob())
            .then(blob => {
              const file = new File([blob], "google_drive_audio.mp3", { type: "audio/mp3" });
              onFileSelected(file);
              onUploadResponse(response.transcript, response.download_url);
            });
        } else {
          throw new Error(`Error: ${xhr.statusText}`);
        }
      };

      xhr.onerror = function() {
        setUploadXhr(null);
        throw new Error('Network error occurred');
      };

      xhr.onabort = function() {
        console.log('Upload aborted');
      };

      xhr.send(formData);
    } catch (error) {
      console.error('URL upload failed:', error);
      setUrlError('There was an error processing your URL. Please make sure it\'s accessible.');
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
      if (pastedText.includes('drive.google.com') || 
          pastedText.includes('docs.google.com') || 
          pastedText.includes('storage.googleapis.com')) {
        setIsValidUrl(true);
        setUrlError(null);
      } else {
        setIsValidUrl(false);
        setUrlError('Please enter a valid Google Drive URL');
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
        <label htmlFor="drive-url" className="block text-sm font-medium text-gray-700 mb-1">
          Google Drive Audio URL
        </label>
        <div className="relative">
          <input
            id="drive-url"
            type="text"
            value={url}
            onChange={handleUrlChange}
            onPaste={handlePaste}
            placeholder="https://drive.google.com/file/d/..."
            className={`w-full p-3 pr-10 border rounded-lg focus:outline-none focus:ring-2 ${
              url && !isValidUrl 
                ? 'border-red-300 focus:ring-red-500 focus:border-red-500' 
                : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
            }`}
          />
          {url && (
            <button
              type="button"
              onClick={handleClearUrl}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-gray-500"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          )}
        </div>
        {urlError && (
          <p className="mt-1 text-sm text-red-600">{urlError}</p>
        )}
        <p className="mt-1 text-sm text-gray-500">
          Paste a link to a Google Drive audio file. Make sure the file is publicly accessible.
        </p>
      </div>
      
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">How to get a shareable link from Google Drive</h3>
            <div className="mt-2 text-sm text-blue-700">
              <ol className="list-decimal pl-5 space-y-1">
                <li>Open your file in Google Drive</li>
                <li>Click on "Share" in the top right</li>
                <li>Click on "Change to anyone with the link"</li>
                <li>Click "Copy link" and paste it here</li>
              </ol>
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
              ? 'bg-gray-400 cursor-not-allowed' 
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
          Cancel Upload
        </button>
      )}
    </div>
  );
};

export default UrlUpload;
// components/FileUpload.tsx
import React, { useState } from 'react';

interface FileUploadProps {
  onFileSelected: (file: File) => void;
  onUploadResponse: (transcript: string, downloadUrl: string) => void;
  setIsUploading: (isUploading: boolean) => void;
  setUploadProgress: (progress: number) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  startProcessing: () => void;
  clearTranscription: () => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onFileSelected, 
  onUploadResponse, 
  setIsUploading, 
  setUploadProgress,
  setIsProcessing,
  startProcessing,
  clearTranscription
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadXhr, setUploadXhr] = useState<XMLHttpRequest | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      // Clear any previous transcription when selecting a new file
      clearTranscription();
      
      const selectedFile = event.target.files[0];
      setFile(selectedFile);
      onFileSelected(selectedFile);
    }
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      // Clear any previous transcription when dropping a new file
      clearTranscription();
      
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      onFileSelected(droppedFile);
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
    if (!file) {
      alert('Please select a file first.');
      return;
    }

    // Clear any previous transcription when uploading
    clearTranscription();
    
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Implement XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      setUploadXhr(xhr); // Store XHR reference for cancel functionality
      
      xhr.open('POST', 'http://localhost:8000/transcribe');
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
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
          onUploadResponse(response.transcript, response.download_url);
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
      console.error('Upload failed:', error);
      alert('There was an error uploading your file.');
      setIsUploading(false);
      setUploadXhr(null);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div 
        className={`w-full border-2 border-dashed rounded-lg p-6 mb-4 cursor-pointer text-center transition-colors
          ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('fileInput')?.click()}
      >
        <input
          id="fileInput"
          type="file"
          accept=".mp3,.wav,.m4a,.ogg"
          onChange={handleFileChange}
          className="hidden"
        />
        
        <div className="flex flex-col items-center justify-center py-5">
          <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
          </svg>
          <p className="mb-2 text-sm text-gray-500">
            <span className="font-semibold">Click to upload</span> or drag and drop
          </p>
          <p className="text-xs text-gray-500">MP3, WAV, M4A or OGG (MAX. 20MB)</p>
        </div>
      </div>
      
      {file && (
        <div className="w-full bg-gray-100 rounded-lg p-3 mb-4 flex items-center">
          <div className="bg-blue-100 rounded-lg p-2 mr-3">
            <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
            </svg>
          </div>
          <div className="flex-grow truncate">
            <p className="text-sm font-medium truncate">{file.name}</p>
            <p className="text-xs text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setFile(null);
            }}
            className="ml-2 text-gray-400 hover:text-red-500"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
      )}
      
      {!uploadXhr ? (
        <button
          onClick={handleUpload}
          disabled={!file}
          className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            ${!file 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg'}`
          }
        >
          Transcribe Audio
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

export default FileUpload;
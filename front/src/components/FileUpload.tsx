// components/FileUpload.tsx
import React, { useState, useRef } from 'react';

interface FileUploadProps {
  onUploadResponse: (transcript: string, downloadUrl: string, audioBlob: Blob) => void;
  setLoading: (loading: boolean) => void;
  updateProgress: (progress: number) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onUploadResponse, 
  setLoading, 
  updateProgress 
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select an audio file first.');
      return;
    }

    setLoading(true);
    updateProgress(0);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Create a new XMLHttpRequest to track upload progress
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = Math.round((event.loaded / event.total) * 50);
          updateProgress(percentComplete); // Only go up to 50% on upload, the other 50% is for processing
        }
      });

      xhr.addEventListener('load', async () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          const data = JSON.parse(xhr.responseText);
          // Simulate processing progress (from 50% to 100%)
          let progress = 50;
          const progressInterval = setInterval(() => {
            progress += 2;
            updateProgress(Math.min(progress, 99));
            if (progress >= 100) clearInterval(progressInterval);
          }, 200);
          
          // Convert the audio file to a blob for playback
          const audioBlob = file;
          
          setTimeout(() => {
            updateProgress(100);
            onUploadResponse(data.transcript, data.download_url, audioBlob);
            clearInterval(progressInterval);
          }, 2000);
        } else {
          throw new Error(`Error: ${xhr.statusText}`);
        }
      });

      xhr.addEventListener('error', () => {
        throw new Error('Network error occurred');
      });

      xhr.open('POST', 'http://localhost:8000/transcribe');
      xhr.send(formData);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('There was an error uploading your file.');
      setLoading(false);
      updateProgress(0);
    }
  };

  return (
    <div 
      className={`bg-slate-800/50 border ${dragActive ? 'border-blue-400 bg-slate-800/80' : 'border-slate-600'} rounded-xl p-6 mb-6 transition-all duration-300`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <div 
        className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors duration-300"
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept=".mp3,.wav"
          onChange={handleFileChange}
          className="hidden"
        />
        
        <div className="flex flex-col items-center justify-center">
          <svg 
            className="w-12 h-12 text-slate-400 mb-3" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M9 19V13.5M12 15V10.5M15 17V14.5M5 21H19C20.1046 21 21 20.1046 21 19V5C21 3.89543 20.1046 3 19 3H5C3.89543 3 3 3.89543 3 5V19C3 20.1046 3.89543 21 5 21Z" 
            />
          </svg>
          <p className="mb-2 text-sm text-slate-400">
            <span className="font-semibold">Click to upload</span> or drag and drop
          </p>
          <p className="text-xs text-slate-500">MP3 or WAV (Max 10MB)</p>
        </div>
      </div>
      
      {file && (
        <div className="mt-4 bg-slate-700/50 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center">
            <svg 
              className="w-8 h-8 text-blue-400 mr-3" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={1.5} 
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" 
              />
            </svg>
            <div className="overflow-hidden">
              <p className="text-sm font-medium text-white truncate">{file.name}</p>
              <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          </div>
          <button 
            onClick={() => setFile(null)}
            className="ml-4 p-1 rounded-full hover:bg-slate-600 text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
      
      <button
        onClick={handleUpload}
        disabled={!file}
        className={`w-full mt-4 py-3 px-5 text-white font-medium rounded-lg transition-all duration-300 flex items-center justify-center
          ${file ? 'bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg' : 'bg-slate-700 cursor-not-allowed'}`}
      >
        <svg 
          className="w-5 h-5 mr-2" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24" 
          xmlns="http://www.w3.org/2000/svg"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
          />
        </svg>
        Process Audio
      </button>
    </div>
  );
};

export default FileUpload;
import React, { useState } from 'react';
import config from '../config';

interface FileUploadProps {
  setTaskId: (taskId: string) => void;
  setIsUploading: (isUploading: boolean) => void;
  setUploadProgress: (progress: number) => void;
  clearTranscription: () => void;
  onUploadSuccess: (previewUrl: string, taskId: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  setTaskId,
  setIsUploading, 
  setUploadProgress,
  clearTranscription,
  onUploadSuccess
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadXhr, setUploadXhr] = useState<XMLHttpRequest | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  // Validate file type - prioritizing WAV and MP3 formats (most compatible)
  const isValidFileType = (file: File): boolean => {
    // Primary formats (fully supported)
    const primaryExtensions = ['.wav', '.mp3'];
    const primaryMimeTypes = [
      'audio/wav', 
      'audio/mpeg', 
      'audio/mp3'
    ];
    
    // Secondary formats (may require conversion)
    const secondaryExtensions = ['.mp4', '.webm', '.ogg', '.flac', '.m4a', '.aac'];
    const secondaryMimeTypes = [
      'video/mp4', 
      'audio/mp4',
      'audio/webm',
      'video/webm',
      'audio/ogg',
      'application/ogg',
      'audio/flac',
      'audio/m4a',
      'audio/aac'
    ];
    
    // Combine all valid formats
    const validExtensions = [...primaryExtensions, ...secondaryExtensions];
    const validMimeTypes = [...primaryMimeTypes, ...secondaryMimeTypes];
    
    // Check file extension
    const hasValidExtension = validExtensions.some(ext => 
      file.name.toLowerCase().endsWith(ext)
    );
    
    // Check MIME type
    const hasValidMimeType = validMimeTypes.some(type => 
      file.type === type
    );
    
    return hasValidExtension || hasValidMimeType;
  };

  // Validate file size against the configured limit
  const isValidFileSize = (file: File): boolean => {
    return file.size <= config.upload.maxFileSizeBytes;
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      // Clear any previous transcription when selecting a new file
      clearTranscription();
      
      const selectedFile = event.target.files[0];
      
      // First check file type
      if (!isValidFileType(selectedFile)) {
        setFileError('Invalid file type. Please use .wav or .mp3 files for best results.');
        setFile(null);
        setFileName(null);
        return;
      }
      
      // Then check file size
      if (!isValidFileSize(selectedFile)) {
        const fileSizeMB = (selectedFile.size / (1024 * 1024)).toFixed(2);
        setFileError(`File size exceeds the ${config.upload.maxFileSizeMB}MB limit. Your file is ${fileSizeMB}MB.`);
        setFile(null);
        setFileName(null);
        return;
      }
      
      setFileError(null);
      setFile(selectedFile);
      setFileName(selectedFile.name);
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
      
      // First check file type
      if (!isValidFileType(droppedFile)) {
        setFileError('Invalid file type. Please use .wav or .mp3 files for best results.');
        setFile(null);
        setFileName(null);
        return;
      }
      
      // Then check file size
      if (!isValidFileSize(droppedFile)) {
        const fileSizeMB = (droppedFile.size / (1024 * 1024)).toFixed(2);
        setFileError(`File size exceeds the ${config.upload.maxFileSizeMB}MB limit. Your file is ${fileSizeMB}MB.`);
        setFile(null);
        setFileName(null);
        return;
      }
      
      setFileError(null);
      setFile(droppedFile);
      setFileName(droppedFile.name);
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
    
    // Final size check before upload
    if (!isValidFileSize(file)) {
      const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
      setFileError(`File size exceeds the ${config.upload.maxFileSizeMB}MB limit. Your file is ${fileSizeMB}MB.`);
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Implement XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      setUploadXhr(xhr); // Store XHR reference for cancel functionality
      
      xhr.open('POST', `${config.api.baseUrl}/upload-file`);
      
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
          
          // Parse response to get task_id
          const response = JSON.parse(xhr.responseText);
          
          if (response.task_id && response.preview_url) {
            console.log('File uploaded. Task ID:', response.task_id);
            setTaskId(response.task_id);
            
            // Reset file and file name as we'll show the audio player instead
            setFile(null);
            setFileName(null);
            
            onUploadSuccess(response.preview_url, response.task_id);
          } else {
            throw new Error('No task ID returned from server');
          }
        } else if (xhr.status === 413) {
          // Handle "Payload Too Large" error specifically
          throw new Error(`File size exceeds the ${config.upload.maxFileSizeMB}MB limit.`);
        } else {
          throw new Error(`Error: ${xhr.statusText || 'Server returned an error'}`);
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
      setFileError(`There was an error uploading your file: ${(error as Error).message}`);
      setIsUploading(false);
      setUploadXhr(null);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-bold text-gray-200 mb-4">File Upload</h2>
      <p className="text-gray-400 mb-6 text-center">
        Upload your audio or video file for transcription. WAV and MP3 formats are recommended for best results.
      </p>
      
      {/* Conditional rendering: File selection area or file name display */}
      {fileName ? (
        <div className="w-full border-2 border-gray-600 rounded-lg p-6 mb-4 bg-gray-700 transition-colors">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <svg className="h-10 w-10 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
              </svg>
              <div>
                <p className="text-lg font-medium text-gray-200 truncate max-w-xs">{fileName}</p>
                <p className="text-sm text-gray-400">
                  {file?.size ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : ''}
                </p>
              </div>
            </div>
            <button 
              onClick={(e) => {
                e.stopPropagation();
                setFile(null);
                setFileName(null);
              }}
              className="text-gray-400 hover:text-gray-300 bg-gray-800 rounded-full p-2"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      ) : (
        <div 
          className={`w-full border-2 border-dashed rounded-lg p-6 mb-4 cursor-pointer text-center transition-colors
            ${dragActive ? 'border-blue-500 bg-gray-700' : 'border-gray-600 hover:border-blue-400'}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('fileInput')?.click()}
        >
          <input
            id="fileInput"
            type="file"
            accept=".wav,.mp4,.mp3,.webm,.ogg,.flac,.m4a,.aac"
            onChange={handleFileChange}
            className="hidden"
          />
          
          <div className="flex flex-col items-center justify-center py-5">
            <svg className="w-10 h-10 mb-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
            <p className="mb-2 text-sm text-gray-400">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500">Audio/video files only (MAX. {config.upload.maxFileSizeMB}MB)</p>
            <div className="mt-3 flex items-center text-xs text-blue-400">
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              Recommended formats: .wav, .mp3 (best compatibility)
            </div>
          </div>
        </div>
      )}
      
      {fileError && (
        <div className="w-full bg-red-900/50 border border-red-700 rounded-lg p-3 mb-4 text-red-200 text-sm">
          <p>{fileError}</p>
        </div>
      )}
      
      {!uploadXhr ? (
        <button
          onClick={handleUpload}
          disabled={!file}
          className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 
            ${!file 
              ? 'bg-gray-600 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg'}`
          }
        >
          Upload Audio
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
      
      {/* File format explanation */}
      <div className="mt-6 bg-gray-700 p-4 rounded-lg text-sm border border-gray-600 w-full">
        <h3 className="font-semibold text-gray-200 mb-2 flex items-center">
          <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          Supported File Formats
        </h3>
        <p className="mt-2 text-gray-400">For best results, use WAV or MP3 files with clear audio and minimal background noise.</p>
        <p className="mt-2 text-gray-400">Maximum file size: <span className="font-semibold text-blue-400">{config.upload.maxFileSizeMB}MB</span></p>
      </div>
    </div>
  );
};

export default FileUpload;
// components/FileUpload.tsx
import React, { useState, useRef } from 'react';

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
  const [fileError, setFileError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    const secondaryExtensions = ['.mp4', '.webm', '.ogg'];
    const secondaryMimeTypes = [
      'video/mp4', 
      'audio/mp4',
      'audio/webm',
      'video/webm',
      'audio/ogg',
      'application/ogg'
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

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      // Clear any previous transcription when selecting a new file
      clearTranscription();
      
      const selectedFile = event.target.files[0];
      
      if (!isValidFileType(selectedFile)) {
        setFileError('Invalid file type. Please use .wav or .mp3 files for best results.');
        setFile(null);
        return;
      }
      
      setFileError(null);
      setFile(selectedFile);
      onFileSelected(selectedFile);
      
      // Auto-start transcription after file selection
      handleUpload(selectedFile);
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
      
      if (!isValidFileType(droppedFile)) {
        setFileError('Invalid file type. Please use .wav or .mp3 files for best results.');
        setFile(null);
        return;
      }
      
      setFileError(null);
      setFile(droppedFile);
      onFileSelected(droppedFile);
      
      // Auto-start transcription after file drop
      handleUpload(droppedFile);
    }
  };

  const handleCancelUpload = () => {
    if (uploadXhr) {
      uploadXhr.abort(); // Abort the XHR request
      setUploadXhr(null);
    }
    
    // Reset uploading state
    setIsUploading(false);
    setIsProcessing(false);
    setUploadProgress(0);
  };

  const handleUpload = async (fileToUpload: File = file) => {
    if (!fileToUpload) {
      alert('Please select a file first.');
      return;
    }

    // Clear any previous transcription when uploading
    clearTranscription();
    
    setIsUploading(true);
    setIsProcessing(false); // Make sure we start in the upload phase
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', fileToUpload);

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
          try {
            const response = JSON.parse(xhr.responseText);
            console.log("Upload response:", response);
            
            // If the response contains a task_id, start monitoring progress with WebSocket
            if (response.task_id) {
              setTaskId(response.task_id);
              
              // Set up WebSocket connection for progress monitoring
              const ws = new WebSocket(`ws://localhost:8000/ws/progress/${response.task_id}`);
              
              ws.onopen = () => {
                console.log("WebSocket connection opened for processing progress");
              };
              
              ws.onmessage = (event) => {
                try {
                  const data = JSON.parse(event.data);
                  console.log("Processing progress:", data);
                  
                  // Update progress
                  setUploadProgress(data.progress);
                  
                  // When processing is complete
                  if (data.progress >= 100) {
                    setIsProcessing(false);
                    
                    // Fetch the result
                    fetch(`http://localhost:8000/task/${response.task_id}/result`)
                      .then(response => response.json())
                      .then(result => {
                        console.log("Task result:", result);
                        
                        if (result.error) {
                          throw new Error(result.error);
                        }
                        
                        if (!result.download_url) {
                          throw new Error("No download URL in response");
                        }
                        
                        // Fetch the transcript content
                        return fetch(`http://localhost:8000${result.download_url}`)
                          .then(response => response.text())
                          .then(transcript => {
                            console.log("Got transcript, length:", transcript.length);
                            if (transcript) {
                              onUploadResponse(transcript, result.download_url);
                            } else {
                              throw new Error("Empty transcript received");
                            }
                          });
                      })
                      .catch(error => {
                        console.error("Error fetching transcription:", error);
                        setFileError(`Error: ${error.message}`);
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
              };
              
              ws.onclose = () => {
                console.log("WebSocket connection closed");
              };
            } else if (response.transcript && response.download_url) {
              // Direct response with transcript (legacy mode)
              onUploadResponse(response.transcript, response.download_url);
            } else {
              throw new Error("Invalid response format from server");
            }
          } catch (error) {
            console.error("Error parsing response:", error);
            setFileError(`Server error: ${error instanceof Error ? error.message : 'Unknown error'}`);
            setIsProcessing(false);
          }
        } else {
          throw new Error(`Error: ${xhr.statusText}`);
        }
      };

      xhr.onerror = function() {
        setUploadXhr(null);
        setFileError("Network error occurred during upload");
        setIsUploading(false);
        setIsProcessing(false);
      };

      xhr.onabort = function() {
        console.log('Upload aborted');
        setIsUploading(false);
        setIsProcessing(false);
      };

      xhr.send(formData);
    } catch (error) {
      console.error('Upload failed:', error);
      setFileError(`Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsUploading(false);
      setIsProcessing(false);
      setUploadXhr(null);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-bold text-gray-200 mb-4">Upload Audio File</h2>
      <p className="text-gray-400 mb-6 text-center">
        Upload your audio or video file for transcription. WAV and MP3 formats are recommended for best results.
      </p>
      
      <div 
        className={`w-full border-2 border-dashed rounded-lg p-6 mb-4 cursor-pointer text-center transition-colors
          ${dragActive ? 'border-blue-500 bg-gray-700' : 'border-gray-600 hover:border-blue-400'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          id="fileInput"
          type="file"
          accept=".wav,.mp4,.mp3,.webm,.ogg"
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
          <p className="text-xs text-gray-500">Audio/video files only (MAX. 20MB)</p>
          <div className="mt-3 flex items-center text-xs text-blue-400">
            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            Recommended formats: .wav, .mp3 (best compatibility)
          </div>
        </div>
      </div>
      
      {fileError && (
        <div className="w-full bg-red-900/50 border border-red-700 rounded-lg p-3 mb-4 text-red-200 text-sm">
          <p>{fileError}</p>
        </div>
      )}
      
      {file && (
        <div className="w-full bg-gray-700 rounded-lg p-3 mb-4 flex items-center">
          <div className="bg-blue-900 rounded-lg p-2 mr-3">
            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
            </svg>
          </div>
          <div className="flex-grow truncate">
            <p className="text-sm font-medium truncate text-gray-200">{file.name}</p>
            <p className="text-xs text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setFile(null);
              clearTranscription();
              if (fileInputRef.current) {
                fileInputRef.current.value = '';
              }
            }}
            className="ml-2 text-gray-400 hover:text-red-400"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
      )}
      
      {/* Show cancel button only during upload */}
      {uploadXhr && (
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
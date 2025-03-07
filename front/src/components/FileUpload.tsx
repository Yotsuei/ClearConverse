// components/FileUpload.tsx
import React, { useState } from 'react';

interface FileUploadProps {
  onUploadResponse: (transcript: string, downloadUrl: string) => void;
  setLoading: (loading: boolean) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadResponse, setLoading }) => {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file first.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Update the URL if your API runs on a different port or path
      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      // The API returns a JSON with keys: audio_file, transcript, download_url
      onUploadResponse(data.transcript, data.download_url);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('There was an error uploading your file.');
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#334155] border border-gray-600 rounded-lg p-6 mb-6">
      <input
        type="file"
        accept=".mp3,.wav"
        onChange={handleFileChange}
        className="block w-full text-sm text-gray-300 border border-gray-500 rounded-lg cursor-pointer bg-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 mb-4 p-3"
      />
      <button
        onClick={handleUpload}
        className="w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 transform bg-[#14b8a6] hover:bg-[#0d9488] active:scale-95 shadow-lg"
      >
        Upload File
      </button>
    </div>
  );
};

export default FileUpload;

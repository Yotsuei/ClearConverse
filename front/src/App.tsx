// App.tsx
import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import TranscriptionDisplay from './components/TranscriptionDisplay';

const App: React.FC = () => {
  const [transcript, setTranscript] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUploadResponse = (transcript: string, downloadUrl: string) => {
    setTranscript(transcript);
    setDownloadUrl(downloadUrl);
    setLoading(false);
  };

  return (
    <div className="min-h-screen w-full bg-[#0f172a] flex flex-col items-center justify-center p-6 text-white">
      <div className="w-full max-w-3xl bg-[#1e293b] shadow-2xl rounded-2xl p-8">
        <h1 className="text-4xl font-extrabold text-center mb-8 tracking-tight">
          Overlapping Speech Transcription
        </h1>
        <FileUpload 
          onUploadResponse={handleUploadResponse} 
          setLoading={setLoading} 
        />
        {loading && (
          <div className="flex items-center justify-center mt-6">
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-blue-400"></div>
          </div>
        )}
        {transcript && downloadUrl && (
          <TranscriptionDisplay transcript={transcript} downloadUrl={downloadUrl} />
        )}
      </div>
    </div>
  );
};

export default App;
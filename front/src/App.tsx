// App.tsx
import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import ProgressBar from './components/ProgressBar';

const App: React.FC = () => {
  const [transcript, setTranscript] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleUploadResponse = (transcript: string, downloadUrl: string, audioBlob: Blob) => {
    setTranscript(transcript);
    setDownloadUrl(downloadUrl);
    setAudioBlob(audioBlob);
    setLoading(false);
  };

  const handleReset = () => {
    setTranscript(null);
    setDownloadUrl(null);
    setAudioBlob(null);
    setProgress(0);
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 to-slate-800 flex flex-col items-center justify-center p-4 md:p-8">
      <div className="w-full max-w-3xl bg-slate-800/50 backdrop-blur-lg shadow-2xl rounded-2xl p-6 md:p-8">
        <header className="mb-8 text-center">
          <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent mb-3">
            Overlapping Speech Transcription
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto">
            Upload audio files with overlapping speech and get accurate transcriptions with speaker identification.
          </p>
        </header>

        {!transcript && !downloadUrl ? (
          <>
            <FileUpload 
              onUploadResponse={handleUploadResponse} 
              setLoading={setLoading} 
              updateProgress={setProgress}
            />
            {loading && <ProgressBar progress={progress} />}
          </>
        ) : (
          transcript && downloadUrl && audioBlob && (
            <TranscriptionDisplay 
              transcript={transcript} 
              downloadUrl={downloadUrl} 
              audioBlob={audioBlob}
              onReset={handleReset}
            />
          )
        )}

        <footer className="mt-8 text-center text-xs text-slate-500">
          <p>Powered by advanced speech recognition and diarization technology</p>
          <p className="mt-1">© 2025 OverlappingSpeech-App</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
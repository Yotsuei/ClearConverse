import React, { useState } from 'react';
import './App.css';

interface TranscriptionResult {
  source1_transcription: string;
  source2_transcription: string;
}

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setTranscription(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/process-audio', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process audio');
      }

      const result = await response.json();
      setTranscription(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-500 to-teal-500 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-3xl bg-white shadow-2xl rounded-2xl p-8">
        <h1 className="text-4xl font-extrabold text-gray-800 mb-8 tracking-tight">SPEECH TRANSCRIPTION</h1>
        
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 mb-6">
          <input
            type="file"
            accept=".mp3,.wav"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-600 border border-gray-300 rounded-lg cursor-pointer bg-white focus:outline-none focus:ring-2 focus:ring-teal-400 focus:border-teal-400 mb-4 p-3"
          />
          <button
            onClick={handleUpload}
            className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 transform ${
              loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-teal-500 hover:bg-teal-600 active:scale-95 shadow-lg'
            }`}
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Upload and Transcribe'}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-6">
            {error}
          </div>
        )}

        {transcription && (
          <div className="bg-gray-100 border border-gray-200 rounded-lg p-6 mt-6">
            <h2 className="text-2xl font-bold text-gray-700 mb-4">Transcription Results</h2>
            
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Speaker 1</h3>
                <pre className="bg-white p-4 rounded-lg text-gray-800 whitespace-pre-wrap overflow-y-auto max-h-48 border border-gray-300 shadow-inner text-left">
                  {transcription.source1_transcription}
                </pre>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Speaker 2</h3>
                <pre className="bg-white p-4 rounded-lg text-gray-800 whitespace-pre-wrap overflow-y-auto max-h-48 border border-gray-300 shadow-inner text-left">
                  {transcription.source2_transcription}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
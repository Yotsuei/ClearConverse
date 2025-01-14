import React, { useState } from 'react';
import './App.css'; // Import Tailwind CSS

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
    setTranscription(null);

    setTimeout(() => {
      // Mock transcription result
      const mockTranscription = `
        Speaker 1: Hello, how are you?
        Speaker 2: I'm doing great, thank you! How about you?
      `;

      setTranscription(mockTranscription);
      setLoading(false);
    }, 2000); // Simulate a 2-second processing delay
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
            className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 transform ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-teal-500 hover:bg-teal-600 active:scale-95 shadow-lg'}`}
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Upload and Transcribe'}
          </button>
        </div>

        {transcription && (
          <div className="bg-gray-100 border border-gray-200 rounded-lg p-6 mt-6">
            <h2 className="text-2xl font-bold text-gray-700 mb-4">Transcription Result</h2>
            <pre className="bg-white p-4 rounded-lg text-gray-800 whitespace-pre-wrap overflow-y-auto max-h-96 border border-gray-300 shadow-inner text-left">{transcription}</pre>
          </div>
        )}
      </div>

    </div>
  );
};

export default App;
import React, { useState } from 'react';
import './App.css'; // Import Tailwind CSS

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showTranscription, setShowTranscription] = useState(false);

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
    setShowTranscription(false);

    // Simulate a delay to mimic processing time
    setTimeout(() => {
      const mockTranscription = `
        Speaker 1: Hello, how are you?
        Speaker 2: I'm doing great, thank you! How about you?
        Speaker 1: I'm good too. Let's discuss the project.
      `;

      setTranscription(mockTranscription);
      setLoading(false);
    }, 2000); // Simulate a 2-second processing delay
  };

  return (
    <div className="min-h-screen w-full bg-[#0f172a] flex flex-col items-center justify-center p-6 text-white">
      <div className="w-full max-w-3xl bg-[#1e293b] shadow-2xl rounded-2xl p-8">
        <h1 className="text-4xl font-extrabold text-center mb-8 tracking-tight">Overlapping Speech Transcription</h1>
        <div className="bg-[#334155] border border-gray-600 rounded-lg p-6 mb-6">
          <input
            type="file"
            accept=".mp3,.wav"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-300 border border-gray-500 rounded-lg cursor-pointer bg-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 mb-4 p-3"
          />
          <button
            onClick={handleUpload}
            className={`w-full py-3 px-5 text-white font-bold rounded-lg transition-all duration-300 transform ${loading ? 'bg-gray-600 cursor-not-allowed' : 'bg-[#14b8a6] hover:bg-[#0d9488] active:scale-95 shadow-lg'}`}
            disabled={loading}
          >
            {loading ? 'Uploading...' : 'Upload File'}
          </button>
        </div>

        {loading && (
          <div className="flex items-center justify-center mt-6">
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-blue-400"></div>
          </div>
        )}

        {transcription && !showTranscription && (
          <button
            onClick={() => setShowTranscription(true)}
            className="w-full mt-6 py-3 px-5 text-white font-bold rounded-lg bg-[#0ea5e9] hover:bg-[#0284c7] transition-all duration-300 shadow-lg"
          >
            Show Transcription
          </button>
        )}

        {showTranscription && transcription && (
          <div className="bg-[#334155] border border-gray-600 rounded-lg p-6 mt-6">
            <h2 className="text-2xl font-bold mb-4">Transcription Result</h2>
            <pre className="bg-gray-900 p-4 rounded-lg text-gray-300 whitespace-pre-wrap overflow-y-auto max-h-96 border border-gray-600 shadow-inner">{transcription}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
// components/TranscriptionDisplay.tsx
import React, { useState } from 'react';

interface TranscriptionDisplayProps {
  transcript: string;
  downloadUrl: string;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ transcript, downloadUrl }) => {
  const [showTranscription, setShowTranscription] = useState(false);

  return (
    <>
      {!showTranscription && (
        <button
          onClick={() => setShowTranscription(true)}
          className="w-full mt-6 py-3 px-5 text-white font-bold rounded-lg bg-[#0ea5e9] hover:bg-[#0284c7] transition-all duration-300 shadow-lg"
        >
          Show Transcription
        </button>
      )}
      {showTranscription && (
        <div className="bg-[#334155] border border-gray-600 rounded-lg p-6 mt-6">
          <h2 className="text-2xl font-bold mb-4">Transcription Result</h2>
          <pre className="bg-gray-900 p-4 rounded-lg text-gray-300 whitespace-pre-wrap overflow-y-auto max-h-96 border border-gray-600 shadow-inner">
            {transcript}
          </pre>
          <a
            href={`http://localhost:8000${downloadUrl}`}
            download="transcript.txt"
            className="mt-4 inline-block py-2 px-4 bg-green-500 hover:bg-green-600 rounded-lg font-bold transition-colors"
          >
            Download Transcript
          </a>
        </div>
      )}
    </>
  );
};

export default TranscriptionDisplay;
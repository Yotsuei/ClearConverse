// components/TranscriptionDisplay.tsx
import React, { useState } from 'react';

interface TranscriptionDisplayProps {
  transcript: string;
  downloadUrl: string;
  onClear: () => void;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ 
  transcript, 
  downloadUrl,
  onClear
}) => {
  const [expanded, setExpanded] = useState(false);

  // Parse the transcript to highlight speaker segments
  const formatTranscript = (text: string) => {
    // Split by speaker tags [SPEAKER_X]
    const parts = text.split(/(\[SPEAKER_[A-Z]\])/g);
    
    return parts.map((part, index) => {
      if (part.match(/\[SPEAKER_[A-Z]\]/)) {
        const speaker = part.replace(/[\[\]]/g, '');
        // Assign different colors based on speaker
        const color = speaker === 'SPEAKER_A' ? 'text-blue-600' : 'text-green-600';
        return (
          <span key={index} className={`font-semibold ${color}`}>
            {part.replace('SPEAKER_A', 'Speaker A').replace('SPEAKER_B', 'Speaker B')}
          </span>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 mt-6 shadow-sm">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Transcription Result</h2>
      
      <div className={`bg-gray-50 p-4 rounded-lg text-gray-700 border border-gray-200 whitespace-pre-wrap ${expanded ? '' : 'max-h-80'} overflow-y-auto transition-all duration-300 shadow-inner`}>
        {formatTranscript(transcript)}
      </div>
      
      {/* Show more/less button */}
      {transcript.length > 500 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-blue-600 hover:text-blue-800 text-sm font-medium"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
      
      <div className="mt-6 flex flex-wrap gap-3">
        <a
          href={`http://localhost:8000${downloadUrl}`}
          download="transcript.txt"
          className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
          </svg>
          Download Transcript
        </a>
        
        <button
          onClick={onClear}
          className="flex items-center px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
          Clear & Reset
        </button>
        
        <button
          onClick={() => {
            navigator.clipboard.writeText(transcript);
            // Could add a toast notification here
          }}
          className="flex items-center px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"></path>
          </svg>
          Copy to Clipboard
        </button>
      </div>
      
      {/* Stats section */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
          <p className="text-xs text-blue-600 font-medium">Word Count</p>
          <p className="text-2xl font-bold text-blue-800">{transcript.split(/\s+/).length}</p>
        </div>
        
        <div className="bg-green-50 p-3 rounded-lg border border-green-100">
          <p className="text-xs text-green-600 font-medium">Duration</p>
          <p className="text-2xl font-bold text-green-800">
            {transcript.match(/\d+\.\d+s/g)?.pop()?.replace('s', '') || "N/A"}
          </p>
        </div>
        
        <div className="bg-purple-50 p-3 rounded-lg border border-purple-100">
          <p className="text-xs text-purple-600 font-medium">Speaker A Turns</p>
          <p className="text-2xl font-bold text-purple-800">
            {(transcript.match(/\[SPEAKER_A\]/g) || []).length}
          </p>
        </div>
        
        <div className="bg-orange-50 p-3 rounded-lg border border-orange-100">
          <p className="text-xs text-orange-600 font-medium">Speaker B Turns</p>
          <p className="text-2xl font-bold text-orange-800">
            {(transcript.match(/\[SPEAKER_B\]/g) || []).length}
          </p>
        </div>
      </div>
    </div>
  );
};

export default TranscriptionDisplay;
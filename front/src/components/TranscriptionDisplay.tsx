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
  const [copySuccess, setCopySuccess] = useState<string | null>(null);

  // Parse the transcript to highlight speaker segments
  const formatTranscript = (text: string) => {
    // Check if the transcript includes speaker tags like [SPEAKER_X]
    const hasSpeakerTags = /\[SPEAKER_[A-Z]\]/g.test(text);
    
    if (hasSpeakerTags) {
      // Split by speaker tags [SPEAKER_X]
      const parts = text.split(/(\[SPEAKER_[A-Z]\])/g);
      
      return parts.map((part, index) => {
        if (part.match(/\[SPEAKER_[A-Z]\]/)) {
          const speaker = part.replace(/[\[\]]/g, '');
          // Assign different colors based on speaker
          const color = speaker === 'SPEAKER_A' ? 'text-blue-400' : 
                       speaker === 'SPEAKER_B' ? 'text-gray-300' : 
                       speaker === 'SPEAKER_C' ? 'text-gray-400' : 'text-gray-500';
          return (
            <span key={index} className={`font-semibold ${color}`}>
              {part.replace('SPEAKER_A', 'Speaker A')
                  .replace('SPEAKER_B', 'Speaker B')
                  .replace('SPEAKER_C', 'Speaker C')
                  .replace('SPEAKER_D', 'Speaker D')}
            </span>
          );
        }
        return <span key={index}>{part}</span>;
      });
    } else {
      // If no speaker tags, add paragraph breaks for better readability
      return text.split('\n').map((paragraph, i) => (
        <p key={i} className={i > 0 ? 'mt-4' : ''}>{paragraph}</p>
      ));
    }
  };

  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(transcript).then(
      () => {
        setCopySuccess('Copied!');
        setTimeout(() => setCopySuccess(null), 2000);
      },
      () => {
        setCopySuccess('Failed to copy');
        setTimeout(() => setCopySuccess(null), 2000);
      }
    );
  };

  // Calculate statistics
  const wordCount = transcript.split(/\s+/).length;
  const speakerATurns = (transcript.match(/\[SPEAKER_A\]/g) || []).length;
  const speakerBTurns = (transcript.match(/\[SPEAKER_B\]/g) || []).length;
  const durationMatch = transcript.match(/(\d+\.\d+)s/);
  const duration = durationMatch ? parseFloat(durationMatch[1]).toFixed(1) : "N/A";

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mt-6 shadow-sm">
      <h2 className="text-2xl font-bold mb-4 text-gray-200">Transcription Result</h2>
      
      <div 
        className={`bg-gray-750 p-4 rounded-lg text-gray-300 border border-gray-700 whitespace-pre-wrap overflow-y-auto transition-all duration-300 shadow-inner ${
          expanded ? 'max-h-[600px]' : 'max-h-80'
        }`}
      >
        {formatTranscript(transcript)}
      </div>
      
      {/* Show more/less button */}
      {transcript.length > 500 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-blue-400 hover:text-blue-300 text-sm font-medium inline-flex items-center"
        >
          {expanded ? (
            <>
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7"></path>
              </svg>
              Show less
            </>
          ) : (
            <>
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
              </svg>
              Show more
            </>
          )}
        </button>
      )}
      
      <div className="mt-6 flex flex-wrap gap-3">
        <a
          href={`http://localhost:8000${downloadUrl}`}
          download="transcript.txt"
          className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
          </svg>
          Download Transcript
        </a>
        
        <button
          onClick={onClear}
          className="flex items-center px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
          Clear & Reset
        </button>
        
        <button
          onClick={handleCopyToClipboard}
          className="flex items-center px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors shadow-sm relative"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"></path>
          </svg>
          {copySuccess || "Copy to Clipboard"}
          
          {copySuccess && (
            <span className="absolute top-0 right-0 -mt-2 -mr-2 bg-green-600 text-white text-xs px-2 py-1 rounded-full">
              {copySuccess}
            </span>
          )}
        </button>
      </div>
      
      {/* Stats section */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-700 p-3 rounded-lg border border-gray-600">
          <p className="text-xs text-blue-400 font-medium">Word Count</p>
          <p className="text-2xl font-bold text-gray-200">{wordCount}</p>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-lg border border-gray-600">
          <p className="text-xs text-blue-400 font-medium">Duration (sec)</p>
          <p className="text-2xl font-bold text-gray-200">{duration}</p>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-lg border border-gray-600">
          <p className="text-xs text-blue-400 font-medium">Speaker A Turns</p>
          <p className="text-2xl font-bold text-gray-200">{speakerATurns || 'N/A'}</p>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-lg border border-gray-600">
          <p className="text-xs text-blue-400 font-medium">Speaker B Turns</p>
          <p className="text-2xl font-bold text-gray-200">{speakerBTurns || 'N/A'}</p>
        </div>
      </div>

      {/* Additional information */}
      <div className="mt-6 bg-gray-750 p-4 rounded-lg border border-gray-700">
        <h3 className="text-lg font-semibold mb-2 text-gray-200">About This Transcription</h3>
        <p className="text-gray-400 text-sm">
          This transcription was generated using advanced speech recognition technology. 
          {speakerATurns > 0 && speakerBTurns > 0 ? 
            " Speaker diarization was applied to identify different speakers in the conversation." : 
            " The audio appears to have a single speaker."}
          {duration !== "N/A" ? 
            ` Total audio duration was approximately ${duration} seconds.` : 
            ""}
        </p>
      </div>
    </div>
  );
};

export default TranscriptionDisplay;
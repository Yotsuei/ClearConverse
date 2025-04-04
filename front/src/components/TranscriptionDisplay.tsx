import React, { useState } from 'react';

interface TranscriptionDisplayProps {
  transcript: string;
  downloadUrl: string;
  onClear: () => void;
  onReset: () => void;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ 
  transcript, 
  downloadUrl,
  onClear,
  onReset
}) => {
  const [expanded, setExpanded] = useState(false);
  const [copySuccess, setCopySuccess] = useState<string | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState<string | null>(null);

  // Parse the transcript to highlight speaker segments
  const formatTranscript = (text: string) => {
    if (!text) return <p>No transcription available.</p>;
    
    // Check if the transcript includes speaker tags like [SPEAKER_X]
    const hasSpeakerTags = /\[(SPEAKER_[A-Z]|[A-Z]+)\]/g.test(text);
    
    if (hasSpeakerTags) {
      // Split by speaker tags [SPEAKER_X] or other bracketed tags
      const parts = text.split(/(\[[A-Z_]+\])/g);
      
      return parts.map((part, index) => {
        if (part.match(/\[[A-Z_]+\]/)) {
          const speaker = part.replace(/[\[\]]/g, '');
          // Assign different colors based on speaker
          const color = speaker.includes('SPEAKER_A') ? 'text-blue-400' : 
                       speaker.includes('SPEAKER_B') ? 'text-green-300' : 
                       speaker.includes('SPEAKER_C') ? 'text-yellow-300' : 
                       speaker.includes('SPEAKER_D') ? 'text-purple-300' : 'text-gray-400';
          
          return (
            <span key={index} className={`font-semibold ${color}`}>
              {part.replace('SPEAKER_A', 'Speaker A')
                  .replace('SPEAKER_B', 'Speaker B')
                  .replace('SPEAKER_C', 'Speaker C')
                  .replace('SPEAKER_D', 'Speaker D')}
            </span>
          );
        }
        // Add line breaks for timestamps
        if (part.includes('s - ')) {
          return <span key={index}>{part}<br /></span>;
        }
        return <span key={index}>{part}</span>;
      });
    } else {
      // If no speaker tags, add paragraph breaks for better readability
      return text.split('\n').map((paragraph, i) => (
        paragraph.trim() ? <p key={i} className={i > 0 ? 'mt-4' : ''}>{paragraph}</p> : <br key={i} />
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

  const handleClearConfirm = () => {
    setShowConfirmDialog('clear');
  };

  const handleResetConfirm = () => {
    setShowConfirmDialog('reset');
  };

  const handleDialogCancel = () => {
    setShowConfirmDialog(null);
  };

  const handleActionConfirmed = () => {
    if (showConfirmDialog === 'clear') {
      onClear();
    } else if (showConfirmDialog === 'reset') {
      onReset();
    }
    setShowConfirmDialog(null);
  };

  // Calculate statistics
  const wordCount = transcript.split(/\s+/).filter(word => word.trim().length > 0).length;
  const speakerATurns = (transcript.match(/\[SPEAKER_A\]/g) || []).length;
  const speakerBTurns = (transcript.match(/\[SPEAKER_B\]/g) || []).length;
  
  // Try to extract duration from transcript
  const durationMatch = transcript.match(/(\d+\.\d+)s/);
  const duration = durationMatch ? parseFloat(durationMatch[1]).toFixed(1) : "N/A";

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-200">Transcription Result</h2>
        
        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleClearConfirm}
            className="flex items-center text-yellow-400 hover:text-yellow-300 transition-colors"
            title="Clear transcription but keep audio"
          >
            <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
            </svg>
            Clear
          </button>
          
          <button
            onClick={handleResetConfirm}
            className="flex items-center text-red-400 hover:text-red-300 transition-colors"
            title="Reset everything (audio and transcription)"
          >
            <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            Reset All
          </button>
        </div>
      </div>
      
      {/* Confirmation modal for actions */}
      {showConfirmDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-lg max-w-md">
            <h3 className="text-xl font-bold mb-4 text-gray-200">
              {showConfirmDialog === 'clear' ? 'Clear Transcription' : 'Reset Everything'}
            </h3>
            <p className="text-gray-300 mb-6">
              {showConfirmDialog === 'clear' 
                ? 'Are you sure you want to clear the transcription? The audio file will be kept.' 
                : 'Are you sure you want to reset everything? This will clear both the transcription and audio file.'}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={handleDialogCancel}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleActionConfirmed}
                className={`px-4 py-2 ${
                  showConfirmDialog === 'clear' 
                    ? 'bg-yellow-600 hover:bg-yellow-700' 
                    : 'bg-red-600 hover:bg-red-700'
                } text-white font-medium rounded-lg transition-colors`}
              >
                {showConfirmDialog === 'clear' ? 'Yes, Clear' : 'Yes, Reset'}
              </button>
            </div>
          </div>
        </div>
      )}
      
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
        
        {/* Clear button (keeps audio) */}
        <button
          onClick={handleClearConfirm}
          className="flex items-center px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
          </svg>
          Clear Transcription
        </button>
        
        {/* Reset button (clears everything) */}
        <button
          onClick={handleResetConfirm}
          className="flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
          </svg>
          Reset Everything
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
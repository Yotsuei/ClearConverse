// components/ClearButton.tsx
import React, { useState } from 'react';

interface ClearButtonProps {
  onClear: () => void;
  isProcessing: boolean;
<<<<<<< HEAD
  includeAudio?: boolean; // New prop to indicate if audio should be cleared too
}

const ClearButton: React.FC<ClearButtonProps> = ({ 
  onClear, 
  isProcessing,
  includeAudio = true // Default to true to clear everything
}) => {
=======
}

const ClearButton: React.FC<ClearButtonProps> = ({ onClear, isProcessing }) => {
>>>>>>> origin/UI
  const [showConfirm, setShowConfirm] = useState(false);
  
  const handleClearClick = () => {
    if (isProcessing) {
      // Show confirmation if processing is in progress
      setShowConfirm(true);
    } else {
      // Otherwise clear directly
      onClear();
    }
  };
  
  const handleConfirmClear = () => {
    setShowConfirm(false);
    onClear();
  };
  
  const handleCancelClear = () => {
    setShowConfirm(false);
  };

  return (
    <>
      <button
        onClick={handleClearClick}
        className={`p-3 rounded-full shadow-lg flex items-center justify-center
          ${isProcessing 
            ? 'bg-red-600 hover:bg-red-700' 
            : 'bg-gray-700 hover:bg-gray-600'} 
          text-white transition-all hover:scale-105`}
<<<<<<< HEAD
        aria-label={includeAudio ? "Clear everything" : "Clear transcription"}
        title={includeAudio ? "Clear everything (audio and transcription)" : "Clear transcription only"}
=======
        aria-label="Clear current audio"
        title="Clear current audio (stay in same mode)"
>>>>>>> origin/UI
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
        </svg>
      </button>
      
      {/* Confirmation dialog */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-lg max-w-md mx-4">
            <h3 className="text-xl font-bold mb-4 text-gray-200">
<<<<<<< HEAD
              {isProcessing ? "Interrupt Processing?" : (includeAudio ? "Clear Everything?" : "Clear Transcription?")}
=======
              {isProcessing ? "Interrupt Processing?" : "Clear Current Audio?"}
>>>>>>> origin/UI
            </h3>
            <p className="text-gray-300 mb-6">
              {isProcessing 
                ? "Are you sure you want to cancel the ongoing transcription process? All progress will be lost."
<<<<<<< HEAD
                : (includeAudio 
                  ? "Are you sure you want to clear both the audio and transcription data? This action cannot be undone."
                  : "Are you sure you want to clear the transcription? The audio file will be kept.")}
=======
                : "Are you sure you want to clear the current audio? This will remove the current audio file but keep you in the same mode."}
>>>>>>> origin/UI
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelClear}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmClear}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
              >
                {isProcessing ? "Yes, Interrupt" : "Yes, Clear"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ClearButton;
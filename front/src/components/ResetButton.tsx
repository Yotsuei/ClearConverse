// components/ResetButton.tsx
import React, { useState } from 'react';

interface ResetButtonProps {
  onReset: () => void;
  isProcessing: boolean;
}

const ResetButton: React.FC<ResetButtonProps> = ({ onReset, isProcessing }) => {
  const [showConfirm, setShowConfirm] = useState(false);
  
  const handleResetClick = () => {
    if (isProcessing) {
      // Show confirmation if processing is in progress
      setShowConfirm(true);
    } else {
      // Otherwise reset directly
      onReset();
    }
  };
  
  const handleConfirmReset = () => {
    setShowConfirm(false);
    onReset();
  };
  
  const handleCancelReset = () => {
    setShowConfirm(false);
  };

  return (
    <>
      <div className="fixed bottom-6 right-6 z-10">
        <button
          onClick={handleResetClick}
          className={`p-3 rounded-full shadow-lg flex items-center justify-center
            ${isProcessing 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-gray-700 hover:bg-gray-600'} 
            text-white transition-all hover:scale-105`}
          aria-label="Reset transcription"
          title="Reset transcription"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
          </svg>
        </button>
      </div>
      
      {/* Confirmation dialog */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-lg max-w-md mx-4">
            <h3 className="text-xl font-bold mb-4 text-gray-200">
              {isProcessing ? "Interrupt Processing?" : "Reset Transcription?"}
            </h3>
            <p className="text-gray-300 mb-6">
              {isProcessing 
                ? "Are you sure you want to cancel the ongoing transcription process? All progress will be lost."
                : "Are you sure you want to reset? This will clear all current data."}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelReset}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmReset}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
              >
                {isProcessing ? "Yes, Interrupt" : "Yes, Reset"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ResetButton;
// components/ResetButton.tsx
import React, { useState } from 'react';
import FloatingActionButton from './FloatingActionButton';

interface ResetButtonProps {
  onReset: () => void;
  onClear?: () => void;
  isProcessing: boolean;
  hasTranscription?: boolean;
}

const ResetButton: React.FC<ResetButtonProps> = ({ 
  onReset, 
  onClear, 
  isProcessing,
  hasTranscription
}) => {
  const [showConfirm, setShowConfirm] = useState<'reset' | 'clear' | null>(null);
  
  const handleResetClick = () => {
    if (isProcessing) {
      // Show confirmation if processing is in progress
      setShowConfirm('reset');
    } else {
      // Otherwise reset directly
      onReset();
    }
  };
  
  const handleClearClick = () => {
    if (onClear) {
      setShowConfirm('clear');
    }
  };
  
  const handleConfirmAction = () => {
    if (showConfirm === 'reset') {
      onReset();
    } else if (showConfirm === 'clear' && onClear) {
      onClear();
    }
    setShowConfirm(null);
  };
  
  const handleCancelAction = () => {
    setShowConfirm(null);
  };

  const resetIcon = (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
    </svg>
  );

  const clearIcon = (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
    </svg>
  );

  return (
    <>
      <div className="fixed bottom-6 right-6 z-10 flex flex-col gap-3">
        {/* Clear button only appears when we have a transcription */}
        {hasTranscription && onClear && (
          <FloatingActionButton
            onClick={handleClearClick}
            icon={clearIcon}
            label="Reset"
            color="yellow"
            tooltip="Reset everything"
          />
        )}
        
        {/* Reset button always appears */}
        <FloatingActionButton
          onClick={handleResetClick}
          icon={resetIcon}
          label="Reset everything"
          color={isProcessing ? "red" : "gray"}
          tooltip="Reset everything"
        />
      </div>
      
      {/* Confirmation dialog */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-lg max-w-md mx-4">
            <h3 className="text-xl font-bold mb-4 text-gray-200">
              Reset Everything
            </h3>
            <p className="text-gray-300 mb-6">
              {isProcessing 
                ? "Are you sure you want to cancel the ongoing transcription process? All progress will be lost and files will be cleaned up from the server."
                : "Are you sure you want to reset? This will clear all current data including audio and transcription files from the server."}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelAction}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmAction}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
              >
                {isProcessing ? "Yes, Cancel & Reset" : "Yes, Reset"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ResetButton;
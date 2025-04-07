// components/ProgressBar.tsx
import React from 'react';

interface ProgressBarProps {
  progress: number;
  type: 'upload' | 'processing';
  onCancel?: () => void;
  message?: string; // Added message prop for more detailed status
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress, type, onCancel, message }) => {
  // Determine if progress bar should show indeterminate loading animation
  const isIndeterminate = progress < 10;
  
  // Determine the stage based on progress percentage and type
  const getStage = () => {
    if (message) {
      return message; // Use the provided message if available
    }
    
    if (type === 'upload') {
      return "Uploading file...";
    } else { // processing
      return "Processing in progress...";
    }
  };

  // Get primary and secondary colors based on type
  const getBgColor = () => {
    return type === 'upload' ? 'bg-blue-600' : 'bg-blue-500';
  };
  
  const getLightBgColor = () => {
    return type === 'upload' ? 'bg-blue-900' : 'bg-blue-900';
  };
  
  const getBorderColor = () => {
    return type === 'upload' ? 'border-blue-800' : 'border-blue-800';
  };
  
  const getTextColor = () => {
    return type === 'upload' ? 'text-blue-300' : 'text-blue-300';
  };

  return (
    <div className={`w-full p-4 rounded-lg ${getLightBgColor()} border ${getBorderColor()} mb-6`}>
      <div className="mb-2 flex justify-between items-center">
        <div className={`text-sm font-medium ${getTextColor()}`}>{getStage()}</div>
        <div className="flex items-center gap-3">
          <div className={`text-sm font-medium ${getTextColor()}`}>
            {isIndeterminate ? '' : `${Math.round(progress)}%`}
          </div>
          
          {/* Cancel button */}
          {onCancel && progress < 100 && (
            <button 
              onClick={onCancel}
              className="text-red-400 hover:text-red-300 text-sm font-medium flex items-center gap-1"
              aria-label="Cancel transcription"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
              Cancel
            </button>
          )}
        </div>
      </div>
      <div className="relative w-full h-2.5 bg-gray-700 rounded-full overflow-hidden">
        {/* Indeterminate animation for initial loading or error states */}
        {isIndeterminate ? (
          <div 
            className={`absolute top-0 bottom-0 left-0 ${getBgColor()}`}
            style={{ 
              width: '30%',
              animation: 'indeterminate 1.5s infinite ease-in-out',
            }}
          />
        ) : (
          <>
            {/* Background pulse animation when in progress */}
            <div 
              className={`absolute inset-0 ${getBgColor()} opacity-30`}
              style={{ 
                animation: progress < 100 ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none',
              }}
            />
            
            {/* Actual progress bar */}
            <div 
              className={`h-full ${getBgColor()} rounded-full transition-all duration-300 ease-out`}
              style={{ width: `${progress}%` }}
            />
          </>
        )}
      </div>
      
      {/* Error message block - show in red if message contains "error" */}
      {/* Error messaging removed */}
      
      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 0.3;
          }
          50% {
            opacity: 0.6;
          }
        }
        
        @keyframes indeterminate {
          0% {
            left: -30%;
          }
          100% {
            left: 100%;
          }
        }
      `}</style>
    </div>
  );
};

export default ProgressBar;
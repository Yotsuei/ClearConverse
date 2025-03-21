// components/ProgressBar.tsx
import React from 'react';

interface ProgressBarProps {
  progress: number;
  type: 'upload' | 'processing';
  onCancel?: () => void;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress, type, onCancel }) => {
  // Determine the stage based on progress percentage and type
  const getStage = () => {
    if (type === 'upload') {
      if (progress < 25) return "Starting upload...";
      if (progress < 50) return "Uploading file...";
      if (progress < 75) return "Verifying upload...";
      if (progress < 90) return "Upload complete...";
      return "Preparing for processing...";
    } else { // processing
      if (progress < 25) return "Analyzing audio...";
      if (progress < 50) return "Processing speech...";
      if (progress < 75) return "Generating transcription...";
      if (progress < 90) return "Finalizing results...";
      return "Transcription complete!";
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
          <div className={`text-sm font-medium ${getTextColor()}`}>{Math.round(progress)}%</div>
          
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
      </div>
      
      {/* Processing stages */}
      <div className="mt-4 grid grid-cols-4 gap-2">
        <Stage 
          title={type === 'upload' ? "Upload" : "Analyze"}
          isActive={progress > 0}
          isComplete={progress >= 25}
          type={type}
        />
        <Stage 
          title={type === 'upload' ? "Verify" : "Process"}
          isActive={progress >= 25}
          isComplete={progress >= 50}
          type={type}
        />
        <Stage 
          title={type === 'upload' ? "Complete" : "Generate"}
          isActive={progress >= 50}
          isComplete={progress >= 75}
          type={type}
        />
        <Stage 
          title="Finalize"
          isActive={progress >= 75}
          isComplete={progress >= 100}
          type={type}
        />
      </div>
      
      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 0.3;
          }
          50% {
            opacity: 0.6;
          }
        }
      `}</style>
    </div>
  );
};

interface StageProps {
  title: string;
  isActive: boolean;
  isComplete: boolean;
  type: 'upload' | 'processing';
}

const Stage: React.FC<StageProps> = ({ title, isActive, isComplete, type }) => {
  const getGradient = () => {
    return type === 'upload' 
      ? 'bg-gradient-to-r from-blue-600 to-blue-700' 
      : 'bg-gradient-to-r from-blue-500 to-blue-600';
  };
  
  const getAnimationColor = () => {
    return type === 'upload' ? 'bg-blue-600' : 'bg-blue-500';
  };
  
  const getTextColor = () => {
    return type === 'upload' ? 'text-blue-400' : 'text-blue-400';
  };

  return (
    <div className="flex flex-col items-center">
      <div 
        className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 transition-colors
          ${isComplete ? getGradient() + ' text-white' : 
            isActive ? getAnimationColor() + ' text-white animate-pulse' : 'bg-gray-700 text-gray-500'}`}
      >
        {isComplete ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        ) : (
          <span className="text-xs">{title.charAt(0)}</span>
        )}
      </div>
      <span className={`text-xs ${isActive ? getTextColor() : 'text-gray-500'}`}>
        {title}
      </span>
    </div>
  );
};

export default ProgressBar;
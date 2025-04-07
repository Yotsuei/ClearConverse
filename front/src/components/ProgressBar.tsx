// components/ProgressBar.tsx
import React from 'react';

interface ProgressBarProps {
  progress: number;
  type: 'upload' | 'processing';
  onCancel?: () => void;
  message?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ 
  progress, 
  type, 
  onCancel, 
  message 
}) => {
  // Determine if progress bar should show indeterminate loading animation
  const isIndeterminate = progress < 10;
  
  // Generate default stage message if none provided
  const getDefaultStageMessage = () => {
    if (type === 'upload') {
      if (progress < 25) return "Starting upload...";
      if (progress < 50) return "Uploading file...";
      if (progress < 75) return "Verifying upload...";
      if (progress < 90) return "Upload complete...";
      return "Preparing for processing...";
    } else { 
      if (progress < 10) return "Initializing transcription...";
      if (progress < 30) return "Building speaker profiles...";
      if (progress < 50) return "Detecting speech segments...";
      if (progress < 70) return "Processing overlapping speech...";
      if (progress < 85) return "Generating transcription...";
      if (progress < 95) return "Finalizing results...";
      return "Transcription complete!";
    }
  };

  // Get display message from prop or generate default
  const displayMessage = message || getDefaultStageMessage();

  // Style configuration based on type
  const styles = {
    container: `w-full p-4 rounded-lg ${type === 'upload' ? 'bg-blue-900' : 'bg-blue-900'} border ${type === 'upload' ? 'border-blue-800' : 'border-blue-800'} mb-6`,
    text: `text-sm font-medium ${type === 'upload' ? 'text-blue-300' : 'text-blue-300'}`,
    progressBar: `absolute top-0 bottom-0 left-0 ${type === 'upload' ? 'bg-blue-600' : 'bg-blue-500'}`,
    pulseBg: `absolute inset-0 ${type === 'upload' ? 'bg-blue-600' : 'bg-blue-500'} opacity-30`
  };

  return (
    <div className={styles.container}>
      <div className="mb-2 flex justify-between items-center">
        <div className={styles.text}>{displayMessage}</div>
        <div className="flex items-center gap-3">
          <div className={styles.text}>
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
        {/* Indeterminate animation for initial loading */}
        {isIndeterminate ? (
          <div 
            className={styles.progressBar}
            style={{ 
              width: '30%',
              animation: 'indeterminate 1.5s infinite ease-in-out',
            }}
          />
        ) : (
          <>
            {/* Background pulse animation when in progress */}
            <div 
              className={styles.pulseBg}
              style={{ 
                animation: progress < 100 ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none',
              }}
            />
            
            {/* Actual progress bar */}
            <div 
              className={`h-full ${styles.progressBar} rounded-full transition-all duration-300 ease-out`}
              style={{ width: `${progress}%` }}
            />
          </>
        )}
      </div>
      
      {/* CSS Animations */}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }
        
        @keyframes indeterminate {
          0% { left: -30%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
};

export default ProgressBar;
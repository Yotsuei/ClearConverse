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
    // If we have an explicit message about cancellation, use it with priority
    if (message && message.toLowerCase().includes('cancel')) {
      return message;
    }
  
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
    container: `w-full p-4 rounded-lg ${
      message && message.toLowerCase().includes('cancel') 
        ? 'bg-red-900 border-red-800' 
        : (type === 'upload' ? 'bg-blue-900 border-blue-800' : 'bg-blue-900 border-blue-800')
    } border mb-6`,
    text: `text-sm font-medium ${
      message && message.toLowerCase().includes('cancel') 
        ? 'text-red-300' 
        : (type === 'upload' ? 'text-blue-300' : 'text-blue-300')
    }`,
    progressBar: `absolute top-0 bottom-0 left-0 ${
      message && message.toLowerCase().includes('cancel')
        ? 'bg-red-600'
        : (type === 'upload' ? 'bg-blue-600' : 'bg-blue-500')
    }`,
    pulseBg: `absolute inset-0 ${
      message && message.toLowerCase().includes('cancel')
        ? 'bg-red-600'
        : (type === 'upload' ? 'bg-blue-600' : 'bg-blue-500')
    } opacity-30`
  };  

  return (
    <div className={styles.container}>
      <div className="mb-2 flex justify-between items-center">
        <div className={styles.text}>{displayMessage}</div>
        <div className="flex items-center gap-3">
          <div className={styles.text}>
            {isIndeterminate ? '' : `${Math.round(progress)}%`}
          </div>
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
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.6; }
          }
          
          @keyframes indeterminate {
            0% { left: -30%; }
            100% { left: 100%; }
          }
        `
      }} />
    </div>
  );
};

export default ProgressBar;
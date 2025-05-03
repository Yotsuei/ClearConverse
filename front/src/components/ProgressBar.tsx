// components/ProgressBar.tsx
import React from 'react';

interface ProgressBarProps {
  type: 'upload' | 'processing';
  onCancel?: () => void;
  message?: string;
  isComplete?: boolean;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ 
  type, 
  onCancel, 
  message,
  isComplete = false
}) => {
  // Generate default stage message if none provided
  const getDefaultStageMessage = () => {
    // If we have an explicit message about cancellation, use it with priority
    if (message && message.toLowerCase().includes('cancel')) {
      return message;
    }
    
    // If we have a completion message, use it
    if (isComplete) {
      return "Transcription complete!";
    }
  
    if (type === 'upload') {
      return "Uploading file...";
    } else { 
      return "Processing audio file...";
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
    progressBarHighlight: `
      absolute top-0 bottom-0 ${
        message && message.toLowerCase().includes('cancel')
          ? 'bg-red-400'
          : (type === 'upload' ? 'bg-blue-400' : 'bg-blue-300')
      } opacity-70 w-20
    `
  };  

  return (
    <div className={styles.container}>
      <div className="mb-2 flex justify-between items-center">
        <div className={styles.text}>{displayMessage}</div>
      </div>
      
      <div className="relative w-full h-2.5 bg-gray-700 rounded-full overflow-hidden">
        {isComplete ? (
          // Complete state - full bar
          <div className={`h-full ${styles.progressBar} rounded-full w-full`} />
        ) : (
          // In progress state - animated passing through effect
          <>
            {/* Base progress bar - subtle background */}
            <div className={`h-full ${styles.progressBar} opacity-40 rounded-full w-full`} />
            
            {/* Animated highlight that passes through */}
            <div 
              className={`h-full ${styles.progressBarHighlight} progress-highlight rounded-full`}
              style={{ 
                animation: 'progressHighlight 1.5s ease-in-out infinite'
              }}
            />
          </>
        )}
      </div>
      
      {/* CSS Animations */}
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes progressHighlight {
            0% { left: -10%; }
            100% { left: 100%; }
          }
        `
      }} />
    </div>
  );
};

export default ProgressBar;
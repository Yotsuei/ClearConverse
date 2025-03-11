// components/ProgressBar.tsx
import React from 'react';

interface ProgressBarProps {
  progress: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  // Determine the stage based on progress percentage
  const getStage = () => {
    if (progress < 50) return "Uploading file...";
    if (progress < 75) return "Processing audio...";
    if (progress < 90) return "Generating transcription...";
    return "Finalizing results...";
  };

  return (
    <div className="w-full bg-slate-700/50 p-6 rounded-xl">
      <div className="mb-2 flex justify-between items-center">
        <div className="text-sm font-medium text-slate-300">{getStage()}</div>
        <div className="text-sm font-medium text-blue-400">{Math.round(progress)}%</div>
      </div>
      <div className="relative w-full h-2.5 bg-slate-700 rounded-full overflow-hidden">
        {/* Background pulse animation when in progress */}
        <div 
          className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-500 opacity-30"
          style={{ 
            animation: progress < 100 ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none',
          }}
        />
        
        {/* Actual progress bar */}
        <div 
          className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      
      {/* Processing stages */}
      <div className="mt-4 grid grid-cols-4 gap-2">
        <Stage 
          title="Upload" 
          isActive={progress > 0}
          isComplete={progress >= 50}
        />
        <Stage 
          title="Process" 
          isActive={progress >= 50}
          isComplete={progress >= 75}
        />
        <Stage 
          title="Transcribe" 
          isActive={progress >= 75}
          isComplete={progress >= 90}
        />
        <Stage 
          title="Complete" 
          isActive={progress >= 90}
          isComplete={progress >= 100}
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
}

const Stage: React.FC<StageProps> = ({ title, isActive, isComplete }) => {
  return (
    <div className="flex flex-col items-center">
      <div 
        className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 transition-colors
          ${isComplete ? 'bg-gradient-to-r from-blue-500 to-cyan-400 text-white' : 
            isActive ? 'bg-slate-600 text-white animate-pulse' : 'bg-slate-800 text-slate-500'}`}
      >
        {isComplete ? (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        ) : (
          <span className="text-xs">{title.charAt(0)}</span>
        )}
      </div>
      <span className={`text-xs ${isActive ? 'text-slate-300' : 'text-slate-500'}`}>
        {title}
      </span>
    </div>
  );
};

export default ProgressBar;
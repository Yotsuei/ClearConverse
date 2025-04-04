// components/AudioPlayer.tsx
import React, { useRef, useState, useEffect } from 'react';

interface AudioPlayerProps {
  audioUrl: string;
  onTranscribe?: () => void; // Now we'll show this button for URL sources
}

const AudioPlayer: React.FC<AudioPlayerProps> = ({ audioUrl, onTranscribe }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Reset audio player when URL changes
    setIsPlaying(false);
    setCurrentTime(0);
    setIsLoading(true);
    setLoadError(null);
    
    // Create a new Audio element to properly load and get metadata
    const audioElement = new Audio(audioUrl);
    
    audioElement.addEventListener('loadedmetadata', () => {
      if (audioRef.current) {
        audioRef.current.load(); // Reload the audio element
        setIsLoading(false);
      }
    });
    
    audioElement.addEventListener('error', () => {
      setLoadError("Failed to load audio. The URL might be invalid or inaccessible.");
      setIsLoading(false);
    });
    
    // Start loading
    audioElement.load();
    
    // Cleanup
    return () => {
      audioElement.remove();
    };
  }, [audioUrl]);

  const formatTime = (seconds: number): string => {
    if (isNaN(seconds) || !isFinite(seconds)) {
      return "0:00";
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
      setIsLoading(false);
    }
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSkipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(audioRef.current.currentTime + 10, duration);
    }
  };

  const handleSkipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(audioRef.current.currentTime - 10, 0);
    }
  };

  const handleProgressBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (progressBarRef.current && audioRef.current && duration > 0) {
      const rect = progressBarRef.current.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      audioRef.current.currentTime = pos * duration;
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow-sm">
      <h3 className="text-lg font-semibold mb-4 text-gray-200">Audio Preview</h3>

      {loadError ? (
        <div className="bg-red-900/50 text-red-200 p-4 rounded-lg mb-4 text-sm">
          <p className="font-medium mb-1">Error Loading Audio</p>
          <p>{loadError}</p>
        </div>
      ) : (
        <>
          <audio 
            ref={audioRef}
            src={audioUrl}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onEnded={handleAudioEnded}
            className="hidden"
            preload="metadata"
          />

          {isLoading ? (
            <div className="flex items-center justify-center h-20 bg-gray-750 rounded-lg mb-4">
              <div className="flex flex-col items-center">
                <svg className="animate-spin h-8 w-8 text-blue-500 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p className="text-sm text-gray-400">Loading audio...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Time display above progress bar */}
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>

              {/* Progress bar */}
              <div 
                ref={progressBarRef}
                className="w-full h-2 bg-gray-700 rounded-full mb-3 cursor-pointer"
                onClick={handleProgressBarClick}
              >
                <div 
                  className="h-2 bg-blue-600 rounded-full"
                  style={{ width: `${((currentTime / duration) * 100) || 0}%` }}
                ></div>
              </div>

              {/* Controls */}
              <div className="flex justify-center items-center gap-6 mb-4">
                <button 
                  onClick={handleSkipBackward}
                  className="w-10 h-10 flex items-center justify-center text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.333 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z"></path>
                  </svg>
                  <span className="sr-only">Backward 10s</span>
                </button>

                <button 
                  onClick={handlePlayPause}
                  className="w-14 h-14 rounded-full bg-blue-600 flex items-center justify-center text-white hover:bg-blue-700 transition-colors"
                >
                  {isPlaying ? (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                  ) : (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                  )}
                  <span className="sr-only">{isPlaying ? 'Pause' : 'Play'}</span>
                </button>

                <button 
                  onClick={handleSkipForward}
                  className="w-10 h-10 flex items-center justify-center text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z"></path>
                  </svg>
                  <span className="sr-only">Forward 10s</span>
                </button>
              </div>
            </>
          )}

          {/* Show transcribe button if callback is provided */}
          {onTranscribe && !isLoading && (
            <button
              onClick={onTranscribe}
              className="w-full py-3 px-5 mt-4 text-white font-bold rounded-lg transition-all duration-300 bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
            >
              Transcribe Audio
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default AudioPlayer;
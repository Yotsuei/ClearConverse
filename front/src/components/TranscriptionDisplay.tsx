// components/TranscriptionDisplay.tsx
import React, { useState, useRef, useEffect } from 'react';

interface TranscriptionDisplayProps {
  transcript: string;
  downloadUrl: string;
  audioBlob: Blob;
  onReset: () => void;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({ 
  transcript, 
  downloadUrl, 
  audioBlob,
  onReset
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showTranscription, setShowTranscription] = useState(true);
  const [audioUrl, setAudioUrl] = useState<string>('');
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Create a URL for the audio blob
    const url = URL.createObjectURL(audioBlob);
    setAudioUrl(url);
    
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [audioBlob]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const setAudioData = () => {
      setDuration(audio.duration);
    };

    const updatePlaybackProgress = () => {
      setCurrentTime(audio.currentTime);
    };

    const handlePlaybackEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    // Event listeners
    audio.addEventListener('loadedmetadata', setAudioData);
    audio.addEventListener('timeupdate', updatePlaybackProgress);
    audio.addEventListener('ended', handlePlaybackEnded);

    return () => {
      audio.removeEventListener('loadedmetadata', setAudioData);
      audio.removeEventListener('timeupdate', updatePlaybackProgress);
      audio.removeEventListener('ended', handlePlaybackEnded);
    };
  }, []);

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleProgressBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const progressBar = progressBarRef.current;
    const audio = audioRef.current;
    if (!progressBar || !audio) return;

    const rect = progressBar.getBoundingClientRect();
    const clickPosition = (e.clientX - rect.left) / rect.width;
    const newTime = clickPosition * duration;
    
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <div className="mt-8 space-y-6">
      {/* Audio Player */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4 text-white">Audio Player</h2>

        <audio ref={audioRef} src={audioUrl} className="hidden" />
        
        <div className="flex flex-col space-y-4">
          {/* Playback controls and progress bar */}
          <div className="flex items-center space-x-4">
            <button 
              onClick={togglePlayback}
              className="flex items-center justify-center w-12 h-12 rounded-full bg-blue-500 text-white hover:bg-blue-600 transition-colors"
            >
              {isPlaying ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                </svg>
              ) : (
                <svg className="w-5 h-5 ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>
            
            <div className="text-sm text-slate-400 w-16">
              {formatTime(currentTime)}
            </div>
            
            <div 
              ref={progressBarRef}
              className="relative flex-1 h-2 bg-slate-700 rounded-full cursor-pointer"
              onClick={handleProgressBarClick}
            >
              <div 
                className="absolute h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"
                style={{ width: `${(currentTime / duration) * 100}%` }}
              />
            </div>
            
            <div className="text-sm text-slate-400 w-16 text-right">
              {formatTime(duration)}
            </div>
          </div>
          
          {/* Waveform visualization (simplified) */}
          <div className="h-16 w-full bg-slate-900/50 rounded-lg flex items-center justify-center overflow-hidden">
            <div className="flex items-end space-x-1 h-12 px-4">
              {Array.from({ length: 60 }).map((_, i) => {
                const height = Math.random() * 100;
                const isCurrentPosition = i / 60 < currentTime / duration;
                
                return (
                  <div
                    key={i}
                    className={`w-1 rounded-t-sm ${isCurrentPosition ? 'bg-gradient-to-t from-blue-500 to-cyan-400' : 'bg-slate-700'}`}
                    style={{ height: `${height}%` }}
                  />
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Transcription Result */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-white">Transcription Result</h2>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowTranscription(!showTranscription)}
              className="p-2 text-slate-400 hover:text-white transition-colors"
            >
              {showTranscription ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                </svg>
              )}
            </button>
          </div>
        </div>
        
        {showTranscription && (
          <pre className="bg-slate-900 p-5 rounded-lg text-slate-300 whitespace-pre-wrap overflow-y-auto max-h-96 border border-slate-700 text-sm leading-relaxed font-mono">
            {transcript}
          </pre>
        )}
        
        <div className="flex flex-wrap gap-4 mt-6">
          <a
            href={`http://localhost:8000${downloadUrl}`}
            download="transcript.txt"
            className="inline-flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors shadow-lg"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download Transcript
          </a>
          
          <button
            onClick={onReset}
            className="inline-flex items-center px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white font-medium rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Process New Audio
          </button>
        </div>
      </div>
    </div>
  );
};

export default TranscriptionDisplay;
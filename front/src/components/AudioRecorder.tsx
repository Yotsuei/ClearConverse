// components/AudioRecorder.tsx
import React, { useState, useRef, useEffect } from 'react';

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [permission, setPermission] = useState<boolean | null>(null);
  const [visualizerData, setVisualizerData] = useState<number[]>(Array(50).fill(0));
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Request microphone permission
  useEffect(() => {
    const getMicrophonePermission = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setAudioStream(stream);
        setPermission(true);
        
        // Set up audio context and analyzer for visualization
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
        
        // Don't connect to destination to avoid feedback loop
        // source.connect(audioContext.destination);
      } catch (err) {
        console.error('Error accessing microphone:', err);
        setPermission(false);
      }
    };

    getMicrophonePermission();

    // Cleanup
    return () => {
      if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Audio visualizer
  const updateVisualizer = () => {
    if (!analyserRef.current) return;
    
    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
    
    // Convert the data to a simpler array for visualization
    const normalizedData = Array.from(dataArray)
      .slice(0, 50)  // Use only a portion of the frequency data
      .map(value => value / 255);  // Normalize to 0-1
    
    setVisualizerData(normalizedData);
    
    animationFrameRef.current = requestAnimationFrame(updateVisualizer);
  };

  const startRecording = () => {
    if (!audioStream) return;

    audioChunksRef.current = [];
    const mediaRecorder = new MediaRecorder(audioStream);
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };
    
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      setRecordingBlob(audioBlob);
      onRecordingComplete(audioBlob);
    };
    
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setIsRecording(true);
    setRecordingTime(0);
    
    // Start timer
    timerRef.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
    
    // Start visualizer
    updateVisualizer();
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      
      // Stop visualizer
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }
  };

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  const resetRecording = () => {
    setRecordingBlob(null);
    setRecordingTime(0);
  };

  return (
    <div className="flex flex-col items-center">
      {permission === false && (
        <div className="bg-red-100 text-red-700 p-4 rounded-lg mb-4 w-full">
          <p className="font-medium">Microphone access denied</p>
          <p className="text-sm">Please allow microphone access to use this feature.</p>
        </div>
      )}
      
      {permission && (
        <>
          {/* Audio Visualizer */}
          <div className="w-full h-20 mb-6 bg-gray-100 rounded-lg overflow-hidden flex items-end">
            {visualizerData.map((value, index) => (
              <div 
                key={index}
                className={`w-1 mx-px ${isRecording ? 'bg-red-500' : 'bg-blue-500'} rounded-t transition-all duration-75`} 
                style={{ 
                  height: `${Math.max(4, value * 100)}%`,
                  opacity: isRecording ? 1 : 0.5
                }}
              ></div>
            ))}
          </div>
          
          {/* Recording Timer */}
          <div className="mb-6 text-center">
            <p className={`text-2xl font-bold ${isRecording ? 'text-red-600' : 'text-gray-700'}`}>
              {formatTime(recordingTime)}
            </p>
            {isRecording && (
              <p className="text-red-600 text-sm animate-pulse">Recording...</p>
            )}
          </div>
          
          {/* Recording Controls */}
          <div className="flex gap-4 justify-center">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="w-16 h-16 bg-red-600 hover:bg-red-700 rounded-full flex items-center justify-center shadow-lg transition-all"
              >
                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="6"></circle>
                </svg>
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="w-16 h-16 bg-gray-700 hover:bg-gray-800 rounded-full flex items-center justify-center shadow-lg transition-all"
              >
                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <rect x="8" y="8" width="8" height="8"></rect>
                </svg>
              </button>
            )}
            
            {recordingBlob && !isRecording && (
              <button
                onClick={resetRecording}
                className="w-12 h-12 bg-gray-200 hover:bg-gray-300 rounded-full flex items-center justify-center shadow-sm transition-all text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
              </button>
            )}
          </div>
          
          {recordingBlob && !isRecording && (
            <div className="mt-6 text-center">
              <p className="text-green-600 font-medium mb-2">Recording complete!</p>
              <p className="text-sm text-gray-500">You can now transcribe this recording</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AudioRecorder;
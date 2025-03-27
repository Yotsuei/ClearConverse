// components/AudioRecorder.tsx
import React, { useState, useRef, useEffect } from 'react';

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  onTranscribe?: () => void;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete, onTranscribe }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [permission, setPermission] = useState<boolean | null>(null);
  const [visualizerData, setVisualizerData] = useState<number[]>(Array(50).fill(2)); // Initialize with minimal height
  const [error, setError] = useState<string | null>(null); // Add error state
  
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
        setError(null); // Clear any previous errors
        
        // Set up audio context and analyzer for visualization
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
      } catch (err) {
        console.error('Error accessing microphone:', err);
        setPermission(false);
        // Display specific error based on the type of error
        if ((err as Error).name === 'NotAllowedError') {
          setError('Microphone access was denied. Please allow microphone access in your browser settings.');
        } else if ((err as Error).name === 'NotFoundError') {
          setError('No microphone detected. Please connect a microphone and try again.');
        } else {
          setError(`Microphone error: ${(err as Error).message || 'Unknown error occurred'}`);
        }
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
    // Take the full range of data and sample it to our 50 bars
    const bars = 50;
    const normalizedData = Array(bars).fill(0);
    
    // Calculate the average of frequency ranges for each bar
    const binSize = Math.floor(dataArray.length / bars) || 1;
    
    for (let i = 0; i < bars; i++) {
      let sum = 0;
      const startBin = i * binSize;
      for (let j = 0; j < binSize && startBin + j < dataArray.length; j++) {
        sum += dataArray[startBin + j];
      }
      // Normalize to 0-1 and ensure a minimum height
      normalizedData[i] = Math.max(2, (sum / binSize) / 255 * 100);
    }
    
    setVisualizerData(normalizedData);
    
    animationFrameRef.current = requestAnimationFrame(updateVisualizer);
  };

  const startRecording = () => {
    if (!audioStream) {
      setError("No audio stream available. Please refresh and try again.");
      return;
    }

    audioChunksRef.current = [];
    try {
      // Determine the best audio format to use based on browser support
      // Try to use WAV directly if possible, fallback to WebM (which will be converted on the server)
      const preferredFormats = [
        { mimeType: 'audio/wav', extension: '.wav' },
        { mimeType: 'audio/mp4', extension: '.mp4' },
        { mimeType: 'audio/webm', extension: '.webm' },
        { mimeType: 'audio/ogg', extension: '.ogg' }
      ];
      
      let selectedFormat = null;
      
      for (const format of preferredFormats) {
        if (MediaRecorder.isTypeSupported(format.mimeType)) {
          selectedFormat = format;
          console.log(`Using supported recording format: ${format.mimeType}`);
          break;
        }
      }
      
      if (!selectedFormat) {
        // If none of the preferred formats are supported, use default
        selectedFormat = { mimeType: '', extension: '.webm' };
        console.log("Using default recording format (browser choice)");
      }
      
      // Create MediaRecorder with the selected format if supported
      const recorderOptions = selectedFormat.mimeType ? 
        { mimeType: selectedFormat.mimeType } : {};
        
      const mediaRecorder = new MediaRecorder(audioStream, recorderOptions);
      
      // Log the actual format being used
      console.log(`MediaRecorder initialized with mimeType: ${mediaRecorder.mimeType}`);
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        try {
          if (audioChunksRef.current.length === 0) {
            throw new Error("No audio data recorded");
          }
          
          // Determine the actual MIME type from the recorder
          const actualMimeType = mediaRecorder.mimeType || 'audio/webm';
          
          // Map MIME type to file extension
          let fileExtension = '.webm'; // Default
          if (actualMimeType.includes('wav')) {
            fileExtension = '.wav';
          } else if (actualMimeType.includes('mp4')) {
            fileExtension = '.mp4';
          } else if (actualMimeType.includes('ogg')) {
            fileExtension = '.ogg';
          }
          
          // Create blob with the correct MIME type
          const audioBlob = new Blob(audioChunksRef.current, { type: actualMimeType });
          console.log(`Created audio blob with type: ${audioBlob.type}, size: ${audioBlob.size} bytes, using extension: ${fileExtension}`);
          
          // Create file object with the correct extension
          const fileName = `recording${fileExtension}`;
          const file = new File([audioBlob], fileName, { type: actualMimeType });
          
          // Create object URL for preview
          const url = URL.createObjectURL(audioBlob);
          
          setRecordingBlob(audioBlob);
          onRecordingComplete(file);
          setError(null); // Clear any errors on successful recording
        } catch (err) {
          console.error("Error finalizing recording:", err);
          setError(`Recording error: ${(err as Error).message || "Failed to process recording"}`);
        }
      };

      mediaRecorder.onerror = (event) => {
        setError(`Recording error: ${(event as any).error?.message || "Unknown error occurred"}`);
        stopRecording();
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
    } catch (err) {
      console.error("Error starting recording:", err);
      setError(`Couldn't start recording: ${(err as Error).message || "Unknown error"}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        mediaRecorderRef.current.stop();
      } catch (err) {
        console.error("Error stopping recording:", err);
        setError(`Error stopping recording: ${(err as Error).message}`);
      }
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
    setError(null);
  };

  const retryMicrophoneAccess = () => {
    // Reset states
    setPermission(null);
    setError(null);
    
    // Re-request microphone access
    const getMicrophonePermission = async () => {
      try {
        // Close existing streams if any
        if (audioStream) {
          audioStream.getTracks().forEach(track => track.stop());
        }
        
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setAudioStream(stream);
        setPermission(true);
        
        // Set up audio context and analyzer for visualization
        if (audioContextRef.current) {
          await audioContextRef.current.close();
        }
        
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
      } catch (err) {
        console.error('Error accessing microphone:', err);
        setPermission(false);
        setError(`Microphone error: ${(err as Error).message || 'Unknown error'}`);
      }
    };

    getMicrophonePermission();
  };

  return (
    <div className="flex flex-col items-center">
      <h2 className="text-xl font-bold text-gray-200 mb-4">Record Audio</h2>
      <p className="text-gray-400 mb-6 text-center">
        Record audio directly from your microphone and transcribe it instantly.
      </p>
      
      {/* Error alert */}
      {error && (
        <div className="bg-red-900 text-red-200 p-4 rounded-lg mb-4 w-full border border-red-700">
          <p className="font-medium flex items-center">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            Recording Error
          </p>
          <p className="text-sm mt-1">{error}</p>
          <button 
            onClick={retryMicrophoneAccess}
            className="mt-2 px-4 py-1 bg-red-800 hover:bg-red-700 text-red-100 rounded text-sm"
          >
            Retry Microphone Access
          </button>
        </div>
      )}
      
      {permission === false && !error && (
        <div className="bg-red-900 text-red-200 p-4 rounded-lg mb-4 w-full border border-red-700">
          <p className="font-medium">Microphone access denied</p>
          <p className="text-sm">Please allow microphone access to use this feature.</p>
          <button 
            onClick={retryMicrophoneAccess}
            className="mt-2 px-4 py-1 bg-red-800 hover:bg-red-700 text-red-100 rounded text-sm"
          >
            Retry Microphone Access
          </button>
        </div>
      )}
      
      {permission && (
        <>
          {/* Audio Visualizer */}
          <div className="w-full h-20 mb-6 bg-gray-800 rounded-lg overflow-hidden flex items-end border border-gray-700">
            {visualizerData.map((value, index) => (
              <div 
                key={index}
                className={`w-1 mx-px ${isRecording ? 'bg-red-500' : 'bg-blue-500'} rounded-t transition-all duration-75`} 
                style={{ 
                  height: `${value}%`,
                  opacity: isRecording ? 1 : 0.5
                }}
              ></div>
            ))}
          </div>
          
          {/* Recording Timer */}
          <div className="mb-6 text-center">
            <p className={`text-2xl font-bold ${isRecording ? 'text-red-400' : 'text-gray-300'}`}>
              {formatTime(recordingTime)}
            </p>
            {isRecording && (
              <p className="text-red-400 text-sm animate-pulse">Recording...</p>
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
                className="w-16 h-16 bg-gray-600 hover:bg-gray-700 rounded-full flex items-center justify-center shadow-lg transition-all"
              >
                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <rect x="8" y="8" width="8" height="8"></rect>
                </svg>
              </button>
            )}
            
            {recordingBlob && !isRecording && (
              <button
                onClick={resetRecording}
                className="w-12 h-12 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center shadow-sm transition-all text-gray-300"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
              </button>
            )}
          </div>
          
          {recordingBlob && !isRecording && onTranscribe && (
            <div className="mt-6 text-center w-full">
              <p className="text-green-400 font-medium mb-2">Recording complete!</p>
              
              <button
                onClick={onTranscribe}
                className="w-full py-3 px-5 mt-2 text-white font-bold rounded-lg transition-all duration-300 bg-blue-600 hover:bg-blue-700 active:scale-98 shadow-lg"
              >
                Transcribe Recording
              </button>
            </div>
          )}
          
          {/* Recording tips */}
          <div className="mt-6 bg-gray-700 p-4 rounded-lg text-sm border border-gray-600 w-full">
            <h3 className="font-semibold text-gray-200 mb-2 flex items-center">
              <svg className="w-5 h-5 mr-2 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              Recording Tips
            </h3>
            <ul className="list-disc list-inside text-gray-300 space-y-1">
              <li>Speak clearly and at a moderate pace</li>
              <li>Minimize background noise for better results</li>
              <li>Keep the microphone at a consistent distance</li>
              <li>Recordings are automatically converted to WAV format for processing</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default AudioRecorder;
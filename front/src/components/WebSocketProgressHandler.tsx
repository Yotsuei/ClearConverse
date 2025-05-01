import { useEffect, useRef } from 'react';
import config from '../config';

interface WebSocketProgressHandlerProps {
  taskId: string | null;
  apiBaseUrl?: string;
  wsBaseUrl?: string;
  onProgressUpdate: (progress: number, message: string) => void;
  onComplete: (downloadUrl: string) => void;
  onConnectionFailed?: () => void;
  maxReconnectAttempts?: number;
}

const WebSocketProgressHandler: React.FC<WebSocketProgressHandlerProps> = ({ 
  taskId, 
  apiBaseUrl = config.api.baseUrl,
  wsBaseUrl = config.api.wsBaseUrl,
  onProgressUpdate,
  onComplete,
  onConnectionFailed,
  maxReconnectAttempts = config.ui.maxWebSocketReconnectAttempts
}) => {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Function to establish WebSocket connection
  const connectWebSocket = () => {
    if (!taskId) return;

    if (socketRef.current && 
        (socketRef.current.readyState === WebSocket.CONNECTING || 
        socketRef.current.readyState === WebSocket.OPEN)) {
      return;
    }


    // Clean up previous socket if exists
    if (socketRef.current && socketRef.current.readyState !== WebSocket.CLOSED) {
      socketRef.current.close();
    }

    const wsUrl = `${wsBaseUrl}/ws/progress/${taskId}`;
    console.log(`Connecting to WebSocket: ${wsUrl}`);

    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      console.log(`WebSocket connection established for task ${taskId}`);
      reconnectAttempts.current = 0; // Reset reconnect attempts on successful connection
      onProgressUpdate(5, "Connected to server. Waiting for processing to begin...");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle initial connection message
        if (data.status === "connected") {
          return;
        }
        
        // Special handling for cancellation messages - always show these
        if (data.message && data.message.toLowerCase().includes('cancel')) {
          onProgressUpdate(data.progress || 99, data.message);
          
          // If progress is 100, this is a completed cancellation
          if (data.progress >= 100) {
            // Check if there's a download URL
            checkTaskResult(taskId);
          }
          return;
        }
        
        // Regular progress updates
        if (data.progress !== undefined && data.message !== undefined) {
          onProgressUpdate(data.progress, data.message);
        }
        
        // Check if processing is complete
        if (data.progress >= 100) {
          checkTaskResult(taskId);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
        onProgressUpdate(5, 'Processing in progress...');
      }
    };    

    ws.onerror = () => {
      // Show generic processing message instead of connection error
      onProgressUpdate(5, 'Processing in progress...');
      if (onConnectionFailed) onConnectionFailed();
    };

    ws.onclose = (event) => {
      console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
      
      // Attempt to reconnect unless this was a normal closure
      if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
        const timeoutDelay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        
        reconnectAttempts.current += 1;
        onProgressUpdate(5, `Processing in progress...`);
        
        // Set timeout for reconnection
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, timeoutDelay);
      } else if (reconnectAttempts.current >= maxReconnectAttempts) {
        onProgressUpdate(5, 'Processing in progress...');
        if (onConnectionFailed) onConnectionFailed();
      }
    };
  };

  useEffect(() => {
    if (taskId) {
      connectWebSocket();
    }

    // Clean up on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [taskId]);

  const checkTaskResult = async (taskId: string) => {
    try {
      const response = await fetch(`${apiBaseUrl}/task/${taskId}/result`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch task result: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.download_url) {
        onComplete(result.download_url);
      } else if (result.status === "cancelled") {
        // Handle cancelled state properly
        onProgressUpdate(100, result.message || "Transcription was cancelled");
        // Don't try to fetch transcription for cancelled tasks
      } else if (result.error) {
        onProgressUpdate(100, `Error: ${result.error}`);
      }
    } catch (err) {
      console.error('Error checking task result:', err);
      onProgressUpdate(100, `Failed to get task result: ${(err as Error).message}`);
    }
  };  

  // This component doesn't render anything
  return null;
};

export default WebSocketProgressHandler;
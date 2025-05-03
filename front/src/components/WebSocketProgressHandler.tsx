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
      reconnectAttempts.current = 0; // Reset reconnect attempts on success
      onProgressUpdate(5, "Connected to server. Waiting for processing to begin...");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle progress updates
        if (data.progress !== undefined && data.message !== undefined) {
          onProgressUpdate(data.progress, data.message);
          
          // For 100% progress, check the message to determine next action
          if (data.progress >= 100) {
            if (data.message && data.message.toLowerCase().includes('cancel')) {
              // Cancelled - no need to fetch result
              console.log("Transcription was cancelled");
            } 
            else if (data.message && data.message.toLowerCase().includes('error')) {
              // Error occurred - no need to fetch result
              console.log(`Transcription error: ${data.message}`);
            }
            else {
              // Completed successfully - check for result
              console.log("Transcription completed, fetching result");
              checkTaskResult(taskId);
            }
          }
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
      
      // Also check status immediately in case of completed/cancelled state
      checkTaskResult(taskId);
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
      const response = await fetch(`${apiBaseUrl}/task/${taskId}/status`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch task result: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.status === "completed" && result.download_url) {
        onProgressUpdate(100, "Processing complete!");
        onComplete(result.download_url);
      } 
      else if (result.status === "cancelled") {
        onProgressUpdate(100, result.message || "Transcription was cancelled");
      } 
      else if (result.status === "error") {
        onProgressUpdate(100, `Error: ${result.message || "Unknown error"}`);
      }
      else if (result.progress !== undefined) {
        // Regular progress update
        onProgressUpdate(result.progress, result.message || "Processing in progress...");
      }
    } catch (err) {
      console.error('Error checking task result:', err);
    }
  };

  // This component doesn't render anything
  return null;
};

export default WebSocketProgressHandler;
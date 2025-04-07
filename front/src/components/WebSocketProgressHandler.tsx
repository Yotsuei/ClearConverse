// components/WebSocketProgressHandler.tsx
import React, { useEffect, useState, useRef } from 'react';

interface WebSocketProgressHandlerProps {
  taskId: string | null;
  onProgressUpdate: (progress: number, message: string) => void;
  onComplete: (downloadUrl: string) => void;
}

const WebSocketProgressHandler: React.FC<WebSocketProgressHandlerProps> = ({ 
  taskId, 
  onProgressUpdate,
  onComplete
}) => {
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Function to establish WebSocket connection
  const connectWebSocket = () => {
    if (!taskId) return;

    // Clean up previous socket if exists
    if (socketRef.current && socketRef.current.readyState !== WebSocket.CLOSED) {
      socketRef.current.close();
    }

    const wsUrl = `ws://localhost:8000/ws/progress/${taskId}`;
    console.log(`Connecting to WebSocket: ${wsUrl}`);

    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    // Set up event handlers
    ws.onopen = () => {
      console.log(`WebSocket connection established for task ${taskId}`);
      reconnectAttempts.current = 0; // Reset reconnect attempts on successful connection
      onProgressUpdate(5, "Connected to server. Waiting for processing to begin...");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket data received:', data);
        
        // Handle initial connection message
        if (data.status === "connected") {
          console.log("WebSocket connection acknowledged");
          return;
        }
        
        // Update the progress with the exact message from backend
        if (data.progress !== undefined && data.message !== undefined) {
          onProgressUpdate(data.progress, data.message);
        }
        
        // Check if processing is complete
        if (data.progress >= 100) {
          // Check task result to get download URL
          checkTaskResult(taskId);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
        setError('Failed to parse progress update');
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError('WebSocket connection error');
      
      // Show generic processing message instead of connection error
      onProgressUpdate(5, 'Processing in progress...');
    };

    ws.onclose = (event) => {
      console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
      
      // Attempt to reconnect unless this was a normal closure
      if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
        const timeoutDelay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        console.log(`Attempting to reconnect in ${timeoutDelay}ms (attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
        
        reconnectAttempts.current += 1;
        onProgressUpdate(5, `Processing in progress...`);
        
        // Set timeout for reconnection
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, timeoutDelay);
      } else if (reconnectAttempts.current >= maxReconnectAttempts) {
        console.log('Maximum reconnection attempts reached');
        onProgressUpdate(5, 'Processing in progress...');
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
      const response = await fetch(`http://localhost:8000/task/${taskId}/result`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch task result: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('Task result:', result);
      
      if (result.download_url) {
        onComplete(result.download_url);
      } else if (result.error) {
        setError(`Task failed: ${result.error}`);
        onProgressUpdate(100, `Error: ${result.error}`);
      }
    } catch (err) {
      console.error('Error checking task result:', err);
      setError(`Failed to get task result: ${(err as Error).message}`);
      onProgressUpdate(100, `Failed to get task result: ${(err as Error).message}`);
    }
  };

  // This component doesn't render anything
  return null;
};

export default WebSocketProgressHandler;
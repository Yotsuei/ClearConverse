// components/WebSocketProgressHandler.tsx
import React, { useEffect, useState } from 'react';

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
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!taskId) return;

    // Clean up previous socket if exists
    if (socket) {
      socket.close();
    }

    // Create new WebSocket connection
    const ws = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`);
    setSocket(ws);

    // Set up event handlers
    ws.onopen = () => {
      console.log(`WebSocket connection established for task ${taskId}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket data received:', data);
        
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
      onProgressUpdate(0, 'Connection error. Please try again.');
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    // Clean up on unmount
    return () => {
      if (ws) {
        ws.close();
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
/**
 * Application configuration that handles both development and production environments
 */

const config = {
    // API endpoints
    api: {
      baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
      wsBaseUrl: import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000',
    },
    
    // File upload limits
    upload: {
      maxFileSizeMB: 10,  // Setting 10MB limit
      maxFileSizeBytes: 10 * 1024 * 1024,  // 10MB in bytes
      acceptedFileFormats: ['.wav', '.mp3', '.mp4', '.webm', '.ogg', '.flac', '.m4a', '.aac'],
      preferredFormats: ['.wav', '.mp3'],
    },
    
    // UI configuration
    ui: {
      applicationName: 'ClearConverse',
      applicationDescription: 'A speech transcription tool powered by Whisper-RESepFormer solution',
      progressPollingInterval: 2000, // ms
      maxWebSocketReconnectAttempts: 5,
    }
  };
  
  export default config;
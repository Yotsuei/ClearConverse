import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Allow connections from Docker network
    host: '0.0.0.0',
    // Use port 5173 as default
    port: 5173,
    // Add CORS headers for development
    cors: true,
    // Automatically open in browser (disable in Docker)
    open: false,
    // Handle HMR properly
    watch: {
      usePolling: true,
    },
  },
  // Proper base path for production builds
  base: '/',
})
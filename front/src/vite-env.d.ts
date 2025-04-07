/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_API_BASE_URL: string
    readonly VITE_WS_BASE_URL: string
    // add more environment variables as needed
  }
  
  interface ImportMeta {
    readonly env: ImportMetaEnv
  }
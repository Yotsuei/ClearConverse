// components/MainMenu.tsx
import React from 'react';

interface MainMenuProps {
  onSelectModule: (module: 'upload' | 'record' | 'url') => void;
}

const MainMenu: React.FC<MainMenuProps> = ({ onSelectModule }) => {
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-5xl font-extrabold text-gray-900 mb-4">
          Speech <span className="text-blue-600">Transcription</span> Tool
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Convert your audio files or recordings into text with our advanced speech recognition technology.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8 mt-12">
        {/* Upload Audio Card */}
        <div 
          onClick={() => onSelectModule('upload')}
          className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200 transition-all duration-300 hover:shadow-xl hover:translate-y-[-8px] cursor-pointer"
        >
          <div className="h-48 bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center">
            <svg className="w-24 h-24 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
          </div>
          <div className="p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Upload File</h3>
            <p className="text-gray-600 mb-4">
              Upload your audio files for transcription. Supports MP3, WAV, M4A, and OGG formats.
            </p>
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
              </svg>
              Max file size: 20MB
            </div>
            <button className="mt-5 w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition-colors">
              Choose File
            </button>
          </div>
        </div>

        {/* Google Drive URL Card */}
        <div 
          onClick={() => onSelectModule('url')}
          className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200 transition-all duration-300 hover:shadow-xl hover:translate-y-[-8px] cursor-pointer"
        >
          <div className="h-48 bg-gradient-to-r from-green-500 to-green-600 flex items-center justify-center">
            <svg className="w-24 h-24 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path>
            </svg>
          </div>
          <div className="p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Google Drive</h3>
            <p className="text-gray-600 mb-4">
              Transcribe audio directly from Google Drive using a shareable URL. No download needed.
            </p>
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path>
              </svg>
              Must be publicly accessible
            </div>
            <button className="mt-5 w-full py-3 bg-green-600 hover:bg-green-700 text-white font-bold rounded-lg transition-colors">
              Paste Drive URL
            </button>
          </div>
        </div>

        {/* Record Audio Card */}
        <div 
          onClick={() => onSelectModule('record')}
          className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200 transition-all duration-300 hover:shadow-xl hover:translate-y-[-8px] cursor-pointer"
        >
          <div className="h-48 bg-gradient-to-r from-red-500 to-red-600 flex items-center justify-center">
            <svg className="w-24 h-24 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
            </svg>
          </div>
          <div className="p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Record Audio</h3>
            <p className="text-gray-600 mb-4">
              Record audio directly from your microphone for instant transcription.
            </p>
            <div className="flex items-center text-sm text-gray-500">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              High-quality audio recording
            </div>
            <button className="mt-5 w-full py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg transition-colors">
              Start Recording
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainMenu;
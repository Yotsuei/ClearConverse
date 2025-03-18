// components/MainMenu.tsx
import React from 'react';

interface MainMenuProps {
  onSelectModule: (module: 'upload' | 'record') => void;
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

      <div className="grid md:grid-cols-2 gap-8 mt-12">
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
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Upload Audio</h3>
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

      {/* Features Section */}
      <div className="mt-16 text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-8">Features</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100">
            <div className="w-12 h-12 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2">Fast Processing</h3>
            <p className="text-gray-600">Quick and accurate transcription within minutes</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100">
            <div className="w-12 h-12 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center text-green-600">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2">Speaker Detection</h3>
            <p className="text-gray-600">Automatically identifies different speakers in conversations</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100">
            <div className="w-12 h-12 mx-auto mb-4 bg-purple-100 rounded-full flex items-center justify-center text-purple-600">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
            </div>
            <h3 className="text-lg font-semibold mb-2">Export Options</h3>
            <p className="text-gray-600">Download your transcription in multiple formats</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainMenu;
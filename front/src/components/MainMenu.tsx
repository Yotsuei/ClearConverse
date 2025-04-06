// components/MainMenu.tsx
import React from 'react';

interface MainMenuProps {
  onSelectModule: (module: 'upload' | 'url') => void;
}

const MainMenu: React.FC<MainMenuProps> = ({ onSelectModule }) => {
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-5xl font-extrabold text-gray-100 mb-4">
          Speech <span className="text-blue-400">Transcription</span> Tool
        </h1>
        <p className="text-xl text-gray-400 max-w-3xl mx-auto">
          Choose one of the options below to convert speech to text using our advanced transcription technology.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mt-12 max-w-3xl mx-auto">
        {/* Upload Audio File Card */}
        <div 
          onClick={() => onSelectModule('upload')}
          className="bg-gray-800 rounded-xl overflow-hidden shadow-lg border border-gray-700 transition-all duration-300 hover:shadow-xl hover:translate-y-[-8px] cursor-pointer"
        >
          <div className="h-48 bg-gradient-to-r from-blue-800 to-blue-900 flex items-center justify-center">
            <svg className="w-24 h-24 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
          </div>
          <div className="p-6">
            <h3 className="text-2xl font-bold text-gray-200 mb-2">Upload Audio File</h3>
            <p className="text-gray-400 mb-4">
              Upload an existing audio or video file for transcription.
            </p>
            <div className="flex items-center text-sm text-gray-500 mb-3">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
              </svg>
              Supported formats: WAV, MP3, MP4
            </div>
            <div className="flex items-center text-sm text-gray-500 mb-5">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              Max file size: 20MB
            </div>
            <button className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition-colors">
              Choose This Option
            </button>
          </div>
        </div>

        {/* URL Upload Card */}
        <div 
          onClick={() => onSelectModule('url')}
          className="bg-gray-800 rounded-xl overflow-hidden shadow-lg border border-gray-700 transition-all duration-300 hover:shadow-xl hover:translate-y-[-8px] cursor-pointer"
        >
          <div className="h-48 bg-gradient-to-r from-purple-800 to-purple-900 flex items-center justify-center">
            <svg className="w-24 h-24 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path>
            </svg>
          </div>
          <div className="p-6">
            <h3 className="text-2xl font-bold text-gray-200 mb-2">Upload from URL</h3>
            <p className="text-gray-400 mb-4">
              Provide a link to an audio or video file hosted online.
            </p>
            <div className="flex items-center text-sm text-gray-500 mb-3">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path>
              </svg>
              Support for direct audio links and Google Drive
            </div>
            <div className="flex items-center text-sm text-gray-500 mb-5">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              Public URL access required
            </div>
            <button className="w-full py-3 bg-purple-600 hover:bg-purple-700 text-white font-bold rounded-lg transition-colors">
              Choose This Option
            </button>
          </div>
        </div>
      </div>

      <div className="mt-12 text-center text-gray-500">
        <p>Both options provide high-quality transcription with speaker detection.</p>
      </div>
    </div>
  );
};

export default MainMenu;
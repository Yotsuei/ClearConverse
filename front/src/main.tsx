import './index.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Create root using the new React 18 API
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
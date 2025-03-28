// components/Tab.tsx
import React from 'react';

interface TabProps {
  active: boolean;
  onClick: () => void;
  label: string;
  icon: string;
}

export const Tab: React.FC<TabProps> = ({ active, onClick, label, icon }) => {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 py-3 px-5 font-medium transition-all duration-300 ${
        active 
          ? 'bg-white text-blue-600 border-t-2 border-blue-600 shadow-sm' 
          : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
      }`}
    >
      <span className="text-xl">{icon}</span>
      {label}
    </button>
  );
};
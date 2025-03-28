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
          ? 'bg-gray-700 text-blue-400 border-t-2 border-blue-400 shadow-sm' 
          : 'bg-gray-800 text-gray-400 hover:bg-gray-750'
      }`}
    >
      <span className="text-xl">{icon}</span>
      {label}
    </button>
  );
};
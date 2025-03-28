// components/FloatingActionButton.tsx
import React from 'react';

interface FloatingActionButtonProps {
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  color: 'blue' | 'gray' | 'red' | 'yellow';
  tooltip: string;
}

const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  onClick,
  icon,
  label,
  color,
  tooltip
}) => {
  // Define color classes based on the color prop
  const getColorClasses = (): string => {
    switch (color) {
      case 'blue':
        return 'bg-blue-600 hover:bg-blue-700';
      case 'red':
        return 'bg-red-600 hover:bg-red-700';
      case 'yellow':
        return 'bg-yellow-600 hover:bg-yellow-700';
      case 'gray':
      default:
        return 'bg-gray-700 hover:bg-gray-600';
    }
  };

  return (
    <button
      onClick={onClick}
      className={`p-3 rounded-full shadow-lg ${getColorClasses()} text-white transition-all hover:scale-105 flex items-center justify-center`}
      aria-label={label}
      title={tooltip}
    >
      {icon}
    </button>
  );
};

export default FloatingActionButton;
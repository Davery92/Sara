import React from 'react';

const SuggestionChip = ({ text, onClick }) => {
  return (
    <div 
      className="bg-card-bg border border-border-color rounded-full px-4 py-2 text-sm cursor-pointer transition-colors hover:bg-hover-color"
      onClick={() => onClick(text)}
    >
      {text}
    </div>
  );
};

export default SuggestionChip;
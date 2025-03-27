import React from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

const BriefingsList = ({ briefings = [], onOpenBriefing, onRefresh }) => {
  const formatFileSize = (bytes) => {
    if (!bytes || bytes < 1024) {
      return (bytes || 0) + ' B';
    } else if (bytes < 1024 * 1024) {
      return (bytes / 1024).toFixed(1) + ' KB';
    } else {
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString(undefined, { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      });
    } catch (e) {
      return 'Unknown date';
    }
  };

  return (
    <div className="bg-card-bg rounded-lg overflow-hidden">
      <div className="p-4 border-b border-border-color flex justify-between items-center">
        <h3 className="font-medium">Available Briefings</h3>
        <button 
          className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
          onClick={onRefresh}
        >
          <ArrowPathIcon className="w-4 h-4" />
          Refresh
        </button>
      </div>
      
      <div className="p-4">
        {briefings.length === 0 ? (
          <div className="text-center py-4 text-muted-color">
            No briefings available
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {briefings.map((briefing, index) => (
              <div 
                key={briefing.filename || index}
                className="bg-card-bg border border-border-color rounded-lg p-4 cursor-pointer transition-colors hover:bg-hover-color"
                onClick={() => onOpenBriefing(briefing.filename)}
              >
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-medium">{briefing.title}</h4>
                  <span className="text-xs text-muted-color">{formatDate(briefing.created)}</span>
                </div>
                <div className="text-xs text-muted-color">
                  {formatFileSize(briefing.size)} • {briefing.filename}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BriefingsList;
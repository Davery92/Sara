import React from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

const SystemStatusCard = ({ systemStatus, onRefresh }) => {
  return (
    <div className="bg-card-bg rounded-lg overflow-hidden">
      <div className="p-4 border-b border-border-color flex justify-between items-center">
        <h3 className="font-medium">System Status</h3>
        <button 
          className="p-1.5 text-sm bg-hover-color text-muted-color rounded flex items-center gap-1.5 hover:text-text-color transition-colors"
          onClick={onRefresh}
        >
          <ArrowPathIcon className="w-4 h-4" />
          Refresh
        </button>
      </div>
      
      <div className="p-4">
        <div className="flex flex-wrap gap-3 mb-4">
          <StatusPill 
            label="Server"
            status={systemStatus.server}
            type={systemStatus.server === 'Online' ? 'success' : 'error'}
          />
          
          <StatusPill 
            label="Redis"
            status={systemStatus.redis}
            type={systemStatus.redis === 'Connected' ? 'success' : 'warning'}
          />
          
          <StatusPill 
            label="Neo4j"
            status={systemStatus.neo4j}
            type={systemStatus.neo4j === 'Connected' ? 'success' : 'warning'}
          />
        </div>
        
        <div className="mt-4">
          <p className="mb-2">GPU Memory Usage</p>
          <div className="bg-input-bg rounded h-2.5 overflow-hidden mb-2">
            <div 
              className="h-full bg-accent-color"
              style={{ width: '45%' }} // This would be calculated from actual data
            ></div>
          </div>
          <p className="text-sm text-muted-color">5.8 GB / 12 GB</p>
        </div>
      </div>
    </div>
  );
};

const StatusPill = ({ label, status, type }) => {
  const getBgColor = () => {
    switch(type) {
      case 'success': return 'bg-green-900/20 text-green-500';
      case 'warning': return 'bg-yellow-900/20 text-yellow-500';
      case 'error': return 'bg-red-900/20 text-red-500';
      default: return 'bg-blue-900/20 text-blue-500';
    }
  };
  
  const getDotColor = () => {
    switch(type) {
      case 'success': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-blue-500';
    }
  };
  
  return (
    <div className={`px-4 py-2 rounded-md flex items-center gap-2 ${getBgColor()}`}>
      <span className={`w-2 h-2 rounded-full ${getDotColor()}`}></span>
      {label}: {status}
    </div>
  );
};

export default SystemStatusCard;
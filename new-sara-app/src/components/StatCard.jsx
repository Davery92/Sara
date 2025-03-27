import React from 'react';

const StatCard = ({ title, value, detail, meter }) => {
  return (
    <div className="bg-card-bg rounded-lg p-5 flex flex-col">
      <span className="text-sm text-muted-color mb-2">{title}</span>
      <span className="text-xl font-semibold mb-1">{value}</span>
      
      {meter && (
        <div className="my-2">
          <div className="bg-input-bg rounded-sm h-2 overflow-hidden">
            <div 
              className={`h-full ${
                meter.color === 'red' ? 'bg-error-color' :
                meter.color === 'yellow' ? 'bg-yellow-500' :
                'bg-accent-color'
              }`}
              style={{ width: `${meter.percentage}%` }}
            ></div>
          </div>
        </div>
      )}
      
      <span className="text-sm text-muted-color">{detail}</span>
    </div>
  );
};

export default StatCard;
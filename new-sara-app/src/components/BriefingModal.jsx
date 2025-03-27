import { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import ReactMarkdown from 'react-markdown';

const BriefingModal = ({ isOpen, onClose, briefingTitle, briefingFilename }) => {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen && briefingFilename) {
      loadBriefingContent(briefingFilename);
    }
  }, [isOpen, briefingFilename]);

  const loadBriefingContent = async (filename) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`/v1/briefings/content/${filename}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setContent(data.content || '');
      }
    } catch (error) {
      setError('Error loading briefing content');
      console.error('Error fetching briefing content:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-bg-color rounded-lg w-11/12 max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-lg animate-[modal-fade-in_0.3s_ease]">
        <div className="p-4 border-b border-border-color flex items-center justify-between">
          <h3 className="text-lg font-semibold">{briefingTitle || 'Briefing'}</h3>
          <button 
            className="text-muted-color p-1 rounded hover:text-text-color hover:bg-hover-color transition-colors"
            onClick={onClose}
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>
        
        <div className="p-6 overflow-y-auto">
          {loading ? (
            <div className="text-center py-4">
              <div className="inline-block w-6 h-6 border-2 border-t-accent-color rounded-full animate-spin"></div>
              <p className="mt-2 text-muted-color">Loading briefing content...</p>
            </div>
          ) : error ? (
            <div className="text-error-color">
              <p>Error: {error}</p>
            </div>
          ) : (
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown>
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BriefingModal;
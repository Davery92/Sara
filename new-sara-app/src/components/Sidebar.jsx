import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChevronDownIcon } from '@heroicons/react/24/outline';

const Sidebar = ({ 
  onNewChat, 
  onClearChat, 
  conversations = [],
  isMobile = false,
  showSidebar = true,
  ttsEnabled = false,
  onTtsToggle,
  selectedVoice,
  onVoiceChange
}) => {
  const location = useLocation();
  const [briefings, setBriefings] = useState([]);
  const [briefingsLoading, setBriefingsLoading] = useState(false);
  const [showBriefings, setShowBriefings] = useState(false);

  useEffect(() => {
    // Load briefings when dropdown is opened
    if (showBriefings && briefings.length === 0 && !briefingsLoading) {
      loadBriefings();
    }
  }, [showBriefings, briefings.length, briefingsLoading]);

  const loadBriefings = async () => {
    try {
      setBriefingsLoading(true);
      const response = await fetch('/v1/briefings/list');
      const data = await response.json();
      
      if (data.error) {
        console.error('Error loading briefings:', data.error);
      } else {
        setBriefings(data.briefings || []);
      }
    } catch (error) {
      console.error('Error fetching briefings:', error);
    } finally {
      setBriefingsLoading(false);
    }
  };

  const openBriefingModal = (filename) => {
    // This would be implemented elsewhere in the app and passed as a prop
    if (window.openBriefingModal) {
      window.openBriefingModal(filename, briefings);
    }
  };

  const toggleBriefingsDropdown = () => {
    setShowBriefings(!showBriefings);
  };

  return (
    <div className={`w-64 bg-sidebar-bg h-full flex flex-col border-r border-border-color overflow-y-auto flex-shrink-0 z-10 transition-transform duration-300 ${isMobile && !showSidebar ? '-translate-x-full' : 'translate-x-0'} ${isMobile ? 'absolute' : 'relative'}`}>
      <div className="p-5 border-b border-border-color flex items-center justify-center">
        <h1 className="text-xl font-semibold">Sara Assistant</h1>
      </div>
      
      {onNewChat && (
        <button 
          className="bg-accent-color text-white border-none rounded py-3 px-4 mx-4 my-4 text-sm font-semibold flex items-center justify-center gap-2 transition-colors hover:bg-accent-hover cursor-pointer"
          onClick={onNewChat}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          New Chat
        </button>
      )}

      <div className="flex-1">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-color p-3 pt-4">
          Navigation
        </div>
        <Link to="/chat" className={`flex items-center p-3 text-text-color no-underline transition-colors mb-1 border-l-4 ${location.pathname === '/chat' ? 'bg-card-bg border-l-accent-color' : 'hover:bg-hover-color border-l-transparent'}`}>
          <svg className="mr-3 w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            <circle cx="9" cy="10" r="1"></circle>
            <circle cx="15" cy="10" r="1"></circle>
          </svg>
          Chat
        </Link>
        <Link to="/dashboard" className={`flex items-center p-3 text-text-color no-underline transition-colors mb-1 border-l-4 ${location.pathname === '/dashboard' ? 'bg-card-bg border-l-accent-color' : 'hover:bg-hover-color border-l-transparent'}`}>
          <svg className="mr-3 w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="7" height="7"></rect>
            <rect x="14" y="3" width="7" height="7"></rect>
            <rect x="14" y="14" width="7" height="7"></rect>
            <rect x="3" y="14" width="7" height="7"></rect>
          </svg>
          Dashboard
        </Link>
        <Link to="/notes" className={`flex items-center p-3 text-text-color no-underline transition-colors mb-1 border-l-4 ${location.pathname === '/notes' ? 'bg-card-bg border-l-accent-color' : 'hover:bg-hover-color border-l-transparent'}`}>
          <svg className="mr-3 w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
          Notes
        </Link>
        <Link to="/documents" className={`flex items-center p-3 text-text-color no-underline transition-colors mb-1 border-l-4 ${location.pathname === '/documents' ? 'bg-card-bg border-l-accent-color' : 'hover:bg-hover-color border-l-transparent'}`}>
          <svg className="mr-3 w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
          </svg>
          Documents
        </Link>
        
        <div className="text-xs font-medium uppercase tracking-wider text-muted-color p-3 pt-4">
          Resources
        </div>
        <div 
          className="flex items-center p-3 text-text-color cursor-pointer hover:bg-hover-color"
          onClick={toggleBriefingsDropdown}
        >
          <svg className="mr-3 w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1M15 7l4 4-4 4"></path>
            <path d="M7 15h8M7 11h4"></path>
          </svg>
          Briefings
          <ChevronDownIcon 
            className={`w-5 h-5 ml-auto transition-transform ${showBriefings ? 'rotate-180' : ''}`} 
          />
        </div>
        
        {showBriefings && (
          <div className="border-t border-b border-border-color max-h-72 overflow-y-auto">
            {briefingsLoading ? (
              <div className="p-2 text-muted-color text-sm">Loading briefings...</div>
            ) : briefings.length === 0 ? (
              <div className="p-2 text-muted-color text-sm">No briefings available</div>
            ) : (
              briefings.map((briefing) => (
                <div
                  key={briefing.filename}
                  className="py-2 px-3 pl-6 cursor-pointer text-sm transition-colors hover:bg-hover-color truncate"
                  onClick={() => openBriefingModal(briefing.filename)}
                >
                  {briefing.title}
                </div>
              ))
            )}
          </div>
        )}
        
        {conversations && conversations.length > 0 && (
          <>
            <div className="text-xs font-medium uppercase tracking-wider text-muted-color p-3 pt-4">
              Recent Conversations
            </div>
            <div className="px-2">
              {conversations.map((conversation) => (
                <div 
                  key={conversation.id} 
                  className={`flex items-center p-2 mb-1 rounded cursor-pointer transition-colors ${conversation.active ? 'bg-card-bg' : 'hover:bg-hover-color'}`}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-3 text-muted-color">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                  </svg>
                  <span className="truncate text-sm">{conversation.title}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="border-t border-border-color p-4">
        {/* Voice selector */}
        {onTtsToggle && (
          <div className={`mb-3 p-2 border border-gray-700 rounded bg-input-bg ${!ttsEnabled ? 'hidden' : ''}`}>
            <label htmlFor="tts-voice" className="block mb-2 text-sm font-bold text-text-color">
              Voice Selection
            </label>
            <select
              id="tts-voice"
              className="w-full p-2 bg-card-bg text-text-color border border-border-color rounded text-sm"
              value={selectedVoice}
              onChange={(e) => onVoiceChange(e.target.value)}
            >
              <option value="af_bella">Bella (African)</option>
              <option value="en_jony">Jony (English)</option>
              <option value="en_rachel">Rachel (English)</option>
              <option value="en_emma">Emma (English)</option>
              <option value="en_antoni">Antoni (English)</option>
            </select>
            <div className="mt-2 text-xs text-muted-color">
              Choose a voice for text-to-speech
            </div>
          </div>
        )}
        
        {onClearChat && (
          <button
            className="flex items-center gap-2 w-full p-2 text-muted-color border-none rounded bg-transparent transition-colors hover:bg-hover-color cursor-pointer text-left text-sm"
            onClick={onClearChat}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 6h18"></path>
              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
              <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
            </svg>
            Clear chat
          </button>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
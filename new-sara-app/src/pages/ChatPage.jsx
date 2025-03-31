import { useState, useEffect, useRef } from 'react';
import { Bars3Icon } from '@heroicons/react/24/outline';
import Sidebar from '../components/Sidebar';
import BriefingModal from '../components/BriefingModal';
import { formatMessage, cleanTextForTTS } from '../utils/formatters'; // Added cleanTextForTTS import
import SuggestionChip from '../components/SuggestionChip';

// Helper to ensure AudioContext is available
const getAudioContext = () => {
  if (typeof window !== 'undefined') {
    window.AudioContext = window.AudioContext || window.webkitAudioContext;
    if (window.AudioContext) {
      return new window.AudioContext();
    }
  }
  console.error('Web Audio API is not supported in this browser.');
  return null;
};

const ChatPage = () => {
  // State
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState('af_bella');
  const [showEmptyState, setShowEmptyState] = useState(true);
  const [briefingModal, setBriefingModal] = useState({
    isOpen: false,
    title: '',
    filename: ''
  });
  const [isPreparingAudio, setIsPreparingAudio] = useState(false);
  
  // Refs
  const messagesEndRef = useRef(null);
  const messageContainerRef = useRef(null);
  const textareaRef = useRef(null);
  const isInitialMount = useRef(true); // Track initial mount
  const clearChatAfterNew = useRef(false); // Track if clear chat is needed after new chat
  // Audio refs - similar to BriefingModal
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const audioBufferRef = useRef(null);

  // WebSocket for briefing notifications
  useEffect(() => {
    // Check if WebSocket is supported
    if (!('WebSocket' in window)) {
      console.warn('WebSockets not supported in this browser');
      return;
    }

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/briefings`;
    
    // Create WebSocket connection
    const socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
      console.log('WebSocket connection established for briefing notifications');
    };
    
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle briefing completion notification
        if (data.type === 'briefing_completed') {
          // Show toast notification using the global showToast function
          if (window.showToast) {
            window.showToast(
              `Your briefing on "${data.query}" is now available.`, 
              'success',
              8000 // Show for 8 seconds
            );
          }
          
          // Optionally play a notification sound
          const notificationSound = new Audio('/notification.mp3');
          notificationSound.play().catch(e => console.warn('Error playing notification sound:', e));
        }
      } catch (error) {
        console.error('Error handling WebSocket message:', error);
      }
    };
    
    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    socket.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    // Clean up the WebSocket connection when component unmounts
    return () => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, []);

  // Initialize AudioContext - similar to BriefingModal
  useEffect(() => {
    if (!audioContextRef.current && ttsEnabled) {
      try {
        audioContextRef.current = getAudioContext();
        console.log('AudioContext initialized state:', audioContextRef.current?.state);
      } catch (e) {
        console.error("Error creating AudioContext:", e);
        if (window.showToast) {
          window.showToast("Audio playback not supported in this browser.", "error");
        }
      }
    }
  }, [ttsEnabled]);

  // Audio cleanup on unmount
  useEffect(() => {
    return () => {
      stopAndCleanupAudio(true);
    };
  }, []);

  // Stop and cleanup audio function - similar to BriefingModal
  const stopAndCleanupAudio = (isUnmounting = false) => {
    console.log('Stopping and cleaning up audio...');
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.onended = null;
        sourceNodeRef.current.stop();
        console.log('Audio source node stopped.');
      } catch (e) {
        console.warn("Error stopping source node (might have already finished):", e);
      }
      sourceNodeRef.current = null;
    }

    // Reset states
    setIsPreparingAudio(false);
    audioBufferRef.current = null;

    // Close the context completely only when the component unmounts
    if (isUnmounting && audioContextRef.current) {
      if (audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().then(() => {
          console.log('AudioContext closed.');
          audioContextRef.current = null;
        }).catch(e => console.error("Error closing AudioContext:", e));
      } else {
        audioContextRef.current = null;
      }
    }
  };

  // Check if mobile on mount and window resize
  useEffect(() => {
    const checkIfMobile = () => {
      setIsMobile(window.innerWidth <= 768);
      if (window.innerWidth <= 768) {
        setShowSidebar(false);
      }
    };
    
    checkIfMobile();
    window.addEventListener('resize', checkIfMobile);
    
    return () => {
      window.removeEventListener('resize', checkIfMobile);
    };
  }, []);

  // Load TTS settings from localStorage
  useEffect(() => {
    const savedTTSEnabled = localStorage.getItem('ttsEnabled') === 'true';
    const savedVoice = localStorage.getItem('ttsVoice');
    
    console.log('Initial TTS enabled state:', savedTTSEnabled);
    setTtsEnabled(savedTTSEnabled);
    if (savedVoice) {
      setSelectedVoice(savedVoice);
    }
    
    // Initialize AudioContext if TTS is enabled
    if (savedTTSEnabled && !audioContextRef.current) {
      try {
        audioContextRef.current = getAudioContext();
        console.log('AudioContext initialized on load:', audioContextRef.current?.state);
      } catch (e) {
        console.error("Error creating AudioContext on load:", e);
      }
    }
  }, []);

  // Make global briefing modal function available
  useEffect(() => {
    window.openBriefingModal = (filename, briefings) => {
      const briefing = briefings.find(b => b.filename === filename) || { title: filename };
      setBriefingModal({
        isOpen: true,
        title: briefing.title,
        filename: filename
      });
    };

    return () => {
      delete window.openBriefingModal;
    };
  }, []);

    // Load conversation history on mount
    useEffect(() => {
      const loadCurrentSession = async () => {
        // Check if chat was cleared during new chat or on initial load
        if (localStorage.getItem('chatCleared') === 'true') {
          localStorage.removeItem('chatCleared');
          setMessages([]);
          setShowEmptyState(true);
          return;
        }
    
        try {
          const response = await fetch('/v1/chat/current-session');
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
    
          const data = await response.json();
          if (data.messages && data.messages.length > 0) {
            setMessages(data.messages);
            setShowEmptyState(false);
          } else {
            setShowEmptyState(true);
          }
        } catch (error) {
          console.error('Error loading current session:', error);
          setShowEmptyState(true);
        } finally {
          isInitialMount.current = false; // Set to false after initial load attempt
        }
      };
    
      // Load session only on initial mount, or after a clear (tracked via ref)
      if (isInitialMount.current) {
        loadCurrentSession();
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Adjust textarea height as user types
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 200);
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [inputValue]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleTtsToggle = (enabled) => {
    console.log('TTS toggle clicked, setting to:', enabled);
    setTtsEnabled(enabled);
    localStorage.setItem('ttsEnabled', String(enabled));
    
    // Initialize AudioContext if enabling TTS
    if (enabled) {
      if (!audioContextRef.current) {
        try {
          audioContextRef.current = getAudioContext();
          console.log('AudioContext initialized on TTS toggle:', audioContextRef.current?.state);
        } catch (e) {
          console.error("Error creating AudioContext:", e);
          if (window.showToast) {
            window.showToast("Audio playback not supported in this browser.", "error");
          }
        }
      } else {
        console.log('Using existing AudioContext, state:', audioContextRef.current.state);
        // Try to resume if suspended
        if (audioContextRef.current.state === 'suspended') {
          audioContextRef.current.resume().then(() => {
            console.log('AudioContext resumed on toggle');
          }).catch(e => {
            console.error('Failed to resume AudioContext on toggle:', e);
          });
        }
      }
    } else {
      console.log('TTS disabled, audio will be cleaned up on next play attempt');
    }
  };

  const handleVoiceChange = (voice) => {
    setSelectedVoice(voice);
    localStorage.setItem('ttsVoice', voice);
  };

  // Updated TTS generator using Web Audio API - similar to BriefingModal
  const generateTTS = async (text) => {
    console.log('generateTTS called with text length:', text?.length);
    console.log('TTS enabled:', ttsEnabled);
    console.log('AudioContext exists:', !!audioContextRef.current);
    
    if (!ttsEnabled) {
      console.log('TTS is disabled, not generating audio');
      return;
    }
    
    if (!text) {
      console.log('No text provided for TTS');
      return;
    }
    
    if (!audioContextRef.current) {
      console.log('No AudioContext available, attempting to create one');
      try {
        audioContextRef.current = getAudioContext();
        console.log('AudioContext created on demand:', audioContextRef.current?.state);
      } catch (e) {
        console.error('Failed to create AudioContext on demand:', e);
        if (window.showToast) {
          window.showToast('Audio playback is not supported in this browser.', 'error');
        }
        return;
      }
    }
    
    // Stop any currently playing audio
    stopAndCleanupAudio();
    
    // Set loading state for audio
    setIsPreparingAudio(true);
    
    try {
      // iOS User Interaction Requirement - Must be sync
      if (audioContextRef.current.state === 'suspended') {
        console.log('AudioContext is suspended, attempting to resume...');
        try {
          await audioContextRef.current.resume();
          console.log('AudioContext resumed successfully.');
        } catch (resumeError) {
          console.error('Failed to resume AudioContext:', resumeError);
          if (window.showToast) {
            window.showToast('Could not enable audio. Please interact with the page again.', 'error');
          }
          setIsPreparingAudio(false);
          return;
        }
      }
      
      // If it's not running, we can't proceed
      if (audioContextRef.current.state !== 'running') {
        console.error('AudioContext is not running:', audioContextRef.current.state);
        if (window.showToast) {
          window.showToast('Audio system is not ready. Please try clicking the text-to-speech toggle again.', 'error');
        }
        setIsPreparingAudio(false);
        return;
      }
      
      // Clean the text for TTS
      const cleanedText = cleanTextForTTS(text);
      console.log(`Cleaned text length: ${cleanedText.length}`);
      
      console.log(`Fetching TTS with voice: ${selectedVoice}`);
      
      // Try the simpler Audio approach first as fallback
      if (window.Audio && !window.AudioContext && !window.webkitAudioContext) {
        console.log('Falling back to simple Audio API');
        const audio = new Audio();
        const blob = await fetch('/v1/tts/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: cleanedText,
            voice: selectedVoice,
            speed: 1.0
          })
        }).then(r => r.blob());
        
        audio.src = URL.createObjectURL(blob);
        audio.play();
        setIsPreparingAudio(false);
        return;
      }
      
      // Web Audio API approach
      const response = await fetch('/v1/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'audio/mpeg' },
        body: JSON.stringify({
          text: cleanedText,
          voice: selectedVoice,
          speed: 1.0
        })
      });
      
      console.log('TTS fetch response status:', response.status);
      
      if (!response.ok) {
        let errorBody = `HTTP error ${response.status}`;
        try { errorBody = (await response.json()).error || errorBody } catch(e){}
        throw new Error(errorBody);
      }
      
      // Get audio data as ArrayBuffer for Web Audio API
      const arrayBuffer = await response.arrayBuffer();
      console.log('TTS ArrayBuffer received, size:', arrayBuffer.byteLength);
      
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        throw new Error("AudioContext was closed unexpectedly.");
      }
      
      // Decode audio data
      console.log('Decoding audio data...');
      const decodedBuffer = await new Promise((resolve, reject) => {
        audioContextRef.current.decodeAudioData(arrayBuffer, resolve, (decodeError) => {
          console.error('Error decoding audio data:', decodeError);
          reject(new Error('Failed to decode audio data'));
        });
      });
      
      console.log('Audio decoded successfully.');
      audioBufferRef.current = decodedBuffer;
      
      // Play the decoded audio
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        throw new Error("AudioContext was closed before playback could start.");
      }
      
      // Stop any previous source before creating a new one
      if (sourceNodeRef.current) {
        try { sourceNodeRef.current.stop(); } catch(e){}
        sourceNodeRef.current = null;
      }
      
      const source = audioContextRef.current.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(audioContextRef.current.destination);
      
      source.onended = () => {
        console.log('Audio source node playback ended.');
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          if (sourceNodeRef.current === source) {
            sourceNodeRef.current = null;
            setIsPreparingAudio(false);
          }
        } else {
          sourceNodeRef.current = null;
          setIsPreparingAudio(false);
        }
      };
      
      sourceNodeRef.current = source;
      setIsPreparingAudio(false);
      source.start(0);
      console.log('Audio source node started.');
      
    } catch (err) {
      console.error('Error in TTS generation process:', err);
      if (window.showToast) {
        window.showToast(`Failed to play speech: ${err.message}`, 'error');
      }
      stopAndCleanupAudio();
    }
  };

  const handleSendMessage = async () => {
    const message = inputValue.trim();
    if (!message || loading) return;
    
    // Clear input
    setInputValue('');
    
    // Add user message to chat
    const newMessages = [
      ...messages,
      { role: 'user', content: message }
    ];
    setMessages(newMessages);
    setShowEmptyState(false);
    
    // Set loading state
    setLoading(true);
    
    try {
      // Prepare the request payload
      const payload = {
        model: "gpt-4o", // This will be mapped to your local model
        messages: newMessages,
        stream: true
      };
      
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Create a reader from the response body stream
      const reader = response.body.getReader();
      let assistantResponse = '';
      let completeMessage = null;
      
      // Function to process stream chunks
      const processStream = async ({ done, value }) => {
        setLoading(false);
        if (done) {
          // Add the complete assistant message if it exists
          if (assistantResponse) {
            const finalMessages = [
              ...newMessages,
              { role: 'assistant', content: assistantResponse }
            ];
            setMessages(finalMessages);
            
            // Generate TTS after the message is complete
            console.log('Stream complete, TTS enabled:', ttsEnabled);
            if (ttsEnabled) {
              // Short timeout to ensure UI is updated first
              setTimeout(() => {
                generateTTS(assistantResponse);
              }, 100);
            }
          }
          
          return;
        }
        
        // Convert the chunk to text
        const chunk = new TextDecoder().decode(value);
        
        // Process SSE format: split by 'data: ' and process each line
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            try {
              const jsonData = JSON.parse(line.substring(6));
              
              // Extract content delta if available
              if (jsonData.choices && jsonData.choices[0] && jsonData.choices[0].delta && jsonData.choices[0].delta.content) {
                const contentDelta = jsonData.choices[0].delta.content;
                assistantResponse += contentDelta;
                
                // Update message with streaming content
                if (!completeMessage) {
                  // Create new assistant message
                  completeMessage = { role: 'assistant', content: assistantResponse };
                  setMessages([...newMessages, completeMessage]);
                } else {
                  // Update existing message
                  completeMessage.content = assistantResponse;
                  setMessages([...newMessages, { ...completeMessage }]);
                }
              }
              
              // Check if stream is finished
              if (jsonData.choices && jsonData.choices[0] && jsonData.choices[0].finish_reason === 'stop') {
              }
            } catch (error) {
              console.error('Error parsing event:', error);
            }
          } else if (line === 'data: [DONE]') {
          }
        }
        
        // Continue reading the stream
        return reader.read().then(processStream);
      };
      
      // Start reading the stream
      reader.read().then(processStream);
    } catch (error) {
      console.error('Fetch error:', error);
      setLoading(false);
      
      // Add error message
      setMessages([
        ...newMessages,
        { 
          role: 'assistant', 
          content: `Sorry, there was an error processing your request: ${error.message}. Please try again.` 
        }
      ]);
    }
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim() && !loading) {
        handleSendMessage();
      }
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the current chat?')) {
      // Synchronously clear local state
      setMessages([]);
      setShowEmptyState(true);
      localStorage.setItem('chatCleared', 'true');

      // Optimistically update the UI (clear the chat)
      setMessages([]);
      setShowEmptyState(true);

      // Clear server-side history
      fetch('/v1/clear-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ conversation_id: 'current_session' })
      })
      .then(response => {
        if (response.ok) {
          console.log('Server-side history cleared successfully');
        } else {
          console.warn('Failed to clear server-side history:', response.statusText);
        }
      })
      .catch(error => {
        console.error('Error clearing server-side history:', error);
      });
    }
  };

    const handleNewChat = () => {
      // Synchronously clear local state
      setMessages([]);
      setShowEmptyState(true);

      // Set localStorage flag synchronously
      localStorage.setItem('chatCleared', 'true');

      // Immediately update UI (clear the chat)
      setMessages([]);
      setShowEmptyState(true);


      // Clear server-side history
      fetch('/v1/clear-conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ conversation_id: 'current_session' })
      })
        .then(response => {
          if (response.ok) {
            console.log('Server-side history cleared successfully');
          } else {
            console.warn('Failed to clear server-side history:', response.statusText);
          }
        })
        .catch(error => {
          console.error('Error clearing server-side history:', error);
        });

      if (isMobile) {
        setShowSidebar(false);
      }
    };

  const handleSuggestionClick = (suggestion) => {
    setInputValue(suggestion);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  return (
    <div className="flex h-screen">
      <Sidebar 
        onNewChat={handleNewChat}
        conversations={[{ id: 'current_session', title: 'Current Conversation', active: true }]}
        isMobile={isMobile}
        showSidebar={showSidebar}
        ttsEnabled={ttsEnabled}
        onTtsToggle={handleTtsToggle}
        selectedVoice={selectedVoice}
        onVoiceChange={handleVoiceChange}
      />
      
      <div className="flex-1 flex flex-col h-screen max-h-screen relative">
        <button 
          className={`absolute top-4 left-4 z-10 p-1 text-muted-color hover:bg-hover-color rounded md:hidden`}
          onClick={() => setShowSidebar(!showSidebar)}
        >
          <Bars3Icon className="w-6 h-6" />
        </button>
        
        <header className="p-4 border-b border-border-color flex items-center justify-center flex-shrink-0">
          <h1 className="text-lg font-semibold">Sara</h1>
          
          <div className="flex items-center gap-2 ml-auto">
            <label className="flex items-center gap-2 cursor-pointer text-muted-color text-sm">
              <svg className="w-4 h-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 6v12"></path>
                <path d="M8 8v8"></path>
                <path d="M16 8v8"></path>
                <path d="M20 10v4"></path>
                <path d="M4 10v4"></path>
              </svg>
              Text-to-Speech
              <div className="relative inline-block w-9 h-5">
                <input
                  type="checkbox"
                  className="opacity-0 w-0 h-0"
                  checked={ttsEnabled}
                  onChange={() => handleTtsToggle(!ttsEnabled)}
                />
                <span className={`absolute cursor-pointer top-0 left-0 right-0 bottom-0 bg-border-color transition-colors rounded-full before:absolute before:content-[''] before:h-4 before:w-4 before:left-0.5 before:bottom-0.5 before:bg-text-color before:transition-transform before:rounded-full ${ttsEnabled ? 'bg-accent-color before:translate-x-4' : ''}`}></span>
              </div>
            </label>
          </div>
        </header>
        
        {showEmptyState ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center mb-16">
            <div className="text-accent-color text-5xl mb-6">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M8 12h8"></path>
                <path d="M12 8v8"></path>
              </svg>
            </div>
            <h2 className="text-2xl font-semibold mb-4">How can I help you today?</h2>
            <p className="text-muted-color mb-6 max-w-md">Type a message to start a conversation, or try one of these suggestions:</p>
            <div className="flex flex-wrap gap-2 justify-center">
              <SuggestionChip 
                text="Explain quantum computing"
                onClick={handleSuggestionClick}
              />
              <SuggestionChip 
                text="Write a short story about time travel"
                onClick={handleSuggestionClick}
              />
              <SuggestionChip 
                text="Help me plan a vacation"
                onClick={handleSuggestionClick}
              />
              <SuggestionChip 
                text="Give me a healthy dinner recipe"
                onClick={handleSuggestionClick}
              />
            </div>
          </div>
        ) : (
          <div 
            className="flex-1 overflow-y-auto pb-20"
            ref={messageContainerRef}
          >
            <div className="py-6">
              {messages.map((message, index) => (
                <div key={index} className={`chat-message ${message.role}`}>
                  <div className={`avatar ${message.role}`}>
                    {message.role === 'user' ? 'U' : 'S'}
                  </div>
                  <div 
                    className="message-content"
                    dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
                  />
                </div>
              ))}
              
              {loading && (
                <div className="chat-message assistant">
                  <div className="avatar assistant">S</div>
                  <div className="flex items-center text-muted-color">
                    <span>Sara is thinking</span>
                    <div className="flex ml-2">
                      <div className="w-1.5 h-1.5 bg-muted-color rounded-full mx-0.5 animate-[pulse_1.5s_infinite_ease-in-out]"></div>
                      <div className="w-1.5 h-1.5 bg-muted-color rounded-full mx-0.5 animate-[pulse_1.5s_0.2s_infinite_ease-in-out]"></div>
                      <div className="w-1.5 h-1.5 bg-muted-color rounded-full mx-0.5 animate-[pulse_1.5s_0.4s_infinite_ease-in-out]"></div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}
        
        <div className="border-t border-border-color p-4 bg-bg-color flex-shrink-0"> {/* Removed fixed positioning classes, added flex-shrink-0 */}
          <div className="relative max-w-3xl mx-auto">
              <textarea
                ref={textareaRef}
                className="w-full bg-input-bg border border-border-color rounded-lg py-3 px-4 pr-12 text-text-color text-sm leading-6 resize-none h-13 max-h-52 overflow-y-auto"
                style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                placeholder="Message Sara..."
                rows="1"
                value={inputValue}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                disabled={loading}
              />
              <button
                className="absolute right-3 bottom-3 text-accent-color hover:text-accent-hover transition-colors disabled:text-muted-color disabled:cursor-not-allowed p-1 rounded"
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || loading}
                type="button"
              >
                {/* SVG Icon */}
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
          </div>
      </div>
      </div>
      
      <BriefingModal
        isOpen={briefingModal.isOpen}
        onClose={() => setBriefingModal({ ...briefingModal, isOpen: false })}
        briefingTitle={briefingModal.title}
        briefingFilename={briefingModal.filename}
      />
    </div>
  );
};

export default ChatPage;
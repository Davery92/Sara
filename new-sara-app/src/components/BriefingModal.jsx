import { useState, useEffect, useRef } from 'react';
import { XMarkIcon, SpeakerWaveIcon, StopIcon } from '@heroicons/react/24/outline';
import ReactMarkdown from 'react-markdown';
import { cleanTextForTTS } from '../utils/formatters';

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


const BriefingModal = ({ isOpen, onClose, briefingTitle, briefingFilename }) => {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isReading, setIsReading] = useState(false);
  const [isPreparingAudio, setIsPreparingAudio] = useState(false); // New state for fetching/decoding

  // Refs for Web Audio API objects
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null); // To store the current playing sound source
  const audioBufferRef = useRef(null); // To potentially store the decoded buffer if needed for replaying (though source nodes are one-time use)

  // --- Initialize Audio Context ---
  // We try to initialize it once when the modal could potentially be used.
  // It MUST be resumed within a user interaction later.
   useEffect(() => {
     if (isOpen && !audioContextRef.current) {
        try {
            // Attempt to create lazily - might still need resume() on click
            audioContextRef.current = getAudioContext();
            console.log('AudioContext initialized state:', audioContextRef.current?.state);
        } catch (e) {
             console.error("Error creating AudioContext:", e);
             setError("Audio playback not supported or failed to initialize.");
        }
     }
   }, [isOpen]); // Create when modal opens if not already created


  // --- Fetch Briefing Content Effect ---
  useEffect(() => {
    if (isOpen && briefingFilename) {
      loadBriefingContent(briefingFilename);
    } else if (!isOpen) {
        // Reset when closing
        setContent('');
        setError(null);
        setLoading(false);
        stopAndCleanupAudio(); // Ensure audio stops when modal closes
    }
  }, [isOpen, briefingFilename]);

  // --- Audio Cleanup Effect ---
  useEffect(() => {
    // Cleanup function runs when component unmounts
    return () => {
      stopAndCleanupAudio(true); // Pass true to indicate final close
    };
  }, []); // Empty dependency array: runs only on unmount

  const stopAndCleanupAudio = (isUnmounting = false) => {
      console.log('Stopping and cleaning up audio...');
      if (sourceNodeRef.current) {
        try {
            sourceNodeRef.current.onended = null; // Remove listener to prevent state changes after explicit stop
            sourceNodeRef.current.stop();
            console.log('Audio source node stopped.');
        } catch (e) {
            console.warn("Error stopping source node (might have already finished):", e);
        }
        sourceNodeRef.current = null;
      }

      // Reset states
      setIsReading(false);
      setIsPreparingAudio(false);
      audioBufferRef.current = null; // Clear buffer cache

      // Close the context completely only when the component unmounts
      // or maybe when the modal closes definitively? Let's try on unmount.
      if (isUnmounting && audioContextRef.current) {
         if (audioContextRef.current.state !== 'closed') {
             audioContextRef.current.close().then(() => {
                 console.log('AudioContext closed.');
                 audioContextRef.current = null;
             }).catch(e => console.error("Error closing AudioContext:", e));
         } else {
              audioContextRef.current = null; // Already closed
         }
      }
  };


  const loadBriefingContent = async (filename) => {
    stopAndCleanupAudio(); // Stop previous audio
    try {
      setLoading(true);
      setError(null);
      setContent('');

      const response = await fetch(`/v1/briefings/content/${filename}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setContent(data.content || '');
      }
    } catch (err) {
      setError(`Error loading briefing content: ${err.message}`);
      console.error('Error fetching briefing content:', err);
    } finally {
      setLoading(false);
    }
  };

  // --- Main TTS and Playback Function ---
  const handleReadBriefing = async () => {
    // --- STOP Action ---
    if (isReading || isPreparingAudio) {
      stopAndCleanupAudio();
      return;
    }

    // --- START Action ---
    if (!content || loading || error || !audioContextRef.current) {
      console.warn('Cannot read briefing: No content, loading, error, or AudioContext unavailable.');
       if (!audioContextRef.current && window.showToast) {
          window.showToast("Audio playback is not available or supported.", "error");
       }
      return;
    }

    // --- iOS User Interaction Requirement ---
    // Resume context if suspended - THIS MUST BE SYNC
    if (audioContextRef.current.state === 'suspended') {
      try {
        await audioContextRef.current.resume();
        console.log('AudioContext resumed successfully.');
      } catch (resumeError) {
        console.error('Failed to resume AudioContext:', resumeError);
        if (window.showToast) {
            window.showToast('Could not enable audio. Please interact with the page again.', 'error');
        }
        // Don't proceed if context can't be resumed
        return;
      }
    }
    // If it's running, we're good to proceed with async fetch/decode
    if (audioContextRef.current.state !== 'running') {
        console.error('AudioContext is not running:', audioContextRef.current.state);
         if (window.showToast) {
             window.showToast('Audio system is not ready.', 'error');
         }
        return;
    }

    // --- Set Loading State for Audio ---
    setIsPreparingAudio(true); // Indicate fetching/decoding start

    try {
      const cleanedText = cleanTextForTTS(content);
      const savedVoice = localStorage.getItem('ttsVoice');
      const voiceToUse = savedVoice || 'af_bella';

      console.log(`Fetching TTS with voice: ${voiceToUse}`);

      const response = await fetch('/v1/tts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'audio/mpeg'}, // Request specific audio type if possible
        body: JSON.stringify({ text: cleanedText, voice: voiceToUse, speed: 1.0 }),
      });

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

      // --- Decode Audio Data ---
      // Use Promise wrapper for decodeAudioData for cleaner async/await
      const decodedBuffer = await new Promise((resolve, reject) => {
          audioContextRef.current.decodeAudioData(arrayBuffer, resolve, (decodeError) => {
              console.error('Error decoding audio data:', decodeError);
              reject(new Error('Failed to decode audio data'));
          });
      });

      console.log('Audio decoded successfully.');
      audioBufferRef.current = decodedBuffer; // Cache the buffer if needed

      // --- Play the Decoded Audio ---
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
          throw new Error("AudioContext was closed before playback could start.");
      }

      // Stop any previous source *before* creating a new one
      if (sourceNodeRef.current) {
         try { sourceNodeRef.current.stop(); } catch(e){}
         sourceNodeRef.current = null;
      }

      const source = audioContextRef.current.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(audioContextRef.current.destination);

      source.onended = () => {
        console.log('Audio source node playback ended.');
        // Check if the context still exists before trying to access state
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
           // Clean up ref, set state *only if* it ended naturally
           // Check ref to ensure it wasn't stopped manually already
           if (sourceNodeRef.current === source) {
               sourceNodeRef.current = null;
               setIsReading(false);
               setIsPreparingAudio(false);
           }
        } else {
            // Context closed or missing, ensure states are reset
            sourceNodeRef.current = null;
            setIsReading(false);
            setIsPreparingAudio(false);
        }
      };

      sourceNodeRef.current = source; // Store the reference
      setIsPreparingAudio(false); // Decoding finished
      setIsReading(true); // Playback starting now
      source.start(0); // Play immediately
      console.log('Audio source node started.');


    } catch (err) {
      console.error('Error in handleReadBriefing process:', err);
      if (window.showToast) {
        window.showToast(`Failed to play speech: ${err.message}`, 'error');
      }
      stopAndCleanupAudio(); // Clean up on any error in the process
    }
  };


  if (!isOpen) return null;

  // Determine button state and icon
  const getButtonState = () => {
    if (isPreparingAudio) return { icon: <div className="w-5 h-5 border-2 border-t-accent-color rounded-full animate-spin"></div>, label: "Preparing audio...", title: "Cancel", disabled: false, class: 'text-accent-color' };
    if (isReading) return { icon: <StopIcon className="w-5 h-5" />, label: "Stop reading", title: "Stop reading", disabled: false, class: 'text-accent-color animate-pulse' };
    // Default: Play button
    const canPlay = !loading && !error && !!content && !!audioContextRef.current;
    return { icon: <SpeakerWaveIcon className="w-5 h-5" />, label: "Read briefing aloud", title: "Read briefing aloud", disabled: !canPlay, class: canPlay ? 'text-muted-color hover:text-text-color' : 'text-muted-color opacity-50' };
  };
  const buttonState = getButtonState();


  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50 p-4">
      <div className="bg-neutral-900 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col border border-neutral-700">
        <div className="p-4 border-b border-border-color flex items-center justify-between">
          <h3 className="text-lg font-semibold">{briefingTitle || 'Briefing'}</h3>
          <div className="flex items-center gap-2">
            {/* Updated TTS Button */}
            <button
              className={`p-1.5 rounded hover:bg-hover-color transition-colors ${buttonState.class} ${buttonState.disabled ? 'cursor-not-allowed' : ''}`}
              onClick={handleReadBriefing}
              disabled={buttonState.disabled && !isReading && !isPreparingAudio} // Only truly disable if cannot play
              title={buttonState.title}
              type="button"
              aria-label={buttonState.label}
            >
              {buttonState.icon}
            </button>

            {/* Close Button */}
            <button
              className="text-muted-color p-1 rounded hover:text-text-color hover:bg-hover-color transition-colors"
              onClick={() => {
                 stopAndCleanupAudio(); // Stop audio on explicit close
                 onClose();
              }}
              aria-label="Close briefing modal"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto flex-grow">
          {loading ? (
            <div className="text-center py-4">
              <div className="w-8 h-8 border-4 border-t-accent-color rounded-full animate-spin mx-auto"></div>
              <p className="mt-2 text-muted-color">Loading briefing...</p>
            </div>
          ) : error ? (
            <div className="text-error-color p-4 bg-error-bg-color rounded">
              <p className="font-medium">Error loading briefing</p>
              <p className="mt-1 text-sm">{error}</p>
            </div>
          ) : content ? (
            <div className="prose prose-sm sm:prose-base prose-invert max-w-none">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          ) : (
            <div className="text-center py-4 text-muted-color">No briefing content available.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BriefingModal;
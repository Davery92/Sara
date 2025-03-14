// renderer.js - Client-side application logic

// Initialize settings and marked parser when the document is loaded
document.addEventListener('DOMContentLoaded', async function() {
  // DOM Elements
  const sidebar = document.getElementById('sidebar');
  const menuButton = document.getElementById('menu-button');
  const newChatBtn = document.getElementById('new-chat-btn');
  const clearChatBtn = document.getElementById('clear-chat');
  const settingsBtn = document.getElementById('settings-btn');
  const emptyState = document.getElementById('empty-state');
  const chatContainer = document.getElementById('chat-container');
  const messageInput = document.getElementById('message-input');
  const sendButton = document.getElementById('send-button');
  const messagesWrapper = document.getElementById('messages-wrapper');
  const connectionStatus = document.getElementById('connection-status');
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.querySelector('.status-text');
  const suggestionChips = document.querySelectorAll('.suggestion-chip');
  
  // Settings Modal Elements
  const settingsModal = document.getElementById('settings-modal');
  const closeSettingsBtn = document.getElementById('close-settings');
  const serverUrlInput = document.getElementById('server-url');
  const saveSettingsBtn = document.getElementById('save-settings');
  const testConnectionBtn = document.getElementById('test-connection');

  // TTS elements and state
  const ttsToggle = document.getElementById('tts-toggle');
  const voiceSelector = document.getElementById('tts-voice');
  const voiceSelectorContainer = document.getElementById('voice-selector-container');
  
  // Chat management variables
  let loadingResponse = false;
  let chatHistory = [];
  let serverUrl = await window.api.getServerUrl();

  // TTS state
  let ttsEnabled = false;
  let currentTTSAudio = null;
  let isPlayingTTS = false;
  let lastAssistantMessage = null;
  let messageCounter = 0;
  
  // Set the initial server URL in the input
  serverUrlInput.value = serverUrl;
  
  // Test connection on startup
  testConnection();

  if (ttsToggle) {
    ttsToggle.addEventListener('change', function() {
      if (this.checked) {
        // TTS was just enabled, process the latest message
        console.log("TTS enabled, processing latest message");
        setTimeout(forceTTSForLatestMessage, 500);
      }
    });
  }

  const style = document.createElement('style');
  style.textContent = `
    #voice-selector-container {
      display: none !important;
    }
  `;
  document.head.appendChild(style);
  // Toggle sidebar on mobile
  menuButton.addEventListener('click', function() {
      sidebar.classList.toggle('show');
  });

  // Adjust textarea height as user types
  messageInput.addEventListener('input', function() {
      // Reset height to auto to get correct scrollHeight
      this.style.height = 'auto';
      
      // Set the height to scrollHeight + padding
      const newHeight = Math.min(this.scrollHeight, 200);
      this.style.height = `${newHeight}px`;
      
      // Enable/disable send button based on input
      sendButton.disabled = this.value.trim().length === 0;
  });

  // Send message on Enter (but allow Shift+Enter for new lines)
  messageInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (!sendButton.disabled && !loadingResponse) {
              sendMessage();
          }
      }
  });

  // Handle send button click
  sendButton.addEventListener('click', function() {
      if (!sendButton.disabled && !loadingResponse) {
          sendMessage();
      }
  });

  // Handle suggestion chips
  suggestionChips.forEach(chip => {
      chip.addEventListener('click', function() {
          messageInput.value = this.textContent;
          messageInput.dispatchEvent(new Event('input'));
          sendMessage();
      });
  });

  // Create a new chat
  newChatBtn.addEventListener('click', function() {
      startNewChat();
  });

  // Clear chat
  clearChatBtn.addEventListener('click', function() {
      if (confirm('Are you sure you want to clear the current chat?')) {
          clearChat();
      }
  });
  
  // Settings button
  settingsBtn.addEventListener('click', function() {
      openSettings();
  });
  
  // Close settings modal
  closeSettingsBtn.addEventListener('click', function() {
      settingsModal.classList.remove('show');
  });
  
  // Save settings
  saveSettingsBtn.addEventListener('click', async function() {
      const newServerUrl = serverUrlInput.value.trim();
      if (newServerUrl) {
          serverUrl = newServerUrl;
          await window.api.setServerUrl(serverUrl);
          settingsModal.classList.remove('show');
          showNotification('Settings saved successfully', 'success');
          testConnection();
      } else {
          showNotification('Please enter a valid server URL', 'error');
      }
  });
  
  // Test connection
  testConnectionBtn.addEventListener('click', function() {
      testConnection(true); // true means show notification even if successful
  });
  
  // Listen for IPC events
  window.api.onNewChat(() => {
      startNewChat();
  });
  
  window.api.onClearChat(() => {
      clearChat();
  });
  
  window.api.onOpenSettings(() => {
      openSettings();
  });
  
  // Initialize the chat interface
  initializeChat();

  // Initialize TTS
  setupTTS();

  // Function to test connection to the server
  async function testConnection(showSuccessNotification = false) {
      try {
          const response = await fetch(`${serverUrl}/health`);
          
          if (response.ok) {
              // Update connection status indicator
              statusDot.classList.remove('offline');
              statusText.textContent = 'Connected';
              
              if (showSuccessNotification) {
                  showNotification('Successfully connected to the server', 'success');
              }
              
              return true;
          } else {
              throw new Error(`Server returned ${response.status}`);
          }
      } catch (error) {
          console.error('Connection test failed:', error);
          
          // Update connection status indicator
          statusDot.classList.add('offline');
          statusText.textContent = 'Disconnected';
          
          showNotification('Could not connect to the server', 'error');
          return false;
      }
  }

  // Function to initialize the chat interface
  async function initializeChat() {
      try {
          const response = await fetch(`${serverUrl}/v1/chat/current-session`);
          
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data = await response.json();
          
          if (data && data.messages && data.messages.length > 0) {
              // Store the messages in chat history
              chatHistory = data.messages;
              
              // Show chat container and hide empty state
              emptyState.classList.add('hidden');
              chatContainer.classList.remove('hidden');
              
              // Render the messages
              renderMessages();
              
              console.log(`Loaded ${chatHistory.length} messages from current session`);
          } else {
              // If no messages, show empty state
              console.log('No messages found in current session');
              emptyState.classList.remove('hidden');
              chatContainer.classList.add('hidden');
          }
      } catch (error) {
          console.error('Error loading current session:', error);
          // Show empty state on error
          emptyState.classList.remove('hidden');
          chatContainer.classList.add('hidden');
      }
  }

  // Function to start a new chat
  function startNewChat() {
      // Clear current messages
      messagesWrapper.innerHTML = '';
      chatHistory = [];
      
      // Show empty state
      chatContainer.classList.add('hidden');
      emptyState.classList.remove('hidden');
      
      // Close sidebar on mobile
      sidebar.classList.remove('show');
      
      // Focus the input
      messageInput.focus();
      
      // Clear the input field if there's any text
      messageInput.value = '';
      messageInput.style.height = 'auto';
      sendButton.disabled = true;
      
      // Make an API call to clear server-side history
      fetch(`${serverUrl}/v1/clear-conversation`, {
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
      
      console.log('Started new chat - history cleared');
  }

  // Function to clear chat
  function clearChat() {
      messagesWrapper.innerHTML = '';
      chatHistory = [];
      
      chatContainer.classList.add('hidden');
      emptyState.classList.remove('hidden');
      
      // Clear server-side history
      fetch(`${serverUrl}/v1/clear-conversation`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ conversation_id: 'current_session' })
      })
      .catch(error => {
          console.error('Error clearing server-side history:', error);
      });
  }

  // Function to open settings modal
  function openSettings() {
      serverUrlInput.value = serverUrl;
      settingsModal.classList.add('show');
  }

  // Function to fix scrolling issues
  function fixScrolling() {
      // Force scroll to bottom with delay to ensure DOM is updated
      setTimeout(() => {
          // Check if chat container is visible first
          if (!chatContainer.classList.contains('hidden')) {
              chatContainer.scrollTop = chatContainer.scrollHeight;
          }
      }, 100);

      setTimeout(() => {
          if (!chatContainer.classList.contains('hidden')) {
              chatContainer.scrollTop = chatContainer.scrollHeight;
          }
      }, 300);
  }

  // Function to render messages
  function renderMessages() {
      // Clear current messages
      messagesWrapper.innerHTML = '';
      
      // Check if we have messages to render
      if (!chatHistory || chatHistory.length === 0) {
          console.log('No messages to render');
          return;
      }
      
      console.log(`Rendering ${chatHistory.length} messages`);
      
      // Add each message
      chatHistory.forEach(msg => {
          // Make sure we have valid message data
          if (!msg || !msg.role || !msg.content) {
              console.warn('Invalid message data:', msg);
              return;
          }
          
          // Only add user and assistant messages (skip system, tool, etc.)
          if (msg.role === 'user' || msg.role === 'assistant') {
              // Add message but mark it as not new (isNew = false)
              addMessageToChat(msg.role, msg.content, false);
          }
      });
      
      // Ensure chat container is visible
      emptyState.classList.add('hidden');
      chatContainer.classList.remove('hidden');
      
      // Fix scrolling issues
      fixScrolling();
      
      console.log('Messages rendered successfully');
  }

  function clearAudioCache() {
    // Clear the current audio reference
    if (currentTTSAudio) {
      currentTTSAudio.pause();
      currentTTSAudio.src = '';
      currentTTSAudio = null;
    }
    
    console.log("Audio cache cleared");
  }

  function forceTTSForLatestMessage() {
    console.log("Forcing TTS for latest assistant message");
    
    if (!ttsEnabled) {
      console.log("TTS is disabled, can't force TTS");
      return false;
    }
    
    // Find the latest assistant message
    const assistantMessages = document.querySelectorAll('.chat-message.assistant');
    if (assistantMessages.length === 0) {
      console.log("No assistant messages found");
      return false;
    }
    
    // Get the last assistant message element
    const latestMessage = assistantMessages[assistantMessages.length - 1];
    
    // Get the message content
    const contentElement = latestMessage.querySelector('.message-content');
    if (!contentElement) {
      console.log("No content element found in the message");
      return false;
    }
    
    // Extract text (strip HTML)
    let messageText = contentElement.textContent || contentElement.innerText;
    
    if (!messageText || messageText.trim().length === 0) {
      console.log("No text content in the message");
      return false;
    }
    
    console.log(`Found latest assistant message with ${messageText.length} characters`);
    
    // Generate TTS using our simplified function
    generateAndPlayTTS(messageText, latestMessage);
    
    return true;
  }
  // Function to send a message to the server
  // Critical fix for sendMessage function to ensure the latest assistant message gets TTS
  // Replace the sendMessage function in renderer.js with this updated version:

  async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || loadingResponse) return;
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendButton.disabled = true;
  
    // Show chat container
    emptyState.classList.add('hidden');
    chatContainer.classList.remove('hidden');
    
    // Add user message to chat
    addMessageToChat('user', message, true);
    
    // Set loading state
    loadingResponse = true;
    addLoadingIndicator();
    
    // Fix scrolling after adding user message
    fixScrolling();
    
    // Add to chat history array
    chatHistory.push({ role: 'user', content: message });
    
    // Prepare the request payload
    const payload = {
        model: "gpt-4o", // This will be mapped to the local model
        messages: chatHistory.filter(msg => msg.role === 'user' || msg.role === 'assistant'),
        stream: true
    };
    
    try {
        // Increment message counter to create a unique ID for this exchange
        const currentMessageId = ++messageCounter;
        console.log(`Starting message exchange #${currentMessageId}`);
        
        // Create the fetch request for streaming
        const response = await fetch(`${serverUrl}/v1/chat/completions`, {
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
        let assistantMessageElement = null;
        
        // Process stream chunks
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                loadingResponse = false;
                break;
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
                            
                            // If this is the first chunk, create the assistant message
                            if (!assistantMessageElement) {
                                // Remove loading indicator
                                removeLoadingIndicator();
                                
                                // Add assistant message with the current message ID
                                assistantMessageElement = addMessageToChat('assistant', assistantResponse, true);
                                
                                // Set a custom attribute to track which message exchange this belongs to
                                assistantMessageElement.setAttribute('data-message-id', currentMessageId);
                                
                                console.log(`Created new assistant message #${currentMessageId}`);
                            } else {
                                // Update the message content
                                const contentElement = assistantMessageElement.querySelector('.message-content');
                                
                                // Process markdown (simplified for this example)
                                contentElement.innerHTML = formatMessage(assistantResponse);
                                
                                // Fix scrolling with each update
                                fixScrolling();
                            }
                        }
                        
                        // Check if stream is finished
                        if (jsonData.choices && jsonData.choices[0] && jsonData.choices[0].finish_reason === 'stop') {
                            loadingResponse = false;
                        }
                    } catch (error) {
                        console.error('Error parsing event:', error);
                    }
                } else if (line === 'data: [DONE]') {
                    loadingResponse = false;
                }
            }
        }
        
        // Update chat history with the assistant's response if we received one
        if (assistantResponse) {
            chatHistory.push({ role: 'assistant', content: assistantResponse });
            
            // Generate TTS after the full message is complete
            if (ttsEnabled && assistantMessageElement) {
                console.log(`Generating TTS for complete message #${currentMessageId}`);
                // Brief timeout to ensure rendering is complete
                setTimeout(() => {
                    // Simple TTS function that doesn't over-complicate things
                    generateAndPlayTTS(assistantResponse, assistantMessageElement);
                }, 100);
            }
        }
        
    } catch (error) {
        console.error('Fetch error:', error);
        loadingResponse = false;
        removeLoadingIndicator();
        
        // Show error message
        showNotification('Error communicating with the server', 'error');
        addMessageToChat('assistant', `I'm sorry, I couldn't connect to the server. Please check your connection and try again. Error: ${error.message}`, true);
    }
  }

  function playTTS(text) {
    if (!ttsEnabled || !text) {
        console.log("TTS disabled or no text to speak");
        return;
    }
    
    console.log(`Generating speech for text: ${text.substring(0, 50)}...`);
    
    // Clean the text for TTS
    const cleanedText = text.replace(/```[\s\S]*?```/g, '') // Remove code blocks
                          .replace(/`([^`]+)`/g, '$1')     // Remove inline code
                          .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
                          .replace(/\*([^*]+)\*/g, '$1')   // Remove italic
                          .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Replace links
                          .replace(/\n/g, ' ');            // Replace newlines
    
    // Get selected voice
    const voice = voiceSelector.value;
    
    // Generate the speech
    window.tts.generateSpeech(cleanedText, voice, 1.0)
        .then(audioPath => {
            console.log(`TTS generated successfully: ${audioPath}`);
            
            // Create and play audio
            const audio = new Audio();
            audio.src = 'file://' + audioPath;
            
            // Store as current audio
            if (currentTTSAudio) {
                currentTTSAudio.pause();
                currentTTSAudio.currentTime = 0;
            }
            currentTTSAudio = audio;
            
            // Play the audio
            audio.play()
                .then(() => console.log("TTS playing"))
                .catch(err => console.error("Error playing TTS:", err));
                    
            // Add ended event listener
            audio.addEventListener('ended', function() {
                console.log("TTS playback complete");
            });
            
            // Add UI for this audio to the last assistant message
            if (lastAssistantMessage) {
                // Find or create audio controls in this message
                addAudioControlsToMessage(lastAssistantMessage, audio, audioPath);
            }
        })
        .catch(error => {
            console.error('TTS generation error:', error);
            showNotification('Error generating speech', 'error');
        });
  }

  function addAudioControlsToMessage(messageElement, audio, audioPath) {
    if (!messageElement) {
      console.error("No message element provided to add audio controls");
      return;
    }
    
    // Check if audio controls already exist
    let audioControls = messageElement.querySelector('.audio-controls');
    if (audioControls) {
      console.log("Audio controls already exist, updating them");
      
      // Update existing controls
      const playButton = audioControls.querySelector('.play-button');
      const audioStatus = audioControls.querySelector('.audio-status');
      
      // Reset UI to ready state
      playButton.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      `;
      playButton.disabled = false;
      audioStatus.textContent = 'Ready to play';
      
      // Store audio path for debugging
      messageElement.setAttribute('data-audio-path', audioPath);
      
      // Clear old event listeners by cloning and replacing the button
      const newPlayButton = playButton.cloneNode(true);
      playButton.parentNode.replaceChild(newPlayButton, playButton);
      
      // Set up event listener for the play button
      setupPlayButtonHandlers(newPlayButton, audioStatus, audio);
      
      return audioControls;
    }
    
    // Create new audio controls
    console.log("Creating new audio controls");
    audioControls = document.createElement('div');
    audioControls.className = 'audio-controls';
    audioControls.innerHTML = `
      <button class="play-button">
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      </button>
      <div class="audio-status">Ready to play</div>
    `;
    
    // Store audio path for debugging
    messageElement.setAttribute('data-audio-path', audioPath);
    
    // Find the content element to append to
    const contentElement = messageElement.querySelector('.message-content');
    if (contentElement) {
      contentElement.appendChild(audioControls);
    } else {
      // Fallback - append directly to message element
      messageElement.appendChild(audioControls);
    }
    
    // Get references to buttons and status
    const playButton = audioControls.querySelector('.play-button');
    const audioStatus = audioControls.querySelector('.audio-status');
    
    // Set up event listener for the play button
    setupPlayButtonHandlers(playButton, audioStatus, audio);
    
    return audioControls;
  }
  function setupPlayButtonHandlers(playButton, audioStatus, audio) {
    // Add click handler to play button
    playButton.addEventListener('click', function() {
      // Make sure audio is valid
      if (!audio || !audio.src) {
        console.error("Invalid audio object for playback");
        audioStatus.textContent = 'Audio unavailable';
        return;
      }
      
      // Check if audio is paused or ended
      if (audio.paused || audio.ended) {
        console.log("Starting audio playback on button click");
        
        // Stop any currently playing audio
        if (currentTTSAudio && currentTTSAudio !== audio) {
          try {
            currentTTSAudio.pause();
            currentTTSAudio.currentTime = 0;
          } catch (error) {
            console.error("Error stopping previous audio:", error);
          }
        }
        
        // Set as current audio
        currentTTSAudio = audio;
        
        // Play the audio with error handling
        audio.play()
          .then(() => {
            console.log("Audio playback started");
          })
          .catch(playError => {
            console.error("Error playing audio:", playError);
            audioStatus.textContent = 'Failed to play';
            
            // Try to play again with a slight delay as a fallback
            setTimeout(() => {
              audio.play().catch(retryError => {
                console.error("Retry failed:", retryError);
              });
            }, 500);
          });
        
        // Update UI for playing state
        playButton.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="4" width="4" height="16"></rect>
            <rect x="14" y="4" width="4" height="16"></rect>
          </svg>
        `;
        audioStatus.textContent = 'Playing...';
        
      } else {
        // Pause if already playing
        console.log("Pausing audio playback");
        audio.pause();
        
        // Update UI for paused state
        playButton.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
        `;
        audioStatus.textContent = 'Paused';
      }
    });
    
    // Also listen for audio events to update UI appropriately
    if (audio) {
      audio.addEventListener('ended', function() {
        console.log("Audio ended event");
        // Reset play button
        playButton.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
        `;
        audioStatus.textContent = 'Ready to play';
      });
      
      audio.addEventListener('error', function(e) {
        console.error("Audio error event:", e);
        playButton.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="red" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        `;
        audioStatus.textContent = 'Audio error';
      });
    }
  }
  // Function to add a message to the chat
  function addMessageToChat(role, content, isNew = false) {
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = `chat-message ${role}`;
    
    // Create avatar
    const avatar = document.createElement('div');
    avatar.className = `avatar ${role}`;
    
    // Set avatar content (first letter of role)
    avatar.textContent = role === 'user' ? 'U' : 'S'; // U for User, S for Sara
    
    // Create message content container
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content markdown';
    
    // Format the message content (handle markdown, code blocks, etc.)
    messageContent.innerHTML = formatMessage(content);
    
    // Append avatar and content to message
    messageElement.appendChild(avatar);
    messageElement.appendChild(messageContent);
    
    // Append message to chat container
    messagesWrapper.appendChild(messageElement);
    
    // Update chat history if this is a new message
    if (isNew && role === 'assistant') {
        chatHistory.push({ role: 'assistant', content: content });
    }
    
    // Fix scrolling
    fixScrolling();
    
    // IMPORTANT: We've removed the TTS code from here since we now handle it
    // in the sendMessage function using the new playTTS function
    
    // Return the message element (useful for streaming updates)
    return messageElement;
}

async function setupTTS() {
  try {
    console.log("Setting up TTS...");
    
    // Get TTS preferences (only the enabled state matters now)
    const prefs = await window.tts.getPreferences();
    console.log("TTS preferences loaded:", prefs);
    
    // Set initial state
    ttsEnabled = prefs.enabled;
    ttsToggle.checked = ttsEnabled;
    
    console.log("TTS enabled:", ttsEnabled);
    
    // Hide the voice selector container since we don't need it anymore
    if (voiceSelectorContainer) {
      voiceSelectorContainer.style.display = 'none';
    }
    
    // Check TTS service status if enabled
    if (ttsEnabled) {
      try {
        const status = await window.tts.checkStatus();
        console.log("TTS service status:", status);
        if (!status) {
          showNotification('TTS service is not available', 'error');
        }
      } catch (statusError) {
        console.error("Error checking TTS status:", statusError);
        showNotification('Could not check TTS service status', 'error');
      }
    }
    
    // Toggle TTS when the switch is clicked
    ttsToggle.addEventListener('change', function() {
      ttsEnabled = this.checked;
      console.log("TTS toggled to:", ttsEnabled);
      
      // Save preference (always use af_bella voice)
      window.tts.savePreferences(ttsEnabled, 'af_bella', 1.0)
        .then(() => {
          console.log("TTS preferences saved");
        })
        .catch(err => {
          console.error("Error saving TTS preferences:", err);
        });
      
      if (ttsEnabled) {
        // Check TTS service status
        window.tts.checkStatus()
          .then(status => {
            console.log("TTS service status after toggle:", status);
            if (!status) {
              showNotification('TTS service is not available', 'error');
            }
          })
          .catch(err => {
            console.error("Error checking TTS status after toggle:", err);
          });
      }
    });
    
    console.log("TTS setup complete");
  } catch (error) {
    console.error('Error setting up TTS:', error);
    showNotification('Failed to set up TTS', 'error');
  }
}
    
    // Function to clean text for TTS
    function cleanTextForTTS(text) {
      // Remove markdown formatting
      text = text.replace(/```[\s\S]*?```/g, ''); // Remove code blocks
      text = text.replace(/`([^`]+)`/g, '$1'); // Remove inline code
      text = text.replace(/\*\*([^*]+)\*\*/g, '$1'); // Remove bold
      text = text.replace(/\*([^*]+)\*/g, '$1'); // Remove italic
      text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1'); // Replace links with just the text
      text = text.replace(/\n/g, ' '); // Replace newlines with spaces
      
      return text;
    }
    
    // Function to generate and play TTS for a given text
    // Enhanced generateTTS function with improved logging and error handling
    function generateAndPlayTTS(text, messageElement) {
      if (!ttsEnabled || !text) {
        console.log("TTS disabled or no text to speak");
        return;
      }
      
      console.log(`Generating speech for text (length: ${text.length})`);
      
      // Clean the text for TTS - keeping the original cleaning logic
      const cleanedText = text.replace(/```[\s\S]*?```/g, '') // Remove code blocks
                            .replace(/`([^`]+)`/g, '$1')     // Remove inline code
                            .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
                            .replace(/\*([^*]+)\*/g, '$1')   // Remove italic
                            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Replace links
                            .replace(/\n/g, ' ');            // Replace newlines
      
      // Get selected voice (always using af_bella as per main.js)
      const voice = "af_bella";
      
      // Generate the speech
      window.tts.generateSpeech(cleanedText, voice, 1.0)
        .then(audioPath => {
          console.log(`TTS generated successfully: ${audioPath}`);
          
          // Make sure we have a valid message element
          if (!messageElement) {
            console.error("No message element available for TTS controls");
            return;
          }
          
          // Create or update audio controls
          let audioControls = messageElement.querySelector('.audio-controls');
          if (!audioControls) {
            // Create new controls if they don't exist
            audioControls = document.createElement('div');
            audioControls.className = 'audio-controls';
            audioControls.innerHTML = `
              <button class="play-button">
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
              </button>
              <div class="audio-status">Ready to play</div>
            `;
            
            // Append to message content
            messageElement.querySelector('.message-content').appendChild(audioControls);
          } else {
            // Update existing controls
            const playButton = audioControls.querySelector('.play-button');
            const audioStatus = audioControls.querySelector('.audio-status');
            
            playButton.innerHTML = `
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
            `;
            playButton.disabled = false;
            audioStatus.textContent = 'Ready to play';
          }
          
          // Get references to buttons and status
          const playButton = audioControls.querySelector('.play-button');
          const audioStatus = audioControls.querySelector('.audio-status');
          
          // Create a new audio object - keeping it simple like the original
          const audio = new Audio();
          audio.src = 'file://' + audioPath;  // Use the original format that worked
          
          // Stop any currently playing audio
          if (currentTTSAudio) {
            currentTTSAudio.pause();
            currentTTSAudio.currentTime = 0;
          }
          
          // Update the current audio reference
          currentTTSAudio = audio;
          
          // Add ended event listener
          audio.addEventListener('ended', function() {
            console.log("TTS playback complete");
            // Reset play button
            playButton.innerHTML = `
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
            `;
            audioStatus.textContent = 'Ready to play';
          });
          
          // Add click handler to play button
          playButton.onclick = function() {
            if (audio.paused || audio.ended) {
              // Play the audio
              audio.play()
                .then(() => console.log("TTS playing"))
                .catch(err => {
                  console.error("Error playing TTS:", err);
                  audioStatus.textContent = 'Failed to play';
                });
              
              // Update UI for playing state
              playButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="6" y="4" width="4" height="16"></rect>
                  <rect x="14" y="4" width="4" height="16"></rect>
                </svg>
              `;
              audioStatus.textContent = 'Playing...';
            } else {
              // Pause if already playing
              audio.pause();
              
              // Update UI for paused state
              playButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
              `;
              audioStatus.textContent = 'Paused';
            }
          };
          
          // Auto-play if enabled
          if (ttsEnabled) {
            console.log("Auto-playing TTS audio");
            
            // Play the audio
            audio.play()
              .then(() => {
                console.log("Auto-playing TTS");
                // Update UI for playing state
                playButton.innerHTML = `
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="6" y="4" width="4" height="16"></rect>
                    <rect x="14" y="4" width="4" height="16"></rect>
                  </svg>
                `;
                audioStatus.textContent = 'Playing...';
              })
              .catch(err => {
                console.error("Error auto-playing TTS:", err);
                audioStatus.textContent = 'Failed to auto-play';
              });
          }
        })
        .catch(error => {
          console.error('TTS generation error:', error);
          showNotification('Error generating speech', 'error');
        });
    }


  // Function to add loading indicator
  function addLoadingIndicator() {
      const loadingElement = document.createElement('div');
      loadingElement.className = 'chat-message assistant loading';
      loadingElement.id = 'loading-indicator';
      
      const avatar = document.createElement('div');
      avatar.className = 'avatar assistant';
      avatar.textContent = 'S';
      
      const loadingContent = document.createElement('div');
      loadingContent.className = 'loading-indicator';
      loadingContent.innerHTML = `
          <span>Sara is thinking</span>
          <div class="loading-dots">
              <div class="dot"></div>
              <div class="dot"></div>
              <div class="dot"></div>
          </div>
      `;
      
      loadingElement.appendChild(avatar);
      loadingElement.appendChild(loadingContent);
      
      messagesWrapper.appendChild(loadingElement);
      
      // Fix scrolling
      fixScrolling();
  }

  // Function to remove loading indicator
  function removeLoadingIndicator() {
      const loadingElement = document.getElementById('loading-indicator');
      if (loadingElement) {
          loadingElement.remove();
      }
  }

  // Function to format message content (simple markdown-like formatting)
  function formatMessage(content) {
      if (!content) return '';
      
      // Escape HTML to prevent XSS
      let formatted = content
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
      
      // Basic markdown-like formatting
      
      // Code blocks with language
      formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)\n```/g, function(match, language, code) {
          return `<pre><code class="language-${language || 'plaintext'}">${code}</code></pre>`;
      });
      
      // Inline code
      formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
      
      // Bold
      formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      
      // Italic
      formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
      
      // Headers (h1, h2, h3)
      formatted = formatted.replace(/^### (.*$)/gm, '<h3>$1</h3>');
      formatted = formatted.replace(/^## (.*$)/gm, '<h2>$1</h2>');
      formatted = formatted.replace(/^# (.*$)/gm, '<h1>$1</h1>');
      
      // Convert line breaks to <br>
      formatted = formatted.replace(/\n/g, '<br>');
      
      return formatted;
  }

  // Function to show notification
  function showNotification(message, type = 'success') {
      // Remove any existing notifications
      const existingNotifications = document.querySelectorAll('.notification');
      existingNotifications.forEach(notification => {
          notification.remove();
      });
      
      // Create notification element
      const notification = document.createElement('div');
      notification.className = `notification ${type}`;
      notification.textContent = message;
      
      // Add to document
      document.body.appendChild(notification);
      
      // Remove after 3 seconds
      setTimeout(() => {
          notification.remove();
      }, 3000);
  }

  // Clean up when the window is closed
  window.addEventListener('beforeunload', () => {
      window.api.removeAllListeners();
  });
});
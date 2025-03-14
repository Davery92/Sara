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
    
    // Set the initial server URL in the input
    serverUrlInput.value = serverUrl;
    
    // Test connection on startup
    testConnection();

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

    // Function to send a message to the server
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
                                    
                                    // Add assistant message
                                    assistantMessageElement = addMessageToChat('assistant', assistantResponse, true);
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
            if (assistantResponse && ttsEnabled) {
                setTimeout(() => {
                  generateTTS(assistantResponse, assistantMessageElement);
                }, 200);
              }
            // Update chat history with the assistant's response if we received one
            if (assistantResponse) {
                chatHistory.push({ role: 'assistant', content: assistantResponse });
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
        
        if (isNew && role === 'assistant' && ttsEnabled) {
            // Generate TTS for new assistant messages if TTS is enabled
            setTimeout(() => {
              generateTTS(content, messageElement);
            }, 200);
          }

        // Return the message element (useful for streaming updates)
        return messageElement;
    }

    async function setupTTS() {
        try {
          // Get TTS preferences
          const prefs = await window.tts.getPreferences();
          
          // Set initial state
          ttsEnabled = prefs.enabled;
          ttsToggle.checked = ttsEnabled;
          
          if (ttsEnabled) {
            voiceSelectorContainer.style.display = 'block';
          }
          
          // Try to load available voices
          const voices = await window.tts.getVoices();
          if (voices && voices.length > 0) {
            // Clear existing options
            voiceSelector.innerHTML = '';
            
            // Add each voice as an option
            voices.forEach(voice => {
              const option = document.createElement('option');
              option.value = voice.id;
              option.textContent = voice.name;
              voiceSelector.appendChild(option);
            });
            
            // Set selected voice from preferences
            if (prefs.voice) {
              voiceSelector.value = prefs.voice;
            }
          }
          
          // Check TTS service status if enabled
          if (ttsEnabled) {
            const status = await window.tts.checkStatus();
            if (!status) {
              showNotification('TTS service is not available', 'error');
            }
          }
          
          // Toggle TTS when the switch is clicked
          ttsToggle.addEventListener('change', function() {
            ttsEnabled = this.checked;
            voiceSelectorContainer.style.display = ttsEnabled ? 'block' : 'none';
            
            // Save preference
            window.tts.savePreferences(ttsEnabled, voiceSelector.value, 1.0);
            
            if (ttsEnabled) {
              // Check TTS service status
              window.tts.checkStatus().then(status => {
                if (!status) {
                  showNotification('TTS service is not available', 'error');
                }
              });
            }
          });
          
          // Save voice preference when changed
          voiceSelector.addEventListener('change', function() {
            window.tts.savePreferences(ttsEnabled, this.value, 1.0);
          });
          
        } catch (error) {
          console.error('Error setting up TTS:', error);
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
      async function generateTTS(text, messageElement) {
        if (!ttsEnabled || !text) return;
        
        // Skip if this message already has TTS generating or generated
        if (messageElement.dataset.ttsStatus === 'generating' || 
            messageElement.dataset.ttsStatus === 'generated') {
          return;
        }
        
        // Mark this message as TTS generating
        messageElement.dataset.ttsStatus = 'generating';
        
        // Get selected voice
        const voice = voiceSelector.value;
        
        // Create audio controls if they don't exist
        let audioControls = messageElement.querySelector('.audio-controls');
        if (!audioControls) {
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
          
          // Add to message
          messageElement.querySelector('.message-content').appendChild(audioControls);
        }
        
        // Get play button and status
        const playButton = audioControls.querySelector('.play-button');
        const audioStatus = audioControls.querySelector('.audio-status');
        
        // Set loading state
        playButton.innerHTML = '<div class="audio-loading"></div>';
        playButton.disabled = true;
        audioStatus.textContent = 'Generating audio...';
        
        try {
          // Clean text for TTS
          const cleanedText = cleanTextForTTS(text);
          
          // Get audio file path from main process
          const audioPath = await window.tts.generateSpeech(cleanedText, voice, 1.0);
          
          // Mark as generated
          messageElement.dataset.ttsStatus = 'generated';
          messageElement.dataset.audioPath = audioPath;
          
          // Reset play button
          playButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
          `;
          playButton.disabled = false;
          audioStatus.textContent = 'Ready to play';
          
          // Create audio element
          const audio = new Audio();
          audio.src = 'file://' + audioPath;
          
          // Store audio in message element
          messageElement.audio = audio;
          
          // Add event listener to play button
          playButton.addEventListener('click', function() {
            // Stop any currently playing audio
            if (currentTTSAudio && currentTTSAudio !== audio) {
              currentTTSAudio.pause();
              currentTTSAudio.currentTime = 0;
              
              // Reset the play button of any other message that was playing
              document.querySelectorAll('.chat-message').forEach(msg => {
                if (msg !== messageElement && msg.querySelector('.play-button')) {
                  msg.querySelector('.play-button').innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                  `;
                  
                  if (msg.querySelector('.audio-status')) {
                    msg.querySelector('.audio-status').textContent = 'Ready to play';
                  }
                }
              });
            }
            
            // Set as current audio
            currentTTSAudio = audio;
            
            // Check if audio is paused or ended
            if (audio.paused || audio.ended) {
              audio.play();
              
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
          });
          
          // Add ended event listener
          audio.addEventListener('ended', function() {
            // Reset play button
            playButton.innerHTML = `
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
            `;
            audioStatus.textContent = 'Ready to play';
          });
          
          // Auto-play if enabled
          if (ttsEnabled) {
            // Stop any currently playing audio
            if (currentTTSAudio && currentTTSAudio !== audio) {
              currentTTSAudio.pause();
              currentTTSAudio.currentTime = 0;
            }
            
            // Set as current audio
            currentTTSAudio = audio;
            
            // Play the audio
            audio.play();
            
            // Update UI for playing state
            playButton.innerHTML = `
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="6" y="4" width="4" height="16"></rect>
                <rect x="14" y="4" width="4" height="16"></rect>
              </svg>
            `;
            audioStatus.textContent = 'Playing...';
          }
        } catch (error) {
          console.error('TTS error:', error);
          
          // Reset TTS status on error
          messageElement.dataset.ttsStatus = 'error';
          
          // Reset play button with error state
          playButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          `;
          playButton.disabled = true;
          audioStatus.textContent = 'Error generating audio';
        }
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
// preload.js
// This file securely exposes APIs to the renderer process
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', {
    // Server URL management
    getServerUrl: () => ipcRenderer.invoke('get-server-url'),
    setServerUrl: (url) => ipcRenderer.invoke('set-server-url', url),
    
    // Event listeners
    onNewChat: (callback) => ipcRenderer.on('new-chat', callback),
    onClearChat: (callback) => ipcRenderer.on('clear-chat', callback),
    onOpenSettings: (callback) => ipcRenderer.on('open-settings', callback),
    
    // Remove event listeners to prevent memory leaks
    removeAllListeners: () => {
      ipcRenderer.removeAllListeners('new-chat');
      ipcRenderer.removeAllListeners('clear-chat');
      ipcRenderer.removeAllListeners('open-settings');
    }
  },
  'tts', {
    // Get list of available voices
    getVoices: () => ipcRenderer.invoke('tts-get-voices'),
    
    // Generate speech from text
    generateSpeech: (text, voice, speed) => ipcRenderer.invoke('tts-generate-speech', text, voice, speed),
    
    // Check TTS service status
    checkStatus: () => ipcRenderer.invoke('tts-check-status'),
    
    // Save TTS preferences
    savePreferences: (enabled, voice, speed) => ipcRenderer.invoke('tts-save-preferences', enabled, voice, speed),
    
    // Get TTS preferences
    getPreferences: () => ipcRenderer.invoke('tts-get-preferences')
  }
);
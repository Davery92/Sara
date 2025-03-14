// preload.js
// This file securely exposes APIs to the renderer process
const { contextBridge, ipcRenderer } = require('electron');


function logIPC(action, ...args) {
  const safeArgs = args.map(arg => {
    if (typeof arg === 'string' && arg.length > 100) {
      return arg.substring(0, 100) + '... [truncated]';
    }
    return arg;
  });
  console.log(`IPC ${action}:`, ...safeArgs);
  return args[0]; // Return the first argument unchanged
}
// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', {
    // Server URL management
    getServerUrl: () => {
      console.log('IPC Invoke: getServerUrl');
      return ipcRenderer.invoke('get-server-url');
    },
    setServerUrl: (url) => {
      console.log('IPC Invoke: setServerUrl', url);
      return ipcRenderer.invoke('set-server-url', url);
    },
    
    // Event listeners
    onNewChat: (callback) => {
      console.log('IPC On: new-chat');
      return ipcRenderer.on('new-chat', callback);
    },
    onClearChat: (callback) => {
      console.log('IPC On: clear-chat');
      return ipcRenderer.on('clear-chat', callback);
    },
    onOpenSettings: (callback) => {
      console.log('IPC On: open-settings');
      return ipcRenderer.on('open-settings', callback);
    },
    
    // Remove event listeners to prevent memory leaks
    removeAllListeners: () => {
      console.log('IPC: Removing all listeners');
      ipcRenderer.removeAllListeners('new-chat');
      ipcRenderer.removeAllListeners('clear-chat');
      ipcRenderer.removeAllListeners('open-settings');
    }
  }
);

contextBridge.exposeInMainWorld(
  'tts', {
    // Get list of available voices
    getVoices: () => {
      console.log('IPC Invoke: tts-get-voices');
      return ipcRenderer.invoke('tts-get-voices');
    },
    
    // Generate speech from text
    generateSpeech: (text, voice, speed) => {
      console.log(`IPC Invoke: tts-generate-speech, voice=${voice}, speed=${speed}, text length=${text?.length || 0}`);
      return ipcRenderer.invoke('tts-generate-speech', text, voice, speed);
    },
    
    // Check TTS service status
    checkStatus: () => {
      console.log('IPC Invoke: tts-check-status');
      return ipcRenderer.invoke('tts-check-status');
    },
    
    // Save TTS preferences
    savePreferences: (enabled, voice, speed) => {
      console.log(`IPC Invoke: tts-save-preferences, enabled=${enabled}, voice=${voice}, speed=${speed}`);
      return ipcRenderer.invoke('tts-save-preferences', enabled, voice, speed);
    },
    
    // Get TTS preferences
    getPreferences: () => {
      console.log('IPC Invoke: tts-get-preferences');
      return ipcRenderer.invoke('tts-get-preferences');
    }
  }
);
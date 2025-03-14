// This is the main Electron process file that creates the app window
const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const path = require('path');
const Store = require('electron-store');
const http = require('http');
const https = require('https');
const fs = require('fs');

// Initialize settings store
const store = new Store({
  defaults: {
    serverUrl: 'http://10.185.1.8:7009', // Default to the local server port from your code
    windowBounds: { width: 1000, height: 800 }
  }
});

let mainWindow;

function createWindow() {
  // Get saved window dimensions
  const { width, height } = store.get('windowBounds');

  // Create the browser window
  mainWindow = new BrowserWindow({
    width,
    height,
    minWidth: 600,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: path.join(__dirname, 'icon.png'),
    backgroundColor: '#0f1117' // Match the app's background color
  });

  // Load the index.html of the app
  mainWindow.loadFile('index.html');

  // Save window size when resized
  mainWindow.on('resize', () => {
    store.set('windowBounds', mainWindow.getBounds());
  });

  // Create application menu
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Chat',
          accelerator: 'CmdOrCtrl+N',
          click: () => mainWindow.webContents.send('new-chat')
        },
        {
          label: 'Clear Chat',
          click: () => mainWindow.webContents.send('clear-chat')
        },
        { type: 'separator' },
        {
          label: 'Settings',
          click: () => mainWindow.webContents.send('open-settings')
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { type: 'separator' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About Sara Assistant',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              title: 'About Sara Assistant',
              message: 'Sara Assistant v1.0.0',
              detail: 'A desktop interface for the Sara AI Assistant API.',
              buttons: ['OK']
            });
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Quit when all windows are closed, except on macOS.
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// Handle IPC messages from the renderer
ipcMain.handle('get-server-url', () => {
  return store.get('serverUrl');
});

ipcMain.handle('set-server-url', (_, url) => {
  store.set('serverUrl', url);
  return true;
});
ipcMain.handle('tts-get-voices', async () => {
  try {
    console.log("Getting TTS voices from server");
    const serverUrl = store.get('serverUrl', 'http://10.185.1.8:7009');
    
    console.log(`Sending request to ${serverUrl}/v1/tts/voices`);
    const response = await fetch(`${serverUrl}/v1/tts/voices`);
    
    if (response.ok) {
      const data = await response.json();
      console.log("Successfully retrieved voices:", data);
      return data.voices || [];
    } else {
      console.error('Error fetching voices:', response.status, response.statusText);
      // Return fallback voices
      return [
        {"id": "af_bella", "name": "Bella (African)"},
        {"id": "en_jony", "name": "Jony (English)"},
        {"id": "en_rachel", "name": "Rachel (English)"},
        {"id": "en_emma", "name": "Emma (English)"},
        {"id": "en_antoni", "name": "Antoni (English)"}
      ];
    }
  } catch (error) {
    console.error('Error fetching voices:', error);
    // Return fallback voices
    return [
      {"id": "af_bella", "name": "Bella (African)"},
      {"id": "en_jony", "name": "Jony (English)"},
      {"id": "en_rachel", "name": "Rachel (English)"}
    ];
  }
});

ipcMain.handle('tts-generate-speech', async (event, text, voice, speed) => {
  try {
    console.log("Generating speech with:");
    console.log(`- Voice: af_bella (forced)`); // Always log af_bella
    console.log(`- Speed: ${speed}`);
    console.log(`- Text length: ${text?.length || 0}`);
    
    // Get server URL from settings
    const serverUrl = store.get('serverUrl', 'http://localhost:7009');
    console.log(`Server URL: ${serverUrl}`);
    
    // Define output path with unique timestamp
    const userDataPath = app.getPath('userData');
    const timestamp = Date.now();
    const tempAudioPath = path.join(userDataPath, `temp-audio-${timestamp}.mp3`);
    console.log(`Audio will be saved to: ${tempAudioPath}`);
    
    // Clean up old audio files
    cleanupOldAudioFiles(userDataPath);
    
    // Make sure the text isn't too long
    if (text.length > 3000) {
      text = text.substring(0, 3000) + "... (text truncated for speech)";
      console.log("Text truncated to 3000 characters");
    }
    
    // Create request to TTS endpoint
    console.log(`Sending request to ${serverUrl}/v1/tts/generate`);
    const response = await fetch(`${serverUrl}/v1/tts/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: text,
        voice: 'af_bella', // Always use af_bella
        speed: speed || 1.0
      })
    });
    
    if (!response.ok) {
      throw new Error(`TTS request failed with status ${response.status}: ${response.statusText}`);
    }
    
    console.log("TTS request successful, processing response");
    
    // Get the audio data as a buffer
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    
    // Log buffer details
    console.log(`Audio buffer size: ${buffer.length} bytes`);
    
    if (buffer.length === 0) {
      throw new Error("Received empty audio buffer from TTS service");
    }
    
    // Save to temp file
    console.log(`Writing audio buffer to ${tempAudioPath}`);
    fs.writeFileSync(tempAudioPath, buffer);
    console.log("Audio file saved successfully");
    
    return tempAudioPath;
  } catch (error) {
    console.error('Error generating speech:', error);
    throw error;
  }
});

function cleanupOldAudioFiles(userDataPath) {
  try {
    const audioFiles = fs.readdirSync(userDataPath)
      .filter(file => file.startsWith('temp-audio-') && file.endsWith('.mp3'))
      .map(file => ({
        name: file,
        path: path.join(userDataPath, file),
        time: parseInt(file.replace('temp-audio-', '').replace('.mp3', '')) || 0
      }))
      .sort((a, b) => b.time - a.time); // Sort newest first
    
    // Keep the 10 most recent files, delete the rest
    if (audioFiles.length > 10) {
      console.log(`Cleaning up ${audioFiles.length - 10} old audio files`);
      audioFiles.slice(10).forEach(file => {
        try {
          fs.unlinkSync(file.path);
          console.log(`Deleted old audio file: ${file.path}`);
        } catch (err) {
          console.error(`Failed to delete file ${file.path}:`, err);
        }
      });
    }
  } catch (error) {
    console.error('Error cleaning up audio files:', error);
  }
}

ipcMain.handle('tts-check-status', async () => {
  try {
    console.log("Checking TTS service status");
    const serverUrl = store.get('serverUrl', 'http://10.185.1.8:7009');
    
    console.log(`Sending request to ${serverUrl}/v1/tts/status`);
    const response = await fetch(`${serverUrl}/v1/tts/status`);
    
    if (response.ok) {
      const data = await response.json();
      console.log("TTS status:", data);
      return data.status === 'online';
    }
    
    console.log("Failed to get TTS status");
    return false;
  } catch (error) {
    console.error('Error checking TTS status:', error);
    return false;
  }
});

ipcMain.handle('tts-save-preferences', async (event, enabled, voice, speed) => {
  try {
    console.log("Saving TTS preferences:");
    console.log(`- Enabled: ${enabled}`);
    console.log(`- Speed: ${speed}`);
    
    // Always save af_bella as the voice
    store.set('tts.enabled', enabled);
    store.set('tts.voice', 'af_bella');
    store.set('tts.speed', speed || 1.0);
    
    console.log("TTS preferences saved successfully with voice fixed to af_bella");
    return true;
  } catch (error) {
    console.error('Error saving TTS preferences:', error);
    return false;
  }
});

ipcMain.handle('tts-get-preferences', async () => {
  try {
    console.log("Getting TTS preferences");
    const enabled = store.get('tts.enabled', false);
    const speed = store.get('tts.speed', 1.0);
    
    const preferences = {
      enabled: enabled,
      voice: 'af_bella', // Always return af_bella
      speed: speed
    };
    
    console.log("TTS preferences:", preferences);
    return preferences;
  } catch (error) {
    console.error('Error getting TTS preferences:', error);
    return {
      enabled: false,
      voice: 'af_bella',
      speed: 1.0
    };
  }
});
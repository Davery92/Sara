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
    serverUrl: 'http://localhost:7009', // Default to the local server port from your code
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
    const serverUrl = store.get('serverUrl', 'http://localhost:7009');
    const response = await fetch(`${serverUrl}/v1/tts/voices`);
    
    if (response.ok) {
      const data = await response.json();
      return data.voices || [];
    } else {
      console.error('Error fetching voices:', response.statusText);
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
    return [
      {"id": "af_bella", "name": "Bella (African)"},
      {"id": "en_jony", "name": "Jony (English)"},
      {"id": "en_rachel", "name": "Rachel (English)"}
    ];
  }
});

ipcMain.handle('tts-generate-speech', async (event, text, voice, speed) => {
  try {
    const serverUrl = store.get('serverUrl', 'http://localhost:7009');
    const userDataPath = app.getPath('userData');
    const tempAudioPath = path.join(userDataPath, 'temp-audio.mp3');
    
    // Make sure the text isn't too long
    if (text.length > 3000) {
      text = text.substring(0, 3000) + "... (text truncated for speech)";
    }
    
    // Create request to TTS endpoint
    const response = await fetch(`${serverUrl}/v1/tts/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: text,
        voice: voice || 'en_jony',
        speed: speed || 1.0
      })
    });
    
    if (!response.ok) {
      throw new Error(`TTS request failed with status ${response.status}`);
    }
    
    // Get the audio data as a buffer
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    
    // Save to temp file
    fs.writeFileSync(tempAudioPath, buffer);
    
    return tempAudioPath;
  } catch (error) {
    console.error('Error generating speech:', error);
    throw error;
  }
});

ipcMain.handle('tts-check-status', async () => {
  try {
    const serverUrl = store.get('serverUrl', 'http://localhost:7009');
    const response = await fetch(`${serverUrl}/v1/tts/status`);
    
    if (response.ok) {
      const data = await response.json();
      return data.status === 'online';
    }
    
    return false;
  } catch (error) {
    console.error('Error checking TTS status:', error);
    return false;
  }
});

ipcMain.handle('tts-save-preferences', async (event, enabled, voice, speed) => {
  try {
    store.set('tts.enabled', enabled);
    store.set('tts.voice', voice);
    store.set('tts.speed', speed);
    return true;
  } catch (error) {
    console.error('Error saving TTS preferences:', error);
    return false;
  }
});

ipcMain.handle('tts-get-preferences', async () => {
  try {
    return {
      enabled: store.get('tts.enabled', false),
      voice: store.get('tts.voice', 'en_jony'),
      speed: store.get('tts.speed', 1.0)
    };
  } catch (error) {
    console.error('Error getting TTS preferences:', error);
    return {
      enabled: false,
      voice: 'en_jony',
      speed: 1.0
    };
  }
});
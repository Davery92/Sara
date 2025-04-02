// websocketUtil.js - Create this as a new file in your utils directory

/**
 * Creates and manages a singleton WebSocket connection
 * This ensures only one connection is created per endpoint
 */
class WebSocketManager {
    constructor() {
      this.connections = {};
      this.connectionIds = {};
      this.reconnectTimeouts = {};
      this.messageHandlers = {};
      this.defaultReconnectDelay = 5000; // 5 seconds
    }
  
    /**
     * Get or create a WebSocket connection
     * @param {string} url - The WebSocket URL
     * @param {string} name - A unique name for this connection type
     * @param {Object} options - Configuration options
     * @returns {WebSocket} - The WebSocket instance
     */
    getConnection(url, name, options = {}) {
      const {
        autoReconnect = true,
        maxReconnectAttempts = 5,
        reconnectDelay = this.defaultReconnectDelay,
        debug = false
      } = options;
  
      // If we already have an active connection for this name, return it
      if (this.connections[name] && this.connections[name].readyState === WebSocket.OPEN) {
        if (debug) console.log(`Using existing WebSocket connection: ${name}`);
        return this.connections[name];
      }
  
      // If we have a connection that's connecting, return it
      if (this.connections[name] && this.connections[name].readyState === WebSocket.CONNECTING) {
        if (debug) console.log(`WebSocket connection is connecting: ${name}`);
        return this.connections[name];
      }
  
      // Clean up any existing connection
      this.closeConnection(name);
  
      // Create a new WebSocket connection
      if (debug) console.log(`Creating new WebSocket connection: ${name} -> ${url}`);
      const socket = new WebSocket(url);
      this.connections[name] = socket;
      
      // Store reconnect settings with this connection
      socket._wsSettings = {
        url,
        name,
        autoReconnect,
        maxReconnectAttempts,
        reconnectDelay,
        reconnectAttempts: 0,
        debug
      };
  
      // Set up event handlers
      socket.onopen = (event) => {
        if (debug) console.log(`WebSocket connected: ${name}`);
        socket._wsSettings.reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        
        // Call any registered open handlers
        if (this.messageHandlers[name] && this.messageHandlers[name].open) {
          this.messageHandlers[name].open(event);
        }
      };
  
      socket.onmessage = (event) => {
        if (debug) console.log(`WebSocket message received for ${name}:`, event.data);
        
        try {
          // Parse the message as JSON
          const data = JSON.parse(event.data);
          
          // Store connection ID if provided
          if (data.type === 'connection_established' && data.connection_id) {
            this.connectionIds[name] = data.connection_id;
            if (debug) console.log(`Stored connection ID for ${name}: ${data.connection_id}`);
          }
          
          // Handle pings with automatic pong responses
          if (data.type === 'ping') {
            socket.send(JSON.stringify({
              type: 'pong',
              timestamp: data.timestamp,
              connection_id: this.connectionIds[name] || 'unknown'
            }));
            
            if (debug) console.log(`Responded to ping for ${name}`);
            return;
          }
          
          // Call any registered message handlers
          if (this.messageHandlers[name] && this.messageHandlers[name].message) {
            this.messageHandlers[name].message(data);
          }
        } catch (error) {
          console.error(`Error handling WebSocket message for ${name}:`, error);
        }
      };
  
      socket.onerror = (error) => {
        console.error(`WebSocket error for ${name}:`, error);
        
        // Call any registered error handlers
        if (this.messageHandlers[name] && this.messageHandlers[name].error) {
          this.messageHandlers[name].error(error);
        }
      };
  
      socket.onclose = (event) => {
        if (debug) console.log(`WebSocket closed: ${name} (code: ${event.code}, reason: ${event.reason})`);
        
        // Call any registered close handlers
        if (this.messageHandlers[name] && this.messageHandlers[name].close) {
          this.messageHandlers[name].close(event);
        }
        
        // Handle reconnection if enabled
        if (autoReconnect && socket._wsSettings.reconnectAttempts < maxReconnectAttempts) {
          const settings = socket._wsSettings;
          settings.reconnectAttempts++;
          
          if (debug) {
            console.log(`Scheduling WebSocket reconnect for ${name} (attempt ${settings.reconnectAttempts}/${maxReconnectAttempts} in ${reconnectDelay}ms)`);
          }
          
          // Clear any existing timeout
          if (this.reconnectTimeouts[name]) {
            clearTimeout(this.reconnectTimeouts[name]);
          }
          
          // Set new timeout for reconnection
          this.reconnectTimeouts[name] = setTimeout(() => {
            if (debug) console.log(`Attempting to reconnect WebSocket: ${name}`);
            // Create a new connection with the same settings
            this.getConnection(url, name, options);
          }, reconnectDelay);
        } else if (autoReconnect && socket._wsSettings.reconnectAttempts >= maxReconnectAttempts) {
          console.warn(`WebSocket ${name} exceeded maximum reconnection attempts (${maxReconnectAttempts})`);
        }
        
        // Clean up the reference
        if (this.connections[name] === socket) {
          delete this.connections[name];
        }
      };
  
      return socket;
    }
  
    /**
     * Close a WebSocket connection and clean up references
     * @param {string} name - The name of the connection to close
     */
    closeConnection(name) {
      // Clear any reconnect timeout
      if (this.reconnectTimeouts[name]) {
        clearTimeout(this.reconnectTimeouts[name]);
        delete this.reconnectTimeouts[name];
      }
      
      // Close the connection if it exists
      if (this.connections[name]) {
        try {
          if (this.connections[name].readyState === WebSocket.OPEN || 
              this.connections[name].readyState === WebSocket.CONNECTING) {
            this.connections[name].close();
          }
        } catch (error) {
          console.error(`Error closing WebSocket ${name}:`, error);
        }
        
        // Clean up the reference
        delete this.connections[name];
        
        // Clean up the connection ID
        if (this.connectionIds[name]) {
          delete this.connectionIds[name];
        }
      }
    }
  
    /**
     * Register handlers for a specific connection
     * @param {string} name - The connection name
     * @param {Object} handlers - Object with message, error, open and close handlers
     */
    registerHandlers(name, handlers) {
      this.messageHandlers[name] = handlers;
    }
  
    /**
     * Close all connections and clean up
     */
    closeAll() {
      for (const name in this.connections) {
        this.closeConnection(name);
      }
      
      // Clear all handlers
      this.messageHandlers = {};
    }
  
    /**
     * Check if a connection is active
     * @param {string} name - The connection name
     * @returns {boolean} - Whether the connection is open
     */
    isConnected(name) {
      return !!(this.connections[name] && this.connections[name].readyState === WebSocket.OPEN);
    }
  
    /**
     * Send a message through a specific connection
     * @param {string} name - The connection name
     * @param {Object|string} data - The data to send
     * @returns {boolean} - Whether the message was sent
     */
    sendMessage(name, data) {
      if (!this.isConnected(name)) {
        console.warn(`Cannot send message: WebSocket ${name} is not connected`);
        return false;
      }
      
      try {
        const message = typeof data === 'string' ? data : JSON.stringify(data);
        this.connections[name].send(message);
        return true;
      } catch (error) {
        console.error(`Error sending message to WebSocket ${name}:`, error);
        return false;
      }
    }
  }
  
  // Create a singleton instance
  const webSocketManager = new WebSocketManager();
  
  export default webSocketManager;
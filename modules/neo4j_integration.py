"""
Integration module to replace Redis with Neo4j in the RAG system.
Maintains compatibility with existing code while switching the backend.
"""

import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


# Import the Neo4j client
from modules.neo4j_client import Neo4jClient

# Configure logging
# Configure logging
logger = logging.getLogger("redis-message-store")

# Neo4j connection details
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://10.185.1.8:7687")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "Nutman17!")

# Global client instance
neo4j_client = None

def get_neo4j_client():
    """Get or initialize the Neo4j client"""
    global neo4j_client
    if neo4j_client is None:
        try:
            neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)
            logger.info("Neo4j client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j client: {e}")
            raise
    return neo4j_client

def integrate_neo4j_with_server(app):
    """
    Integrate Neo4j with the server
    
    Args:
        app: The FastAPI application instance
    """
    try:
        # Initialize the Neo4j client
        client = get_neo4j_client()
        
        # Check connection
        if client.ping():
            logger.info("Neo4j connection successful")
        else:
            logger.warning("Failed to connect to Neo4j")
        
        # Add health check endpoint
        @app.get("/neo4j/health")
        async def neo4j_health():
            try:
                client = get_neo4j_client()
                status = "healthy" if client.ping() else "unhealthy"
                return {
                    "status": status,
                    "service": "Neo4j",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Neo4j health check failed: {e}")
                return {
                    "status": "error",
                    "service": "Neo4j",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        logger.info("Neo4j integration complete")
    except Exception as e:
        logger.error(f"Error integrating Neo4j with server: {e}")
        raise



class MessageStore:
    """
    Message store that uses Redis for all conversation storage.
    """
    
    def __init__(self):
        """
        Initialize the message store
        """
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Redis client"""
        try:
            # Import RedisClient here to avoid circular imports
            from modules.redis_client import RedisClient
            self.client = RedisClient(host='localhost', port=6379, db=0)
            logger.info("MessageStore using Redis backend")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    def store_message(self, role, content, conversation_id, date=None, embedding=None):
        """
        Store a message with Redis
        
        Args:
            role: The role of the message sender (user/assistant/system)
            content: The message content
            conversation_id: ID of the conversation
            date: Optional date (defaults to now)
            embedding: Optional vector embedding
            
        Returns:
            The message ID
        """
        if not date:
            date = datetime.now()
            
        return self.client.store_message(role, content, conversation_id, date, embedding)
    
    def get_messages_by_conversation(self, conversation_id, limit=100):
        """
        Get messages for a conversation with Redis
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        return self.client.get_messages_by_conversation(conversation_id, limit)
    
    def list_conversations(self, limit=20, offset=0):
        """
        List all conversations with implemented Redis interface
        
        Args:
            limit: Maximum number of conversations to return
            offset: Pagination offset
            
        Returns:
            List of conversation dictionaries
        """
        try:
            # Pattern to match all message keys
            pattern = "message:*"
            all_message_keys = self.client.redis_client.keys(pattern)
            
            # Extract unique conversation IDs
            conversations = {}
            for key in all_message_keys:
                # Skip embedding keys
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                if ":embedding" in key:
                    continue
                
                # Format: message:conversation_id:role:timestamp
                parts = key.split(':')
                if len(parts) >= 4:
                    conv_id = parts[1]
                    role = parts[2]
                    timestamp = parts[3]
                    
                    if conv_id not in conversations or int(timestamp) > int(conversations[conv_id]["timestamp"]):
                        # Get the message content
                        message_data = self.client.get_message_by_key(key)
                        if message_data:
                            content = message_data.get("content", "")
                            content_preview = (content[:100] + "...") if len(content) > 100 else content
                            
                            conversations[conv_id] = {
                                "id": conv_id,
                                "last_message": content_preview,
                                "timestamp": timestamp,
                                "last_role": role
                            }
            
            # Convert to list and sort by timestamp (newest first)
            conversation_list = list(conversations.values())
            conversation_list.sort(key=lambda x: int(x["timestamp"]), reverse=True)
            
            # Apply pagination
            paginated_results = conversation_list[offset:offset+limit]
            
            return paginated_results
        except Exception as e:
            logger.error(f"Error listing conversations from Redis: {e}")
            return []
    
    def update_conversation_title(self, conversation_id, title):
        """
        Store conversation title in Redis
        
        Args:
            conversation_id: ID of the conversation
            title: New title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the title as a separate key
            title_key = f"conversation:title:{conversation_id}"
            self.client.redis_client.set(title_key, title)
            return True
        except Exception as e:
            logger.error(f"Error updating conversation title: {e}")
            return False
    
    def delete_conversation(self, conversation_id):
        """
        Delete a conversation with Redis
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all keys for this conversation
            pattern = f"message:{conversation_id}:*"
            keys = self.client.redis_client.keys(pattern)
            
            # Also get title key
            title_key = f"conversation:title:{conversation_id}"
            
            if keys:
                # Delete all keys
                self.client.redis_client.delete(*keys)
                
            # Delete title key if it exists
            if self.client.redis_client.exists(title_key):
                self.client.redis_client.delete(title_key)
            
            logger.info(f"Deleted conversation {conversation_id} with {len(keys)} messages")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False
    
    def vector_search(self, embedding, k=3):
        """
        Find similar messages with Redis
        
        Args:
            embedding: Vector embedding
            k: Number of results
            
        Returns:
            List of similar messages
        """
        return self.client.vector_search(embedding, k)
    
    def ping(self):
        """Test connection to Redis"""
        return self.client.ping()

# Create a global instance for easy access
message_store = None

def get_message_store():
    """Get the global message store instance"""
    global message_store
    if message_store is None:
        message_store = MessageStore()
    return message_store
# Create a global instance for easy access
message_store = None

def test_message_store():
    """Test the message store implementation"""
    store = get_message_store(use_neo4j=True)
    
    print("Testing MessageStore...")
    if not store.ping():
        print("Database connection failed")
        return
    
    # Create a test conversation
    conversation_id = f"test-conv-{uuid.uuid4()}"
    print(f"Creating test conversation: {conversation_id}")
    
    # Store messages
    store.store_message("user", "Hello from the MessageStore!", conversation_id)
    store.store_message("assistant", "I'm responding through the MessageStore", conversation_id)
    
    # Retrieve messages
    print("Retrieving messages...")
    messages = store.get_messages_by_conversation(conversation_id)
    for i, msg in enumerate(messages):
        print(f"Message {i+1}: {msg['role']} - {msg['content']}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    # Set up basic logging for the test
    logging.basicConfig(level=logging.INFO)
    test_message_store()
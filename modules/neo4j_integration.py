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
logger = logging.getLogger("neo4j-integration")

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
    Message store that abstracts database access.
    Provides compatibility with both Redis and Neo4j backends.
    """
    
    def __init__(self, use_neo4j=True):
        """
        Initialize the message store
        
        Args:
            use_neo4j: Whether to use Neo4j (True) or Redis (False)
        """
        self.use_neo4j = use_neo4j
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the database client based on configuration"""
        if self.use_neo4j:
            try:
                self.client = get_neo4j_client()
                logger.info("MessageStore using Neo4j backend")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j for MessageStore: {e}")
                # Fall back to Redis if Neo4j fails
                self.use_neo4j = False
                self._initialize_redis()
        else:
            self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis client as fallback"""
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
        Store a message with unified interface for both backends
        
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
            
        if self.use_neo4j:
            return self.client.store_message(role, content, conversation_id, date, embedding)
        else:
            # Redis backend
            return self.client.store_message(role, content, conversation_id, date, embedding)
    
    def get_messages_by_conversation(self, conversation_id, limit=100):
        """
        Get messages for a conversation with unified interface
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        # Both backends have compatible interfaces for this method
        return self.client.get_messages_by_conversation(conversation_id, limit)
    
    def list_conversations(self, limit=20, offset=0):
        """
        List all conversations with unified interface
        
        Args:
            limit: Maximum number of conversations to return
            offset: Pagination offset
            
        Returns:
            List of conversation dictionaries
        """
        if self.use_neo4j:
            return self.client.list_conversations(limit, offset)
        else:
            # Redis backend - implement if needed
            # This might not be implemented in your current RedisClient
            if hasattr(self.client, 'list_conversations'):
                return self.client.list_conversations(limit, offset)
            else:
                logger.warning("list_conversations not implemented in Redis client")
                return []
    
    def update_conversation_title(self, conversation_id, title):
        """
        Update conversation title with unified interface
        
        Args:
            conversation_id: ID of the conversation
            title: New title
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_neo4j:
            return self.client.update_conversation_title(conversation_id, title)
        else:
            # Redis backend - implement if needed
            if hasattr(self.client, 'update_conversation_title'):
                return self.client.update_conversation_title(conversation_id, title)
            else:
                logger.warning("update_conversation_title not implemented in Redis client")
                return False
    
    def delete_conversation(self, conversation_id):
        """
        Delete a conversation with unified interface
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_neo4j:
            return self.client.delete_conversation(conversation_id)
        else:
            # Redis backend - implement if needed
            if hasattr(self.client, 'delete_conversation'):
                return self.client.delete_conversation(conversation_id)
            else:
                logger.warning("delete_conversation not implemented in Redis client")
                return False
    
    def vector_search(self, embedding, k=3):
        """
        Find similar messages with unified interface
        
        Args:
            embedding: Vector embedding
            k: Number of results
            
        Returns:
            List of similar messages
        """
        # Both backends have compatible interfaces for this method
        return self.client.vector_search(embedding, k)
    
    def ping(self):
        """Test connection to the database"""
        return self.client.ping()

# Create a global instance for easy access
message_store = None

def get_message_store(use_neo4j=True):
    """Get the global message store instance"""
    global message_store
    if message_store is None:
        message_store = MessageStore(use_neo4j=use_neo4j)
    return message_store

# Test function
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
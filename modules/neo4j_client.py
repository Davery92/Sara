"""
Neo4j client module for integration with the server.
Place this file in the modules directory as neo4j_client.py
"""
from neo4j import GraphDatabase
import logging
import datetime
import uuid
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger("neo4j-client")

class Neo4jClient:
    def __init__(self, uri, user, password, database):
        """Initialize the Neo4j database connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Create indices if they don't exist
        self._ensure_indices()
        
        logger.info("Neo4j client initialized")
    
    def _ensure_indices(self):
        """Create necessary indices for efficient queries"""
        try:
            with self.driver.session(database=self.database) as session:
                # Index on conversation_id for fast conversation retrieval
                session.run("CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.conversation_id)")
                # Index on message_id for fast lookups
                session.run("CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.id)")
                # Index on timestamp for chronological ordering
                session.run("CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.timestamp)")
                logger.info("Neo4j indices created or verified")
        except Exception as e:
            logger.error(f"Failed to create indices: {e}")
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def ping(self):
        """Test connection to Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS result")
                return result.single()["result"] == 1
        except Exception as e:
            logger.error(f"Neo4j ping failed: {e}")
            return False
    
    def store_message(self, role: str, content: str, conversation_id: str, date=None, embedding=None) -> str:
        """
        Store a message with its metadata and embedding
        
        Args:
            role: The role of the message sender (user/assistant/system)
            content: The message content
            conversation_id: ID of the conversation this message belongs to
            date: Optional datetime object (defaults to now)
            embedding: Optional vector embedding of the message content
            
        Returns:
            The ID of the created message
        """
        try:
            # Generate timestamp and ID
            timestamp = date.isoformat() if date else datetime.datetime.now().isoformat()
            message_id = f"message:{conversation_id}:{role}:{int(datetime.datetime.now().timestamp())}"
            
            # Convert embedding to list if it's a numpy array
            if embedding is not None and isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Create the base query without embedding
            query = """
            CREATE (m:Message {
                id: $id,
                role: $role,
                content: $content,
                conversation_id: $conversation_id,
                date: $date,
                timestamp: $timestamp
            })
            RETURN m.id as message_id
            """
            
            params = {
                "id": message_id,
                "role": role,
                "content": content,
                "conversation_id": conversation_id,
                "date": timestamp.split('T')[0] if timestamp else datetime.datetime.now().strftime("%Y-%m-%d"),
                "timestamp": timestamp
            }
            
            # If there's an embedding, add embedding relationship
            embedding_id = None
            if embedding:
                # Store embedding in a separate node
                embedding_id = f"{message_id}:embedding"
                
                # We'll add embedding handling code here that works with Neo4j
                # For now, just storing a placeholder
                params["has_embedding"] = True
                
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                result_id = result.single()["message_id"]
                
                # If we have an embedding, we can store it separately
                if embedding and embedding_id:
                    # For now, just store first few dimensions as a property
                    # In Neo4j 5.5+ we'd use vector indices
                    embedding_query = """
                    MATCH (m:Message {id: $message_id})
                    CREATE (e:Embedding {
                        id: $embedding_id,
                        dimensions: $dimensions,
                        sample: $sample
                    })
                    CREATE (m)-[:HAS_EMBEDDING]->(e)
                    """
                    session.run(
                        embedding_query,
                        message_id=message_id,
                        embedding_id=embedding_id,
                        dimensions=len(embedding),
                        sample=embedding[:10]  # Store first 10 dimensions as sample
                    )
                
                logger.debug(f"Message stored with ID: {result_id}")
                return result_id
                
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            return None
    
    def get_messages_by_conversation(self, conversation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation, ordered by timestamp
        
        Args:
            conversation_id: ID of the conversation to retrieve
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries with role and content
        """
        try:
            query = """
            MATCH (m:Message {conversation_id: $conversation_id})
            RETURN m.id as id, m.role as role, m.content as content, 
                   m.timestamp as timestamp
            ORDER BY m.timestamp ASC
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    conversation_id=conversation_id,
                    limit=limit
                )
                
                messages = []
                for record in result:
                    messages.append({
                        "role": record["role"],
                        "content": record["content"],
                        "timestamp": record["timestamp"]
                    })
                
                logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
                return messages
                
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return []
    
    def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all unique conversations with their last message
        
        Args:
            limit: Maximum number of conversations to return
            offset: Pagination offset
            
        Returns:
            List of conversation dictionaries with id and last message
        """
        try:
            # Query to get unique conversation IDs and their latest message
            query = """
            MATCH (m:Message)
            WITH m.conversation_id AS conv_id, max(m.timestamp) AS latest_time
            ORDER BY latest_time DESC
            SKIP $offset
            LIMIT $limit
            MATCH (latest:Message {conversation_id: conv_id, timestamp: latest_time})
            RETURN conv_id AS conversation_id, 
                   latest.content AS last_message,
                   latest.timestamp AS timestamp,
                   latest.role AS last_role
            ORDER BY latest.timestamp DESC
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    offset=offset,
                    limit=limit
                )
                
                conversations = []
                for record in result:
                    conversations.append({
                        "id": record["conversation_id"],
                        "last_message": record["last_message"][:100] + "..." if len(record["last_message"]) > 100 else record["last_message"],
                        "timestamp": record["timestamp"],
                        "last_role": record["last_role"]
                    })
                
                logger.debug(f"Retrieved {len(conversations)} conversations")
                return conversations
                
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Update the title of a conversation
        
        Args:
            conversation_id: ID of the conversation to update
            title: New title for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, check if a conversation title node exists
            query_check = """
            MATCH (c:ConversationMetadata {conversation_id: $conversation_id})
            RETURN c
            """
            
            create_query = """
            CREATE (c:ConversationMetadata {
                conversation_id: $conversation_id,
                title: $title,
                updated_at: $timestamp
            })
            """
            
            update_query = """
            MATCH (c:ConversationMetadata {conversation_id: $conversation_id})
            SET c.title = $title,
                c.updated_at = $timestamp
            """
            
            with self.driver.session(database=self.database) as session:
                # Check if metadata exists
                result = session.run(query_check, conversation_id=conversation_id)
                exists = result.single() is not None
                
                timestamp = datetime.datetime.now().isoformat()
                
                if exists:
                    # Update existing metadata
                    session.run(
                        update_query,
                        conversation_id=conversation_id,
                        title=title,
                        timestamp=timestamp
                    )
                else:
                    # Create new metadata
                    session.run(
                        create_query,
                        conversation_id=conversation_id,
                        title=title,
                        timestamp=timestamp
                    )
                
                logger.debug(f"Updated title for conversation {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating conversation title: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            MATCH (m:Message {conversation_id: $conversation_id})
            WITH count(m) as count
            MATCH (m:Message {conversation_id: $conversation_id})
            DETACH DELETE m
            RETURN count
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, conversation_id=conversation_id)
                count = result.single()["count"]
                
                # Also delete any metadata
                metadata_query = """
                MATCH (c:ConversationMetadata {conversation_id: $conversation_id})
                DETACH DELETE c
                """
                session.run(metadata_query, conversation_id=conversation_id)
                
                logger.info(f"Deleted conversation {conversation_id} with {count} messages")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False
    
    def vector_search(self, embedding, k=3):
        """
        Find similar messages based on vector similarity
        
        Args:
            embedding: The vector embedding to search with
            k: Number of results to return
            
        Returns:
            List of similar messages with their content and score
        """
        # This is a placeholder for Neo4j vector search
        # For a proper implementation, you would use Neo4j 5.5+ vector indices
        # or integrate with a specialized vector database
        
        logger.warning("Vector search not yet implemented in Neo4j client")
        return []
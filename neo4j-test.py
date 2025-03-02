#!/usr/bin/env python3
"""
Test script for Neo4j integration with RAG system.
Tests connection to Neo4j database and uploads a test message.
"""

from neo4j import GraphDatabase
import logging
import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neo4j-test")

# Neo4j connection details
NEO4J_URI = "bolt://10.185.1.8:7687"  # Using bolt protocol on standard port
NEO4J_DB = "neo4j"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Nutman17!"

class Neo4jClient:
    def __init__(self, uri, user, password, database):
        """Initialize the Neo4j database connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info("Neo4j client initialized")
        
    def close(self):
        """Close the database connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")
        
    def verify_connection(self):
        """Verify the connection to Neo4j is working"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (n) RETURN count(n) AS count")
                count = result.single()["count"]
                logger.info(f"Connection verified. Database has {count} nodes.")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def store_message(self, role, content, conversation_id, embeddings=None):
        """Store a message in Neo4j with optional embeddings"""
        try:
            timestamp = datetime.datetime.now().isoformat()
            message_id = str(uuid.uuid4())
            
            # Create Cypher query - different handling if embeddings are provided
            if embeddings:
                # Limit embedding size for the test
                embeddings_str = str(embeddings[:10]) + "..."  # Truncated for readability
                logger.info(f"Storing message with embeddings (truncated): {embeddings_str}")
                
                query = """
                CREATE (m:Message {
                    id: $id,
                    role: $role,
                    content: $content,
                    conversation_id: $conversation_id,
                    timestamp: $timestamp,
                    has_embedding: true
                })
                """
                # In a real implementation, you would store embeddings appropriately
                # This might be as a property on the node or in a specialized vector index
            else:
                query = """
                CREATE (m:Message {
                    id: $id,
                    role: $role,
                    content: $content,
                    conversation_id: $conversation_id,
                    timestamp: $timestamp,
                    has_embedding: false
                })
                """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    id=message_id,
                    role=role,
                    content=content,
                    conversation_id=conversation_id,
                    timestamp=timestamp
                )
                logger.info(f"Message stored with ID: {message_id}")
                return message_id
                
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            return None
    
    def get_conversation(self, conversation_id, limit=10):
        """Retrieve messages for a conversation, ordered by timestamp"""
        try:
            query = """
            MATCH (m:Message {conversation_id: $conversation_id})
            RETURN m.id, m.role, m.content, m.timestamp
            ORDER BY m.timestamp
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
                        "id": record["m.id"],
                        "role": record["m.role"],
                        "content": record["m.content"],
                        "timestamp": record["m.timestamp"]
                    })
                
                logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
                return messages
                
        except Exception as e:
            logger.error(f"Failed to retrieve conversation: {e}")
            return []

def main():
    """Test Neo4j connection and basic operations"""
    client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)
    
    try:
        # Test 1: Verify connection
        logger.info("Test 1: Verifying connection to Neo4j...")
        if not client.verify_connection():
            logger.error("Failed to connect to Neo4j. Exiting tests.")
            return
        
        # Test 2: Store a message without embeddings
        logger.info("Test 2: Storing a test message without embeddings...")
        test_conversation_id = f"test-conv-{uuid.uuid4()}"
        message_id = client.store_message(
            role="user",
            content="This is a test message for Neo4j integration",
            conversation_id=test_conversation_id
        )
        
        if not message_id:
            logger.error("Failed to store test message. Skipping remaining tests.")
            return
        
        # Test 3: Store a message with embeddings
        logger.info("Test 3: Storing a test message with embeddings...")
        fake_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]  # Simplified for testing
        message_id_with_embedding = client.store_message(
            role="assistant",
            content="This is a response with an embedding vector",
            conversation_id=test_conversation_id,
            embeddings=fake_embeddings
        )
        
        # Test 4: Retrieve conversation
        logger.info("Test 4: Retrieving test conversation...")
        messages = client.get_conversation(test_conversation_id)
        
        if messages:
            logger.info("Successfully retrieved conversation:")
            for i, msg in enumerate(messages):
                logger.info(f"  Message {i+1}: {msg['role']} - {msg['content'][:30]}...")
        else:
            logger.error("Failed to retrieve conversation")
        
        logger.info("All tests completed.")
    finally:
        client.close()

if __name__ == "__main__":
    main()
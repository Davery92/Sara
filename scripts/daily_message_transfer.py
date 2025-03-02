#!/usr/bin/env python3
"""
Daily Message Transfer from Redis to Neo4j

This script:
1. Extracts all conversations from Redis
2. Processes them as documents (one per conversation)
3. Stores them in Neo4j using the existing document processing pipeline
4. Clears Redis of the processed messages
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import argparse
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import asyncio

# Add the modules directory to sys.path
sys.path.append('/home/david/Sara/modules')

# Import the Neo4j RAG manager and Redis client
from neo4j_rag_integration import Neo4jRAGManager, get_neo4j_rag_manager
from redis_client import RedisClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/david/Sara/logs/daily_message_transfer.log")
    ]
)

logger = logging.getLogger("daily-message-transfer")

# Constants
TEMP_DIR = "/tmp/daily_conversations"
os.makedirs(TEMP_DIR, exist_ok=True)


class DailyMessageTransfer:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the transfer system"""
        self.redis_client = RedisClient(host=redis_host, port=redis_port, db=redis_db)
        self.neo4j_rag = get_neo4j_rag_manager()
        self.yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Initializing daily transfer for conversations from {self.yesterday}")

    def get_conversation_ids(self):
        """Get all unique conversation IDs from Redis"""
        try:
            # Pattern to match all message keys
            pattern = "message:*"
            all_message_keys = self.redis_client.redis_client.keys(pattern)
            
            # Extract unique conversation IDs
            conversation_ids = set()
            for key in all_message_keys:
                # Skip embedding keys
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                if ":embedding" in key:
                    continue
                
                # Format: message:conversation_id:role:timestamp
                parts = key.split(':')
                if len(parts) >= 3:
                    conversation_ids.add(parts[1])
            
            logger.info(f"Found {len(conversation_ids)} unique conversations in Redis")
            return list(conversation_ids)
        except Exception as e:
            logger.error(f"Error retrieving conversation IDs: {e}")
            return []

    def get_conversation_messages(self, conversation_id):
        """Get all messages for a specific conversation"""
        try:
            pattern = f"message:{conversation_id}:*"
            message_keys = self.redis_client.redis_client.keys(pattern)
            
            # Skip embedding keys
            filtered_keys = []
            for key in message_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                if not key.endswith(":embedding"):
                    filtered_keys.append(key)
            
            # Get all messages
            messages = []
            for key in filtered_keys:
                message_data = self.redis_client.get_message_by_key(key)
                if message_data:
                    messages.append(message_data)
            
            # Sort by timestamp
            messages.sort(key=lambda x: int(x.get('timestamp', 0)))
            
            return messages
        except Exception as e:
            logger.error(f"Error retrieving messages for conversation {conversation_id}: {e}")
            return []

    def format_conversation_as_document(self, conversation_id, messages):
        """Format a conversation as a document for Neo4j storage"""
        try:
            # Skip empty conversations
            if not messages:
                logger.warning(f"Conversation {conversation_id} has no messages, skipping")
                return None
            
            # Get conversation date from first message
            first_message = messages[0]
            conv_date = first_message.get('date', self.yesterday)
            
            # Format the conversation
            conversation_text = ""
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                
                # Try to convert timestamp to readable time
                try:
                    time_str = datetime.fromtimestamp(int(timestamp)).strftime("%H:%M:%S")
                except:
                    time_str = timestamp
                
                conversation_text += f"[{time_str}] {role}: {content}\n\n"
            
            # Create document title
            title = f"Conversation {conversation_id} - {conv_date}"
            
            # Create temporary file
            file_path = os.path.join(TEMP_DIR, f"{conversation_id}_{conv_date}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(conversation_text)
            
            return {
                "file_path": file_path,
                "title": title,
                "conversation_id": conversation_id,
                "date": conv_date,
                "message_count": len(messages)
            }
        except Exception as e:
            logger.error(f"Error formatting conversation {conversation_id}: {e}")
            return None

    async def process_conversation_as_document(self, conversation_data):
        """Process a conversation as a document and store in Neo4j"""
        if not conversation_data:
            return None
        
        try:
            file_path = conversation_data["file_path"]
            title = conversation_data["title"]
            
            # Process the document using the Neo4j RAG Manager
            result = await self.neo4j_rag.process_document(
                file_path=file_path,
                filename=os.path.basename(file_path),
                content_type="text/plain",
                title=title,
                tags=["conversation", conversation_data["conversation_id"], conversation_data["date"]]
            )
            
            # Clean up temp file
            try:
                os.remove(file_path)
            except:
                pass
            
            return result
        except Exception as e:
            logger.error(f"Error processing conversation as document: {e}")
            return {"error": str(e)}

    def delete_conversation_from_redis(self, conversation_id):
        """Delete all messages for a conversation from Redis"""
        try:
            # Get all keys for this conversation
            pattern = f"message:{conversation_id}:*"
            keys = self.redis_client.redis_client.keys(pattern)
            
            if keys:
                # Delete all keys
                self.redis_client.redis_client.delete(*keys)
                logger.info(f"Deleted {len(keys)} keys for conversation {conversation_id}")
            
            return len(keys)
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return 0

    async def process_all_conversations(self):
        """Process all conversations from Redis and transfer to Neo4j"""
        # Get all conversation IDs
        conversation_ids = self.get_conversation_ids()
        logger.info(f"Starting to process {len(conversation_ids)} conversations")
        
        processed_count = 0
        failed_count = 0
        
        for conversation_id in conversation_ids:
            try:
                # Get all messages for this conversation
                messages = self.get_conversation_messages(conversation_id)
                logger.info(f"Processing conversation {conversation_id} with {len(messages)} messages")
                
                # Format as document
                conversation_data = self.format_conversation_as_document(conversation_id, messages)
                if not conversation_data:
                    logger.warning(f"Failed to format conversation {conversation_id}, skipping")
                    failed_count += 1
                    continue
                
                # Process and store in Neo4j
                result = await self.process_conversation_as_document(conversation_data)
                
                if result and "error" not in result:
                    # Delete from Redis if successfully transferred
                    deleted_count = self.delete_conversation_from_redis(conversation_id)
                    processed_count += 1
                    logger.info(f"Successfully processed and transferred conversation {conversation_id} ({deleted_count} keys deleted)")
                else:
                    logger.error(f"Failed to transfer conversation {conversation_id} to Neo4j")
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error processing conversation {conversation_id}: {e}")
                failed_count += 1
        
        return {
            "total": len(conversation_ids),
            "processed": processed_count,
            "failed": failed_count
        }


async def main():
    parser = argparse.ArgumentParser(description='Transfer Redis messages to Neo4j documents')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis database')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (don\'t delete from Redis)')
    args = parser.parse_args()
    
    logger.info("Starting daily message transfer from Redis to Neo4j")
    
    transfer = DailyMessageTransfer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db
    )
    
    results = await transfer.process_all_conversations()
    
    logger.info(f"Transfer completed: {results['processed']} processed, {results['failed']} failed out of {results['total']} total conversations")

if __name__ == "__main__":
    asyncio.run(main())
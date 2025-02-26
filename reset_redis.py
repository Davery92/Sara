#!/usr/bin/env python3
"""
Complete Redis Reset Script for RAG Module

This script completely resets Redis by removing all keys and indices.
Run this before implementing the new key structure.

Usage:
  python complete_redis_reset.py
"""
import redis
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("redis-reset")

def reset_redis_completely(host='localhost', port=6379, db=0):
    """Completely reset Redis by flushing all keys in the database"""
    logger.info(f"Connecting to Redis at {host}:{port}, db={db}")
    
    try:
        # Connect to Redis
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        
        # Check connection
        if not r.ping():
            logger.error("Failed to connect to Redis")
            return False
            
        logger.info("Connected to Redis successfully")
        
        # 1. Drop indices if they exist
        for index_name in ['docs_idx', 'chunks_idx', 'message_idx']:
            try:
                r.execute_command('FT.DROPINDEX', index_name)
                logger.info(f"Dropped index {index_name}")
            except redis.exceptions.ResponseError as e:
                if "Unknown index name" in str(e):
                    logger.info(f"{index_name} doesn't exist, no need to drop")
                else:
                    logger.warning(f"Error dropping {index_name}: {e}")
        
        # 2. Count keys before deletion
        doc_keys = len(r.keys('doc:*'))
        chunk_keys = len(r.keys('chunk:*'))
        message_keys = len(r.keys('message:*'))
        
        logger.info(f"Found {doc_keys} document keys, {chunk_keys} chunk keys, {message_keys} message keys")
        
        # 3. Flush the database to remove all keys
        r.flushdb()
        logger.info("Flushed all keys from the database")
        
        # 4. Verify keys are gone
        remaining_keys = r.keys('*')
        if remaining_keys:
            logger.warning(f"There are still {len(remaining_keys)} keys in the database")
            logger.warning(f"First few keys: {remaining_keys[:5]}")
        else:
            logger.info("Database is empty - reset successful")
        
        logger.info("Redis completely reset")
        return True
        
    except Exception as e:
        logger.error(f"Error resetting Redis: {e}")
        return False

if __name__ == "__main__":
    print("WARNING: This will DELETE ALL KEYS in your Redis database!")
    print("Make sure you're using the correct database number.")
    confirmation = input("Type 'YES' to confirm: ")
    
    if confirmation == "YES":
        if reset_redis_completely():
            logger.info("Redis has been completely reset. Restart your server to create new indices.")
        else:
            logger.error("Failed to reset Redis. Check the errors above.")
    else:
        print("Reset cancelled.")
#!/usr/bin/env python3
"""
Script to remove document-related keys from Redis
Usage: python redis_document_cleaner.py [--dry-run] [--confirm]
"""

import redis
import argparse
import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis-document-cleaner")

# Document storage directory - change this to match your configuration
DOCUMENTS_DIRECTORY = "/home/david/Sara/documents"

def connect_to_redis(host='localhost', port=6379, db=0):
    """Connect to Redis and return client"""
    try:
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True  # For key operations
        )
        client.ping()  # Test connection
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None

def remove_document_keys(client, dry_run=False, confirm=False):
    """Remove all document-related keys from Redis"""
    if not client:
        return False
    
    # Document-related key patterns
    key_patterns = [
        "doc:*",         # Document metadata
        "chunk:*",       # Chunk metadata and content
        "chunk:json:*",  # Chunk JSON data
        "chunk:vector:*" # Chunk vector data
    ]
    
    # Count keys to be deleted
    key_counts = {}
    total_keys = 0
    
    for pattern in key_patterns:
        keys = client.keys(pattern)
        key_counts[pattern] = len(keys)
        total_keys += len(keys)
    
    # Show key counts
    logger.info(f"Found {total_keys} document-related keys in Redis:")
    for pattern, count in key_counts.items():
        logger.info(f"  {pattern}: {count} keys")
    
    if total_keys == 0:
        logger.info("No document-related keys found. Nothing to delete.")
        return True
    
    # Check if we should proceed
    if not confirm:
        user_confirm = input(f"Delete {total_keys} document-related keys? (yes/no): ").lower().strip()
        if user_confirm != "yes":
            logger.info("Operation cancelled by user.")
            return False
    
    # Delete keys
    if dry_run:
        logger.info("DRY RUN: Would delete the following keys:")
        for pattern in key_patterns:
            keys = client.keys(pattern)
            if keys:
                logger.info(f"  Pattern {pattern}: {len(keys)} keys")
                # Show sample keys (up to 5)
                for key in list(keys)[:5]:
                    logger.info(f"    - {key}")
        return True
    
    # Actually delete the keys
    deleted_count = 0
    for pattern in key_patterns:
        keys = client.keys(pattern)
        if keys:
            deleted = client.delete(*keys)
            deleted_count += deleted
            logger.info(f"Deleted {deleted} keys matching pattern {pattern}")
    
    logger.info(f"Successfully deleted {deleted_count} document-related keys from Redis")
    
    # Also clean up document directory if it exists
    if os.path.exists(DOCUMENTS_DIRECTORY):
        if not confirm:
            dir_confirm = input(f"Also delete document files in {DOCUMENTS_DIRECTORY}? (yes/no): ").lower().strip()
        else:
            dir_confirm = "yes"
            
        if dir_confirm == "yes":
            if dry_run:
                logger.info(f"DRY RUN: Would delete document directory contents: {DOCUMENTS_DIRECTORY}")
            else:
                # Delete all files in the directory but keep the directory itself
                for item in os.listdir(DOCUMENTS_DIRECTORY):
                    item_path = os.path.join(DOCUMENTS_DIRECTORY, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        logger.info(f"Deleted directory: {item_path}")
                    else:
                        os.remove(item_path)
                        logger.info(f"Deleted file: {item_path}")
                logger.info(f"Document directory cleared: {DOCUMENTS_DIRECTORY}")
        else:
            logger.info("Document files not deleted.")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Remove document-related keys from Redis")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--host", default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument("--db", type=int, default=0, help="Redis database (default: 0)")
    
    args = parser.parse_args()
    
    logger.info("Redis Document Cleaner")
    logger.info(f"Connecting to Redis at {args.host}:{args.port}, db={args.db}")
    
    client = connect_to_redis(host=args.host, port=args.port, db=args.db)
    if not client:
        logger.error("Failed to connect to Redis. Exiting.")
        return 1
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no actual deletions will be performed")
    
    if args.confirm:
        logger.info("Running with auto-confirmation")
    
    success = remove_document_keys(client, dry_run=args.dry_run, confirm=args.confirm)
    
    if success:
        logger.info("Operation completed successfully")
        return 0
    else:
        logger.error("Operation failed")
        return 1

if __name__ == "__main__":
    exit(main())
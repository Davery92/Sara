#!/usr/bin/env python3
# redis_client.py
import redis
import json
import numpy as np
from datetime import datetime
import logging

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis client with JSON and Vector search capabilities"""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        # Separate client for binary data
        self.redis_binary = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False
        )
        
    def ping(self):
        """Test connection to Redis"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False
            
    def store_message(self, role, content, conversation_id, date, embedding=None):
        """Store a message with its metadata and embedding"""
        try:
            # Generate a unique message ID
            timestamp = int(date.timestamp())
            message_id = f"message:{conversation_id}:{role}:{timestamp}"
            
            # Prepare message data
            message_data = {
                "role": role,
                "content": content,
                "conversation_id": conversation_id,
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": timestamp
            }
            
            # Store the message
            self.redis_client.hset(message_id, mapping=message_data)
            
            # If embedding is provided, store it
            if embedding is not None:
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                self.redis_binary.hset(message_id, "embedding", embedding_bytes)
            
            return message_id
            
        except Exception as e:
            print(f"Error storing message: {e}")
            raise
            
    def get_messages_by_conversation(self, conversation_id, limit=100):
        """Get messages for a conversation (simple implementation)"""
        try:
            # Use pattern matching to find all messages for this conversation
            pattern = f"message:{conversation_id}:*"
            message_keys = self.redis_client.keys(pattern)
            
            messages = []
            for key in message_keys[:limit]:
                message_data = self.redis_client.hgetall(key)
                if message_data:
                    messages.append({
                        "role": message_data.get(b"role", b"").decode('utf-8'),
                        "content": message_data.get(b"content", b"").decode('utf-8')
                    })
                    
            return messages
            
        except Exception as e:
            print(f"Error retrieving messages: {e}")
            return []
            
    def vector_search(self, embedding, k=3):
        """Simple implementation that returns empty list for now"""
        # This is a placeholder until we implement vector search properly
        print("Vector search not fully implemented yet")
        return []
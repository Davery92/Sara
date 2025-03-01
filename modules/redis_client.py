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
            
            # Prepare message data (ensure all values are strings)
            message_data = {
                "role": str(role),
                "content": str(content),
                "conversation_id": str(conversation_id),
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": str(timestamp)
            }
            
            # Store the message
            self.redis_client.hset(message_id, mapping=message_data)
            
            # If embedding is provided, store it separately
            if embedding is not None:
                embedding_key = f"{message_id}:embedding"
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                self.redis_binary.set(embedding_key, embedding_bytes)
            
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
            
            # Skip embedding keys
            filtered_keys = []
            for key in message_keys:
                # Skip embedding-related keys
                if isinstance(key, bytes):
                    decoded_key = key.decode('utf-8', errors='ignore')
                    if not decoded_key.endswith(":embedding"):
                        filtered_keys.append(key)
                elif not key.endswith(":embedding"):
                    filtered_keys.append(key)
            
            messages = []
            for key in filtered_keys[:limit]:
                try:
                    message_data = self.redis_client.hgetall(key)
                    if message_data:
                        # Process each field, handling binary data
                        processed_data = {}
                        for field, value in message_data.items():
                            # Convert bytes to strings safely
                            if isinstance(field, bytes):
                                field = field.decode('utf-8', errors='replace')
                            
                            if isinstance(value, bytes):
                                # Skip binary embedding data
                                if field == "embedding":
                                    continue
                                try:
                                    value = value.decode('utf-8', errors='replace')
                                except:
                                    value = "(binary data)"
                            
                            processed_data[field] = value
                        
                        # Extract the role and content
                        messages.append({
                            "role": processed_data.get("role", "unknown"),
                            "content": processed_data.get("content", "")
                        })
                except Exception as e:
                    print(f"Error processing message key {key}: {e}")
                    continue
                        
            return messages
                
        except Exception as e:
            print(f"Error retrieving messages: {e}")
            return []

    def vector_search(self, embedding, k=3):
        """Find similar messages based on vector similarity"""
        try:
            # Since we haven't implemented actual vector search yet,
            # let's just return an empty list for now
            return []  # Placeholder until proper implementation
        except Exception as e:
            return []

    def get_message_by_key(self, key):
        """Retrieve message by key, handling binary data properly"""
        try:
            # Decode the key if it's bytes
            if isinstance(key, bytes):
                key = key.decode('utf-8', errors='replace')
                
            # Get message data
            message_data = self.redis_client.hgetall(key)
            
            if not message_data:
                return None
                
            # Process each field, handling binary data
            processed_data = {}
            for field, value in message_data.items():
                # Convert bytes to strings safely
                if isinstance(field, bytes):
                    field = field.decode('utf-8', errors='replace')
                
                if isinstance(value, bytes):
                    # Skip binary embedding data
                    if field == "embedding":
                        continue
                    try:
                        value = value.decode('utf-8', errors='replace')
                    except:
                        value = "(binary data)"
                
                processed_data[field] = value
            
            return processed_data
                
        except Exception as e:
            print(f"Error retrieving message by key {key}: {e}")
            return None
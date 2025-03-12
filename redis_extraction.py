import json
import re
import os
from datetime import datetime, timedelta
import argparse
import redis
import time
from typing import List, Dict, Any, Optional
import ollama

class RedisMessageExtractor:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True  # For general commands
        )
        # Separate client for binary data (vectors)
        self.redis_binary = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False  # For binary data
        )
        
    def ping(self):
        """Test the Redis connection."""
        try:
            return self.redis_client.ping()
        except Exception as e:
            print(f"Redis ping failed: {e}")
            return False
    
    def get_conversations_by_timestamp(self, start_timestamp, end_timestamp):
        """
        Get all conversation IDs with messages from a specific timestamp range.
        
        Args:
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (exclusive)
        
        Returns:
            List of conversation IDs that have messages in the timestamp range
        """
        try:
            # Get all message keys
            all_message_keys = self.redis_client.keys("message:*")
            print(f"Total message keys: {len(all_message_keys)}")
            
            # Track unique conversation IDs
            conversation_ids = set()
            
            # Process each key
            for key in all_message_keys:
                # Skip embedding keys
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                if key.endswith(":embedding"):
                    continue
                
                # Try to extract the timestamp from the key
                # Key format: message:conversation_id:role:timestamp
                key_parts = key.split(":")
                if len(key_parts) >= 4:
                    try:
                        timestamp = int(key_parts[3])
                        if start_timestamp <= timestamp < end_timestamp:
                            # This message is within our date range
                            conversation_ids.add(key_parts[1])
                    except (ValueError, IndexError):
                        # Skip if timestamp conversion fails
                        continue
            
            print(f"Found {len(conversation_ids)} conversations with messages in the timestamp range")
            return list(conversation_ids)
        except Exception as e:
            print(f"Error getting conversations by timestamp: {e}")
            return []
    
    def get_messages_for_timestamp_range(self, start_timestamp, end_timestamp):
        """
        Get all messages within a specific timestamp range.
        
        Args:
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (exclusive)
        
        Returns:
            Dictionary mapping conversation IDs to lists of messages
        """
        try:
            # Get all conversation IDs for the timestamp range
            conversation_ids = self.get_conversations_by_timestamp(start_timestamp, end_timestamp)
            
            # Dictionary to store messages by conversation ID
            conversations = {}
            
            # For each conversation ID, get messages within the timestamp range
            for conv_id in conversation_ids:
                # Use pattern matching to find all messages for this conversation
                pattern = f"message:{conv_id}:*"
                message_keys = self.redis_client.keys(pattern)
                
                # Skip embedding keys and filter by timestamp
                filtered_keys = []
                for key in message_keys:
                    if isinstance(key, bytes):
                        key = key.decode('utf-8', errors='ignore')
                    
                    if key.endswith(":embedding"):
                        continue
                    
                    # Extract timestamp from key
                    key_parts = key.split(":")
                    if len(key_parts) >= 4:
                        try:
                            timestamp = int(key_parts[3])
                            if start_timestamp <= timestamp < end_timestamp:
                                filtered_keys.append(key)
                        except (ValueError, IndexError):
                            continue
                
                # Skip if no messages in range
                if not filtered_keys:
                    continue
                
                # Process each key
                messages = []
                for key in filtered_keys:
                    try:
                        # Check what type the key is
                        key_type = self.redis_client.type(key)
                        
                        # Handle different Redis data types
                        message_data = None
                        if key_type == "ReJSON-RL" or key_type == "json":
                            # It's a JSON object
                            message_data = self.redis_client.json().get(key)
                        elif key_type == "hash":
                            # It's a hash - get all fields
                            message_data = self.redis_client.hgetall(key)
                        elif key_type == "string":
                            # It's a string - try to parse as JSON
                            try:
                                raw_data = self.redis_client.get(key)
                                if isinstance(raw_data, bytes):
                                    raw_data = raw_data.decode('utf-8', errors='ignore')
                                message_data = json.loads(raw_data)
                            except (json.JSONDecodeError, TypeError):
                                # If it's not JSON, use the raw string
                                raw_data = self.redis_client.get(key)
                                if isinstance(raw_data, bytes):
                                    raw_data = raw_data.decode('utf-8', errors='ignore')
                                message_data = {"content": raw_data}
                        else:
                            print(f"Unsupported Redis data type for key {key}: {key_type}")
                            continue
                        
                        # Extract message information
                        if message_data:
                            # Get key parts for role and timestamp
                            key_parts = key.split(":")
                            
                            # Create message object
                            message_obj = {
                                "role": message_data.get("role", "unknown") if isinstance(message_data, dict) else "unknown",
                                "content": message_data.get("content", "") if isinstance(message_data, dict) else str(message_data),
                                "timestamp": message_data.get("timestamp", 0) if isinstance(message_data, dict) else 0,
                                "conversation_id": conv_id
                            }
                            
                            # If role or timestamp not in data, extract from key
                            if message_obj["role"] == "unknown" and len(key_parts) >= 3:
                                message_obj["role"] = key_parts[2]
                            
                            if message_obj["timestamp"] == 0 and len(key_parts) >= 4:
                                try:
                                    message_obj["timestamp"] = int(key_parts[3])
                                except (ValueError, IndexError):
                                    pass
                            
                            messages.append(message_obj)
                    except Exception as e:
                        print(f"Error processing key {key}: {e}")
                        continue
                
                # Sort messages by timestamp
                messages.sort(key=lambda x: x.get("timestamp", 0))
                
                # Add to conversations dictionary
                if messages:
                    conversations[conv_id] = messages
            
            return conversations
        except Exception as e:
            print(f"Error getting messages for timestamp range: {e}")
            return {}

def convert_to_chat_format(messages):
    """
    Convert Redis message structure to chat log format.
    
    Args:
        messages: List of message objects from Redis
        
    Returns:
        String containing formatted chat log
    """
    chat_log = []
    
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        
        chat_log.append(f"{role}:\n{content}")
    
    return "\n\n".join(chat_log)

def analyze_with_ollama(chat_log, model="qwen2.5:32b"):
    """
    Use Ollama to analyze a chat log and identify contexts.
    
    Args:
        chat_log: Formatted chat log string
        model: Ollama model to use
        
    Returns:
        Dict containing the analysis results
    """
    prompt = f"""
    Analyze this conversation and identify the main context type.
    
    CONVERSATION:
    {chat_log}
    
    Identify a single word or short phrase as the context_type (e.g., diet, fitness, greeting, commute, worklife).
    Then write a brief summary of what the user is discussing.
    
    Create a JSON object with exactly these fields:
    - context_type: A single word or short phrase for the category
    - data.content: A brief summary of what the user is asking about or discussing
    - data.confidence_score: A number between 0.5-0.99 representing confidence
    - data.last_updated: Today's date (YYYY-MM-DD)
    
    Return ONLY valid JSON with NOTHING else.
    """
    
    try:
        # Call Ollama API
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        json_text = response['message']['content'].strip()
        
        # Clean up any markdown formatting
        json_text = re.sub(r'^```json', '', json_text)
        json_text = re.sub(r'^```', '', json_text)
        json_text = re.sub(r'```$', '', json_text)
        json_text = json_text.strip()
        
        # Parse the JSON
        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response: {e}")
            print(f"Response text: {json_text}")
            # Fall back to basic analysis
            return basic_context_analysis(chat_log)
            
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return basic_context_analysis(chat_log)

def basic_context_analysis(chat_log):
    """Basic rule-based context analysis when Ollama is unavailable."""
    chat_log_lower = chat_log.lower()
    
    # Simple keyword matching
    keywords = {
        "diet": ["diet", "nutrition", "food", "eat", "meal", "carnivore", "protein", "carb", "fat"],
        "fitness": ["workout", "exercise", "training", "strength", "gym", "muscle", "weight"],
        "greeting": ["hello", "hi ", "hey", "morning", "afternoon", "evening"],
        "commute": ["drive", "commute", "traffic", "road", "car", "bus", "train"],
        "routine": ["routine", "schedule", "agenda", "plan", "time", "clock", "day"],
        "work": ["work", "job", "office", "career", "meeting", "email", "boss", "colleague"],
        "pets": ["dog", "cat", "pet", "animal", "walk"],
    }
    
    # Find matching context
    max_matches = 0
    best_context = "general"
    
    for context, terms in keywords.items():
        matches = sum(1 for term in terms if term in chat_log_lower)
        if matches > max_matches:
            max_matches = matches
            best_context = context
    
    # Generate a simple summary
    summary = chat_log[:100].replace("\n", " ") + "..."
    
    # Calculate a confidence score (higher for more keyword matches)
    confidence = min(0.5 + (max_matches * 0.1), 0.95)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    return {
        "context_type": best_context,
        "data": {
            "content": f"Conversation about {best_context}: {summary}",
            "confidence_score": confidence,
            "last_updated": today
        }
    }

def process_conversations_for_timestamp_range(extractor, start_timestamp, end_timestamp, use_ollama=True, output_dir="json_outputs"):
    """
    Process all conversations within a specific timestamp range.
    
    Args:
        extractor: RedisMessageExtractor instance
        start_timestamp: Start timestamp (inclusive)
        end_timestamp: End timestamp (exclusive)
        use_ollama: Whether to use Ollama for analysis
        output_dir: Directory to save JSON outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all conversations for the timestamp range
    conversations = extractor.get_messages_for_timestamp_range(start_timestamp, end_timestamp)
    
    date_str = datetime.fromtimestamp(start_timestamp).strftime("%Y-%m-%d")
    print(f"Found {len(conversations)} conversations for timestamp range (date: {date_str})")
    
    # Dictionary to group results by context type
    grouped_results = {}
    
    # Process each conversation
    for conv_id, messages in conversations.items():
        # Convert to chat log format
        chat_log = convert_to_chat_format(messages)
        
        # Skip if empty
        if not chat_log.strip():
            continue
        
        print(f"Processing conversation {conv_id} with {len(messages)} messages")
        
        # Analyze conversation
        if use_ollama:
            result = analyze_with_ollama(chat_log)
        else:
            result = basic_context_analysis(chat_log)
        
        # Get context type
        context_type = result.get("context_type", "unknown")
        
        # Add conversation ID for reference
        result["conversation_id"] = conv_id
        
        # Group by context type
        if context_type not in grouped_results:
            grouped_results[context_type] = []
        
        grouped_results[context_type].append(result)
        
        print(f"Identified context type: {context_type}")
    
    # Save each context type to a separate file
    for context_type, results in grouped_results.items():
        # Clean filename
        clean_type = re.sub(r'[^\w\-]', '_', context_type.lower())
        
        # Create consolidated result
        consolidated = {
            "context_type": context_type,
            "data": {
                "content": f"Multiple conversations about {context_type}",
                "entries": [r.get("data", {}).get("content", "") for r in results],
                "confidence_score": max([r.get("data", {}).get("confidence_score", 0) for r in results]),
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "conversation_ids": [r.get("conversation_id", "") for r in results]
            }
        }
        
        # Save to file
        filename = f"{clean_type}.json"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2)
        
        print(f"Saved context: {context_type} with {len(results)} conversations to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Extract and analyze Redis chat messages by timestamp")
    parser.add_argument("--date", "-d", help="Target date (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output", "-o", help="Output directory", default="json_outputs")
    parser.add_argument("--no-ollama", action="store_true", help="Don't use Ollama for analysis")
    parser.add_argument("--host", help="Redis host", default="localhost")
    parser.add_argument("--port", type=int, help="Redis port", default=6379)
    parser.add_argument("--db", type=int, help="Redis database", default=0)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = RedisMessageExtractor(host=args.host, port=args.port, db=args.db)
    
    # Check Redis connection
    if not extractor.ping():
        print("Failed to connect to Redis. Please check your connection settings.")
        return
    
    print(f"Connected to Redis. Processing conversations for {args.date}")
    
    # Convert date to timestamp range
    try:
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        start_timestamp = int(date_obj.timestamp())
        end_timestamp = start_timestamp + 86400  # Add 24 hours in seconds
        
        print(f"Timestamp range: {start_timestamp} to {end_timestamp}")
        print(f"Date range: {datetime.fromtimestamp(start_timestamp)} to {datetime.fromtimestamp(end_timestamp)}")
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD format.")
        return
    
    # If debug mode is enabled, print a sample of keys
    if args.debug:
        print("DEBUG MODE: Examining Redis keys")
        all_keys = extractor.redis_client.keys("message:*")
        print(f"Found {len(all_keys)} message keys in Redis")
        
        # Show a few sample keys
        if all_keys:
            print("\nSample keys:")
            for key in all_keys[:10]:
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                key_type = extractor.redis_client.type(key)
                print(f"  - {key} (Type: {key_type})")
                
                # Try to extract timestamp from key
                key_parts = key.split(":")
                if len(key_parts) >= 4:
                    try:
                        timestamp = int(key_parts[3])
                        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        print(f"    Timestamp: {timestamp}, Date: {date_str}")
                    except (ValueError, IndexError):
                        print(f"    Invalid timestamp format")
            
            # Find keys with today's date in their timestamp
            today_keys = []
            for key in all_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                key_parts = key.split(":")
                if len(key_parts) >= 4:
                    try:
                        timestamp = int(key_parts[3])
                        if start_timestamp <= timestamp < end_timestamp:
                            today_keys.append(key)
                    except (ValueError, IndexError):
                        continue
            
            print(f"\nFound {len(today_keys)} keys with timestamp from {args.date}")
            if today_keys:
                print("Sample keys from today:")
                for key in today_keys[:5]:
                    print(f"  - {key}")
    
    # Process conversations
    process_conversations_for_timestamp_range(
        extractor=extractor,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        use_ollama=not args.no_ollama,
        output_dir=args.output
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
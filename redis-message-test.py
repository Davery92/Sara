#!/usr/bin/env python3
"""
Test script to retrieve all messages from a specific day and sort them chronologically
using the Redis database.
"""

import sys
import os
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Redis client
try:
    from modules.redis_client import RedisClient
except ImportError:
    # Fallback: define a basic Redis client if the module isn't available
    import redis
    
    class RedisClient:
        def __init__(self, host='localhost', port=6379, db=0):
            self.redis_client = redis.Redis(host=host, port=port, db=db)
            
        def ping(self):
            return self.redis_client.ping()
            
        def get_message_by_key(self, key):
            try:
                # Check what type the key is
                key_type = self.redis_client.type(key)
                
                if key_type == "ReJSON-RL" or key_type == "JSON":
                    # It's a JSON object
                    try:
                        data = self.redis_client.json().get(key)
                        return data
                    except Exception as e:
                        logger.warning(f"Error getting JSON data for key {key}: {e}")
                
                elif key_type == "hash":
                    # It's a hash - get all fields
                    try:
                        data = self.redis_client.hgetall(key)
                        # Convert bytes to string for hash fields
                        result = {}
                        for field, value in data.items():
                            field_name = field.decode('utf-8', errors='replace') if isinstance(field, bytes) else field
                            field_value = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
                            result[field_name] = field_value
                        return result
                    except Exception as e:
                        logger.warning(f"Error processing hash for key {key}: {e}")
                
                elif key_type == "string":
                    # It's a string - try to parse as JSON
                    try:
                        raw_data = self.redis_client.get(key)
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode('utf-8', errors='replace')
                        
                        try:
                            return json.loads(raw_data)
                        except json.JSONDecodeError:
                            # If can't parse as JSON, return as is
                            return {"content": raw_data}
                    except Exception as e:
                        logger.warning(f"Error getting string for key {key}: {e}")
                
                # Fallback for other types
                try:
                    data = self.redis_client.get(key)
                    if data:
                        if isinstance(data, bytes):
                            return {"content": data.decode('utf-8', errors='replace')}
                        return {"content": str(data)}
                except Exception as e:
                    logger.warning(f"Fallback error for key {key}: {e}")
                
                return {"content": "(error retrieving content)"}
            
            except Exception as e:
                logger.warning(f"Error in get_message_by_key for {key}: {e}")
                return {"content": f"(error: {str(e)})"}
                
            return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis-message-test")

# Redis connection details - default values that can be overridden by command-line args
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve and sort messages from Redis by date")
    
    parser.add_argument(
        "--date", 
        type=str, 
        help="Date to retrieve messages for (YYYY-MM-DD format). Defaults to today."
    )
    
    parser.add_argument(
        "--previous", 
        action="store_true",
        help="Use the previous day's date"
    )
    
    parser.add_argument(
        "--today", 
        action="store_true",
        help="Use today's date (this is the default)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=DEFAULT_REDIS_HOST,
        help=f"Redis host. Default: {DEFAULT_REDIS_HOST}"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=DEFAULT_REDIS_PORT,
        help=f"Redis port. Default: {DEFAULT_REDIS_PORT}"
    )
    
    parser.add_argument(
        "--db", 
        type=int, 
        default=DEFAULT_REDIS_DB,
        help=f"Redis database number. Default: {DEFAULT_REDIS_DB}"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path. If not provided, output will be printed to stdout."
    )
    
    parser.add_argument(
        "--conversations", 
        action="store_true",
        help="Group messages by conversation ID."
    )
    
    return parser.parse_args()

def get_messages_by_date(client: RedisClient, date_str: str, debug=False) -> List[Dict[str, Any]]:
    """
    Get all messages for a specific date.
    
    Args:
        client: Redis client instance
        date_str: Date string in YYYY-MM-DD format
        debug: Whether to print debug information
        
    Returns:
        List of message dictionaries
    """
    try:
        # Get all message keys
        all_message_keys = client.redis_client.keys("message:*")
        
        if debug:
            logger.info(f"Found {len(all_message_keys)} total message keys")
            
        # Filter out embedding keys and decode bytes to strings
        message_keys = []
        for key in all_message_keys:
            if isinstance(key, bytes):
                key_str = key.decode('utf-8', errors='replace')
                if ":embedding" not in key_str:
                    message_keys.append(key_str)
            elif isinstance(key, str) and ":embedding" not in key:
                message_keys.append(key)
        
        if debug:
            logger.info(f"After filtering embeddings: {len(message_keys)} keys")
            # Sample some keys
            if message_keys:
                logger.info(f"Sample keys: {message_keys[:5]}")
        
        date_messages = []
        skipped_keys = 0
        non_integer_timestamps = 0
        date_mismatches = 0
        
        # Parse date from key and filter messages from the specified date
        for key_str in message_keys:
            try:
                # Format: message:conversation_id:role:timestamp
                parts = key_str.split(':')
                if len(parts) >= 4:
                    try:
                        timestamp = int(parts[3])
                    except ValueError:
                        non_integer_timestamps += 1
                        if debug and non_integer_timestamps <= 5:
                            logger.warning(f"Non-integer timestamp in key: {key_str}")
                        continue
                    
                    # Convert timestamp to date
                    message_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                    
                    # Check if message is from the specified date
                    if message_date == date_str:
                        # Get message content
                        message_data = client.get_message_by_key(key_str)
                        
                        if message_data:
                            # Try to extract content based on type of message_data
                            content = ""
                            if isinstance(message_data, dict):
                                content = message_data.get("content", "")
                            elif isinstance(message_data, str):
                                content = message_data
                            
                            message = {
                                "id": key_str,
                                "role": parts[2],
                                "content": content,
                                "conversation_id": parts[1],
                                "timestamp": timestamp,
                                "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                            }
                            date_messages.append(message)
                    else:
                        date_mismatches += 1
                        if debug and date_mismatches <= 5:
                            logger.info(f"Date mismatch: key={key_str}, message_date={message_date}, target_date={date_str}")
                else:
                    skipped_keys += 1
                    if debug and skipped_keys <= 5:
                        logger.warning(f"Skipped key with wrong format: {key_str}")
            except Exception as e:
                logger.warning(f"Error processing key {key_str}: {e}")
                continue
        
        # Sort messages by timestamp
        date_messages.sort(key=lambda x: x["timestamp"])
        
        if debug:
            logger.info(f"Processing stats: skipped_keys={skipped_keys}, non_integer_timestamps={non_integer_timestamps}, date_mismatches={date_mismatches}")
            
            # Print date range found in keys
            if message_keys:
                try:
                    dates_found = {}
                    for key in message_keys[:100]:  # Sample the first 100 keys
                        parts = key.split(':')
                        if len(parts) >= 4:
                            try:
                                ts = int(parts[3])
                                date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                                if date in dates_found:
                                    dates_found[date] += 1
                                else:
                                    dates_found[date] = 1
                            except (ValueError, Exception):
                                continue
                    logger.info(f"Dates found in sampled keys: {dates_found}")
                except Exception as e:
                    logger.error(f"Error analyzing date distribution: {e}")
        
        logger.info(f"Retrieved {len(date_messages)} messages for date {date_str}")
        return date_messages
        
    except Exception as e:
        logger.error(f"Error retrieving messages for date {date_str}: {e}")
        return []

def get_conversations_for_date(client: RedisClient, date_str: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all conversations for a specific date, with messages grouped by conversation ID.
    
    Args:
        client: Redis client instance
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Dictionary mapping conversation IDs to lists of message dictionaries
    """
    # Get all messages for the date
    messages = get_messages_by_date(client, date_str)
    
    # Group messages by conversation ID
    conversations = {}
    for message in messages:
        conv_id = message["conversation_id"]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(message)
    
    # Sort messages by timestamp within each conversation
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda m: m["timestamp"])
    
    logger.info(f"Grouped messages into {len(conversations)} conversations")
    return conversations

def get_conversation_title(client: RedisClient, conversation_id: str) -> str:
    """Get the title of a conversation if it exists"""
    try:
        title_key = f"conversation:title:{conversation_id}"
        title = client.redis_client.get(title_key)
        if title:
            return title.decode('utf-8', errors='replace')
        return f"Conversation {conversation_id}"
    except Exception as e:
        logger.warning(f"Error getting conversation title: {e}")
        return f"Conversation {conversation_id}"

def format_message(message: Dict[str, Any]) -> str:
    """Format a message for display."""
    dt = datetime.fromtimestamp(message["timestamp"])
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    role = message["role"].upper()
    content = message["content"]
    
    return f"[{formatted_time}] {role}: {content}"

def format_conversation(client: RedisClient, conv_id: str, messages: List[Dict[str, Any]]) -> str:
    """Format a conversation for display."""
    title = get_conversation_title(client, conv_id)
    
    result = f"\n{'=' * 80}\n"
    result += f"CONVERSATION: {title} (ID: {conv_id})\n"
    result += f"{'=' * 80}\n\n"
    
    for message in messages:
        result += format_message(message) + "\n\n"
    
    return result

def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine the date to retrieve messages for
    if args.date:
        date_str = args.date
    elif args.previous:
        # Use yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        logger.info(f"Using previous day's date: {date_str}")
    else:
        # Default to today
        today = datetime.now()
        date_str = today.strftime("%Y-%m-%d")
        logger.info(f"Using today's date: {date_str}")
    
    # Initialize Redis client
    client = RedisClient(host=args.host, port=args.port, db=args.db)
    
    try:
        # Check connection
        if not client.ping():
            logger.error("Failed to connect to Redis database")
            return 1
        
        logger.info(f"Connected to Redis database at {args.host}:{args.port}")
        
        # Add debugging flag to diagnose issues
        debug = True
        
        # Get some basic stats about the Redis database
        try:
            key_count = len(client.redis_client.keys("*"))
            message_key_count = len(client.redis_client.keys("message:*"))
            logger.info(f"Redis database stats: {key_count} total keys, {message_key_count} message keys")
            
            # List all available dates
            all_message_keys = client.redis_client.keys("message:*")
            dates = set()
            for key in all_message_keys:
                try:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    if ":embedding" not in key_str:
                        parts = key_str.split(":")
                        if len(parts) >= 4:
                            try:
                                timestamp = int(parts[3])
                                msg_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                                dates.add(msg_date)
                            except (ValueError, Exception) as e:
                                continue
                except Exception:
                    continue
            
            if dates:
                logger.info(f"Available dates in database: {sorted(dates)}")
                # Check if the requested date exists
                if date_str not in dates:
                    logger.warning(f"Requested date {date_str} not found in available dates")
                    
                    # Suggest the closest date
                    if dates:
                        closest_date = min(dates, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(date_str, "%Y-%m-%d")).days))
                        logger.info(f"Closest available date is {closest_date}")
                        
                        # Ask if user wants to use the closest date instead
                        if input(f"Use closest date {closest_date} instead? (y/n): ").lower() == 'y':
                            date_str = closest_date
                            logger.info(f"Using date {date_str} instead")
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
        
        # Get messages or conversations
        if args.conversations:
            # Group messages by conversation
            conversations = get_conversations_for_date(client, date_str)
            
            if not conversations:
                logger.info(f"No conversations found for date {date_str}")
                return 0
            
            # Format output
            output = f"Messages by conversation for date: {date_str}\n\n"
            for conv_id, messages in conversations.items():
                output += format_conversation(client, conv_id, messages)
        else:
            # Get all messages for the date with debug enabled
            messages = get_messages_by_date(client, date_str, debug=debug)
            
            if not messages:
                logger.info(f"No messages found for date {date_str}")
                
                # Try an alternative approach - get all messages and filter by internal date field
                logger.info("Trying alternative date extraction method...")
                
                all_message_keys = client.redis_client.keys("message:*")
                logger.info(f"Found {len(all_message_keys)} total message keys")
                
                alt_messages = []
                checked_keys = 0
                
                for key in all_message_keys:
                    try:
                        if checked_keys >= 100:  # Limit to first 100 keys for performance
                            break
                            
                        checked_keys += 1
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        if ":embedding" not in key_str:
                            # Get the message data
                            message_data = client.get_message_by_key(key_str)
                            
                            if message_data and isinstance(message_data, dict):
                                # Try to extract the date from the message data
                                if "date" in message_data:
                                    msg_date = message_data["date"]
                                    if msg_date == date_str:
                                        # Format based on key structure
                                        parts = key_str.split(':')
                                        timestamp = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else 0
                                        
                                        alt_messages.append({
                                            "id": key_str,
                                            "role": parts[2] if len(parts) >= 3 else "unknown",
                                            "content": message_data.get("content", ""),
                                            "conversation_id": parts[1] if len(parts) >= 2 else "unknown",
                                            "timestamp": timestamp,
                                            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else "unknown"
                                        })
                    except Exception as e:
                        logger.warning(f"Error in alternative method for key {key}: {e}")
                        continue
                
                if alt_messages:
                    logger.info(f"Alternative method found {len(alt_messages)} messages for date {date_str}")
                    messages = sorted(alt_messages, key=lambda x: x.get("timestamp", 0))
                else:
                    logger.info("Alternative method also found no messages for the specified date")
                    return 0
            
            # Format output
            output = f"All messages for date: {date_str}\n\n"
            for message in messages:
                output += format_message(message) + "\n\n"
        
        # Output results
        if args.output:
            # Generate default filename if not specified
            if args.output == "auto":
                output_filename = f"messages_{date_str}.txt"
                args.output = output_filename
                logger.info(f"Using auto-generated filename: {output_filename}")
            
            # Write to file
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.info(f"Output written to {args.output}")
        else:
            # Print to stdout
            print(output)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Set up better logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    sys.exit(main())
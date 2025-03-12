#!/usr/bin/env python3
"""
Script to read messages from a message.txt file, process them through an Ollama model
for contextual and semantic extraction, and prepare them for Neo4j import.

This script reads messages, extracts semantic information (entities, concepts, etc.),
creates a user profile with pattern analysis, and saves the results to files.
"""

import os
import json
import logging
import re
import uuid
import datetime
import argparse
from typing import List, Dict, Any, Optional, Union
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("message-extraction")

# Constants
OLLAMA_MODEL = "command-r7b"  # Using the semantic model from neo4j_rag_integration.py
EMBEDDING_MODEL = "bge-m3"    # Embedding model from neo4j_rag_integration.py
OUTPUT_FILE = "prepared-data.json"  # Changed to .json for clarity
DEFAULT_INPUT_FILE = "message.txt"
CONVERSATION_ANALYSIS = True  # Enable analysis of conversation dynamics

# Import ollama here - this allows the script to run even if ollama is not installed
# The error will only occur when the functions are actually called
try:
    import ollama
except ImportError:
    logger.warning("Ollama module not found. Please install with 'pip install ollama'")


def parse_messages(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the message.txt file into a list of message dictionaries.
    
    This function attempts to detect and parse various message formats commonly used.
    """
    logger.info(f"Parsing messages from {file_path}")
    
    messages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to detect format and split into individual messages
        # Common patterns: timestamps, date markers, sender indicators, etc.
        
        # Try several common message delimiter patterns
        delimiter_patterns = [
            r'\n-{3,}\n',                                 # Simple separators (---)
            r'\n=={3,}\n',                                # Double separators (===)
            r'\n\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # ISO timestamps 
            r'\n\[\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}\]', # WhatsApp-style timestamps
            r'\n\(\d{2}:\d{2}(:\d{2})?\)',                # Parenthesized timestamps
            r'\n\d{2}/\d{2}/\d{4} \d{2}:\d{2}(:\d{2})?',  # US-style dates
        ]
        
        # Try each delimiter pattern
        for pattern in delimiter_patterns:
            message_blocks = re.split(pattern, content)
            if len(message_blocks) > 1:
                logger.info(f"Found message delimiter pattern: {pattern}")
                # If we found a good delimiter pattern, break
                break
        else:
            # If no delimiter patterns worked, use simple newlines for paragraph-based splitting
            logger.warning("No standard delimiter pattern found, using paragraph breaks")
            message_blocks = re.split(r'\n\s*\n', content)
        
        # Common timestamp patterns to extract from message blocks
        timestamp_patterns = [
            (r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', '%Y-%m-%d %H:%M:%S'),  # ISO-like
            (r'\[(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2})\]', '%d/%m/%Y, %H:%M:%S'),  # WhatsApp
            (r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})', '%m/%d/%Y %H:%M:%S'),  # US date+time
            (r'(\d{2}:\d{2}:\d{2})', '%H:%M:%S')  # Just time
        ]
        
        # Common sender/user patterns
        sender_patterns = [
            r'From: ([^:]+)',  # From: User
            r'Sender: ([^:]+)',  # Sender: User
            r'([^:]+): ',  # User: Message
            r'<([^>]+)>',  # <User> Message
        ]
        
        for i, block in enumerate(message_blocks):
            if not block.strip():
                continue
                
            # Parse each message block
            block = block.strip()
            lines = block.split('\n')
            
            # Basic message data
            message_data = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "index": i,  # Keep track of message order
            }
            
            # Try to extract timestamp
            timestamp_found = False
            for line in lines[:3]:  # Check first few lines
                for pattern, time_format in timestamp_patterns:
                    match = re.search(pattern, line)
                    if match:
                        try:
                            dt = datetime.datetime.strptime(match.group(1), time_format)
                            message_data["timestamp"] = dt.isoformat()
                            timestamp_found = True
                            break
                        except ValueError:
                            pass
                if timestamp_found:
                    break
            
            # Try to extract sender/user
            sender_found = False
            for line in lines[:3]:  # Check first few lines
                for pattern in sender_patterns:
                    match = re.search(pattern, line)
                    if match:
                        message_data["role"] = match.group(1).strip()
                        sender_found = True
                        break
                if sender_found:
                    break
            
            # Look for metadata in key-value format
            for i, line in enumerate(lines[:5]):  # Check first few lines
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key, value = parts
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key in ["from", "sender", "user", "name"] and not sender_found:
                            message_data["role"] = value
                            sender_found = True
                        elif key in ["time", "date", "timestamp"] and not timestamp_found:
                            try:
                                # Try several date formats
                                for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y, %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                                    try:
                                        dt = datetime.datetime.strptime(value, fmt)
                                        message_data["timestamp"] = dt.isoformat()
                                        timestamp_found = True
                                        break
                                    except ValueError:
                                        continue
                            except:
                                pass
                        else:
                            # Store other metadata
                            message_data[key] = value
            
            # Extract content - try to find where the actual message starts
            content_start = 0
            for i, line in enumerate(lines):
                # Skip metadata/header lines
                if i < 5 and (":" in line or re.search(r'^\[.*\]', line) or re.search(r'^\(.*\)', line)):
                    content_start = i + 1
                else:
                    # Once we find a non-metadata line, stop searching
                    break
                    
            # Extract the message content
            if content_start >= len(lines):
                # If all lines were metadata, use the full block
                message_data["content"] = block
            else:
                message_data["content"] = "\n".join(lines[content_start:]).strip()
            
            # If content is empty or just the same as a metadata field, use full block
            if not message_data["content"] or message_data["content"] in message_data.values():
                message_data["content"] = block
            
            # Ensure required fields
            message_data.setdefault("role", "user")
            message_data.setdefault("conversation_id", "message-extract-" + datetime.datetime.now().strftime("%Y%m%d"))
            
            messages.append(message_data)
        
        logger.info(f"Parsed {len(messages)} messages")
        return messages
        
    except Exception as e:
        logger.error(f"Error parsing messages: {e}")
        return []


async def extract_semantic_info(text: str) -> Dict[str, Any]:
    """
    Extract semantic information from text using Ollama model.
    
    This function is adapted from neo4j_rag_integration.py but includes
    additional analysis to better support user profiling.
    """
    try:
        # Create a prompt for Ollama with enhanced analysis for user profiling
        prompt = f"""
        Analyze the following text and extract detailed semantic information in JSON format.
        Include these categories:
        - entities: List of named entities (people, places, organizations, etc.) as simple strings
        - concepts: List of abstract concepts and themes as simple strings
        - key_points: List of main ideas or facts as simple strings
        - sentiment: Overall sentiment (positive, negative, or neutral) as a simple string
        - sentiment_strength: Rate the strength of the sentiment (1-10) as a number
        - topics: List specific topics being discussed (more specific than concepts)
        - interests: Any user interests or preferences revealed in the text
        - tone: Descriptive words for the communication tone (formal, casual, urgent, etc.)
        - relevance: Rate the relevance and importance of this text (1-10) as a number
        - question_intent: If there are questions, categorize their intent (seeking information, rhetorical, etc.)
        
        Format the output as a valid JSON object with these fields.
        All values must be simple strings, numbers, or arrays of strings - not complex objects.

        Text to analyze:
        {text}
        """
        
        # Using the ollama API
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a semantic analysis assistant that extracts structured information from text and outputs only valid JSON with simple values - no nested objects or complex structures."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.get('message', {}).get('content', '')
        
        # Extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON found, use the whole response
                json_str = content
        
        # Parse JSON
        try:
            data = json.loads(json_str)
            
            # Process all fields to ensure they are primitive types
            # Helper function to process list fields
            def process_list_field(field_data, name_key='name', text_key='text'):
                processed = []
                if not field_data:
                    return processed
                    
                for item in field_data:
                    if isinstance(item, dict):
                        if name_key in item:
                            processed.append(item[name_key])
                        elif text_key in item:
                            processed.append(item[text_key])
                        elif len(item) > 0:
                            # Get first value in dict
                            processed.append(str(next(iter(item.values()))))
                    elif item:
                        processed.append(str(item))
                return processed
                
            # Helper function to process string fields
            def process_string_field(field_data, default=''):
                if isinstance(field_data, dict):
                    for key in ['value', 'text', 'analysis', 'result']:
                        if key in field_data:
                            return str(field_data[key])
                    # If no standard keys, take first value
                    if field_data:
                        return str(next(iter(field_data.values())))
                    return default
                elif field_data is None:
                    return default
                else:
                    return str(field_data)
                    
            # Helper function to process numeric fields
            def process_numeric_field(field_data, default=5):
                if isinstance(field_data, (int, float)):
                    return field_data
                elif isinstance(field_data, str):
                    try:
                        return float(field_data)
                    except ValueError:
                        return default
                elif isinstance(field_data, dict):
                    for key in ['value', 'score', 'rating', 'result']:
                        if key in field_data:
                            try:
                                return float(field_data[key])
                            except (ValueError, TypeError):
                                continue
                return default
            
            # Process all fields
            data['entities'] = process_list_field(data.get('entities', []))
            data['concepts'] = process_list_field(data.get('concepts', []))
            data['key_points'] = process_list_field(data.get('key_points', []), text_key='text')
            data['topics'] = process_list_field(data.get('topics', []))
            data['interests'] = process_list_field(data.get('interests', []))
            
            data['sentiment'] = process_string_field(data.get('sentiment'), 'neutral')
            data['tone'] = process_string_field(data.get('tone'), 'neutral')
            data['question_intent'] = process_string_field(data.get('question_intent'), '')
            
            data['relevance'] = process_numeric_field(data.get('relevance'), 5)
            data['sentiment_strength'] = process_numeric_field(data.get('sentiment_strength'), 5)
            
            # Ensure default values for all fields
            data.setdefault('entities', [])
            data.setdefault('concepts', [])
            data.setdefault('key_points', [])
            data.setdefault('topics', [])
            data.setdefault('interests', [])
            data.setdefault('sentiment', 'neutral')
            data.setdefault('sentiment_strength', 5)
            data.setdefault('tone', 'neutral')
            data.setdefault('relevance', 5)
            data.setdefault('question_intent', '')
            
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Ollama response: {content}")
            # Return default structure
            return {
                'entities': [],
                'concepts': [],
                'key_points': [],
                'topics': [],
                'interests': [],
                'sentiment': 'neutral',
                'sentiment_strength': 5,
                'tone': 'neutral',
                'relevance': 5,
                'question_intent': ''
            }
                
    except Exception as e:
        logger.error(f"Error extracting semantic info: {e}")
        # Return default structure
        return {
            'entities': [],
            'concepts': [],
            'key_points': [],
            'topics': [],
            'interests': [],
            'sentiment': 'neutral',
            'sentiment_strength': 5,
            'tone': 'neutral',
            'relevance': 5,
            'question_intent': ''
        }


def get_embedding(text: str) -> List[float]:
    """
    Get vector embedding for text using Ollama.
    
    This uses the BGE-M3 model as in the RAG implementation.
    """
    try:
        # Truncate very long texts to avoid Ollama errors
        # BGE-M3 has context limits
        max_chars = 8192
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} characters for embedding")
            # Try to truncate at sentence boundary
            truncation_point = text.rfind(".", 0, max_chars)
            if truncation_point == -1:
                truncation_point = max_chars
            text = text[:truncation_point+1]
        
        embedding_result = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        
        embedding = embedding_result.get('embedding', [])
        
        if not embedding:
            logger.warning("Empty embedding returned")
            
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []


def analyze_user_patterns(messages: List[Dict[str, Any]], semantic_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze messages to identify user patterns and create a profile.
    
    Extracts:
    - Common topics and interests
    - Sentiment patterns
    - Activity patterns (time of day, frequency)
    - Common entities mentioned
    - Writing style characteristics
    """
    if not messages or not semantic_data:
        return {
            "error": "Insufficient data for analysis",
            "topics": [],
            "sentiment_distribution": {},
            "activity_patterns": {},
            "common_entities": [],
            "writing_style": {}
        }
    
    # Extract all user messages (excluding system or assistant)
    user_messages = []
    user_semantics = []
    
    for i, message in enumerate(messages):
        if message.get("role", "").lower() in ["user", ""]:
            user_messages.append(message)
            if i < len(semantic_data):
                user_semantics.append(semantic_data[i])
    
    # Topic analysis - gather concepts across messages
    all_concepts = []
    for semantics in user_semantics:
        all_concepts.extend(semantics.get("concepts", []))
    
    # Count concept frequencies
    concept_counter = {}
    for concept in all_concepts:
        concept_counter[concept] = concept_counter.get(concept, 0) + 1
    
    # Get top concepts/topics
    top_topics = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Sentiment analysis
    sentiment_counter = {}
    for semantics in user_semantics:
        sentiment = semantics.get("sentiment", "neutral")
        sentiment_counter[sentiment] = sentiment_counter.get(sentiment, 0) + 1
    
    # Calculate sentiment percentages
    total_messages = len(user_semantics) or 1  # Avoid division by zero
    sentiment_distribution = {
        sentiment: {
            "count": count,
            "percentage": round((count / total_messages) * 100, 1)
        }
        for sentiment, count in sentiment_counter.items()
    }
    
    # Activity patterns - time analysis
    hour_counter = {}
    weekday_counter = {}
    
    for message in user_messages:
        try:
            timestamp = message.get("timestamp")
            if timestamp:
                dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                weekday = dt.strftime('%A')
                
                hour_counter[hour] = hour_counter.get(hour, 0) + 1
                weekday_counter[weekday] = weekday_counter.get(weekday, 0) + 1
        except (ValueError, TypeError):
            continue
    
    # Entity analysis
    all_entities = []
    for semantics in user_semantics:
        all_entities.extend(semantics.get("entities", []))
    
    # Count entity frequencies
    entity_counter = {}
    for entity in all_entities:
        entity_counter[entity] = entity_counter.get(entity, 0) + 1
    
    # Get top entities
    top_entities = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Writing style analysis
    total_words = 0
    total_sentences = 0
    word_lengths = []
    sentence_lengths = []
    question_count = 0
    exclamation_count = 0
    
    for message in user_messages:
        content = message.get("content", "")
        
        # Sentence detection
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences += len(sentences)
        
        # Track sentence lengths
        for sentence in sentences:
            words = sentence.split()
            if words:
                sentence_lengths.append(len(words))
        
        # Word analysis
        words = content.split()
        total_words += len(words)
        
        # Track word lengths
        for word in words:
            word = word.strip('.,!?:;()"\'')
            if word:
                word_lengths.append(len(word))
        
        # Question and exclamation mark usage
        question_count += content.count('?')
        exclamation_count += content.count('!')
    
    # Calculate averages
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    
    # Create user profile
    user_profile = {
        "topics": [{"topic": topic, "count": count} for topic, count in top_topics],
        "sentiment_distribution": sentiment_distribution,
        "activity_patterns": {
            "by_hour": hour_counter,
            "by_weekday": weekday_counter
        },
        "common_entities": [{"entity": entity, "count": count} for entity, count in top_entities],
        "writing_style": {
            "average_word_length": round(avg_word_length, 1),
            "average_sentence_length": round(avg_sentence_length, 1),
            "total_messages": len(user_messages),
            "total_words": total_words,
            "total_sentences": total_sentences,
            "questions_per_message": round(question_count / len(user_messages), 2) if user_messages else 0,
            "exclamations_per_message": round(exclamation_count / len(user_messages), 2) if user_messages else 0,
        }
    }
    
    # Generate insights based on the analyzed data
    insights = []
    
    # Topic insights
    if top_topics:
        insights.append(f"User frequently discusses topics related to: {', '.join([t[0] for t in top_topics[:3]])}")
    
    # Sentiment insights
    dominant_sentiment = max(sentiment_counter.items(), key=lambda x: x[1])[0] if sentiment_counter else "neutral"
    insights.append(f"User's messages are predominantly {dominant_sentiment} in tone")
    
    # Activity insights
    if hour_counter:
        most_active_hour = max(hour_counter.items(), key=lambda x: x[1])[0]
        insights.append(f"User is most active around {most_active_hour}:00")
    
    # Writing style insights
    if avg_sentence_length > 20:
        insights.append("User tends to write in long, complex sentences")
    elif avg_sentence_length < 8:
        insights.append("User tends to write in short, concise sentences")
    
    if question_count > len(user_messages) * 0.5:
        insights.append("User asks a lot of questions relative to their message count")
    
    user_profile["insights"] = insights
    
    return user_profile


async def analyze_conversation_dynamics(messages: List[Dict[str, Any]], semantic_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the dynamics of the conversation:
    - Flow and coherence between messages
    - Question-answer patterns
    - Topic transitions
    - Engagement metrics
    """
    if not messages or len(messages) < 3:
        return {"error": "Not enough messages to analyze conversation dynamics"}
    
    # Identify different participants
    participants = set()
    for message in messages:
        role = message.get("role", "").lower()
        if role:
            participants.add(role)
    
    # Prepare conversation context for Ollama
    conversation_sample = []
    # Select a representative sample (beginning, middle, end)
    if len(messages) <= 10:
        # Use all messages if 10 or fewer
        sample_indices = range(len(messages))
    else:
        # Select beginning, middle and end portions
        sample_indices = list(range(3)) + list(range(len(messages)//2-1, len(messages)//2+2)) + list(range(len(messages)-3, len(messages)))
    
    for idx in sample_indices:
        if idx < len(messages):
            msg = messages[idx]
            # Truncate very long messages for the sample
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            
            conversation_sample.append({
                "role": msg.get("role", "user"),
                "content": content
            })
    
    # Analyze the conversation with Ollama
    try:
        prompt = f"""
        Analyze this conversation sample and provide a structured analysis in JSON format:
        
        1. "coherence": Rate overall coherence (1-10) where 10 is perfectly coherent
        2. "topic_shifts": Identify any major topic shifts
        3. "interaction_pattern": Describe the interaction pattern (e.g., Q&A, debate, casual chat)
        4. "conversation_type": Categorize the type of conversation (e.g., technical discussion, social chat, support)
        5. "engagement_level": Rate engagement level (1-10) where 10 is highly engaged
        
        Format as valid JSON with simple values (strings, numbers, arrays of strings).
        
        Conversation sample (note this is a subset of the full conversation):
        {json.dumps(conversation_sample, indent=2)}
        """
        
        # Get analysis from Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a conversation analysis expert who provides structured JSON analysis of conversations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.get('message', {}).get('content', '')
        
        # Extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON found, use the whole response
                json_str = content
        
        # Parse JSON
        try:
            dynamics = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Ollama response for conversation dynamics")
            dynamics = {
                "coherence": 5,
                "topic_shifts": ["Unable to identify"],
                "interaction_pattern": "Unknown",
                "conversation_type": "Unknown",
                "engagement_level": 5
            }
        
        # Calculate basic metrics
        message_count = len(messages)
        avg_message_length = sum(len(msg.get("content", "")) for msg in messages) / message_count if message_count else 0
        
        # Enhancement: Calculate response times if timestamps are available
        response_times = []
        for i in range(1, len(messages)):
            try:
                prev_time = datetime.datetime.fromisoformat(messages[i-1].get("timestamp", "").replace('Z', '+00:00'))
                curr_time = datetime.datetime.fromisoformat(messages[i].get("timestamp", "").replace('Z', '+00:00'))
                delta = (curr_time - prev_time).total_seconds()
                if 0 < delta < 86400:  # Only count reasonable deltas (< 1 day)
                    response_times.append(delta)
            except (ValueError, TypeError):
                continue
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        # Add calculated metrics to the analysis
        dynamics["message_count"] = message_count
        dynamics["participant_count"] = len(participants)
        dynamics["avg_message_length"] = round(avg_message_length, 1)
        if avg_response_time:
            dynamics["avg_response_time_seconds"] = round(avg_response_time, 1)
        
        return dynamics
        
    except Exception as e:
        logger.error(f"Error analyzing conversation dynamics: {e}")
        return {
            "error": f"Failed to analyze conversation dynamics: {str(e)}",
            "message_count": len(messages),
            "participant_count": len(participants)
        }


def format_for_neo4j(messages: List[Dict[str, Any]], semantic_data: List[Dict[str, Any]], 
                    embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Format messages and their semantic data for Neo4j import.
    
    Also includes user pattern analysis and profiling.
    """
    # First, analyze user patterns
    user_profile = analyze_user_patterns(messages, semantic_data)
    
    neo4j_data = []
    
    # Process individual messages
    for i, (message, semantics, embedding) in enumerate(zip(messages, semantic_data, embeddings)):
        # Create Neo4j-ready structure
        neo4j_message = {
            "message": {
                "id": message.get("id", f"message:{uuid.uuid4()}"),
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
                "conversation_id": message.get("conversation_id", "default"),
                "timestamp": message.get("timestamp", datetime.datetime.now().isoformat()),
                "date": message.get("date", datetime.datetime.now().strftime("%Y-%m-%d")),
                "index": message.get("index", i),  # Preserve original order
                "has_embedding": len(embedding) > 0
            },
            "semantic": {
                "entities": semantics.get("entities", []),
                "concepts": semantics.get("concepts", []),
                "key_points": semantics.get("key_points", []),
                "sentiment": semantics.get("sentiment", "neutral"),
                "relevance": semantics.get("relevance", 5)
            }
        }
        
        # Add embedding sample (first 10 dimensions) as in neo4j_rag_integration.py
        if embedding:
            neo4j_message["embedding"] = {
                "dimensions": len(embedding),
                "sample": embedding[:10]  # Store first 10 dimensions as sample
            }
        
        neo4j_data.append(neo4j_message)
    
    # Add user profile as a separate object
    neo4j_data.append({
        "user_profile": user_profile,
        "timestamp": datetime.datetime.now().isoformat(),
        "message_count": len(messages),
        "analysis_version": "1.0"
    })
    
    return neo4j_data


def save_to_file(data: List[Dict[str, Any]], output_file: str = OUTPUT_FILE):
    """
    Save the prepared data to a file instead of uploading to Neo4j.
    Also save a separate user profile file for easy review.
    """
    try:
        # Save complete data to primary output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved prepared data to {output_file}")
        
        # Extract and save user profile to a separate file for easy review
        user_profile = None
        for item in data:
            if 'user_profile' in item:
                user_profile = item
                break
                
        if user_profile:
            profile_file = output_file.replace('.txt', '_profile.json').replace('.json', '_profile.json')
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(user_profile, f, indent=2)
            logger.info(f"Saved user profile to {profile_file}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to file: {e}")
        return False


def generate_user_profile_summary(profile: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the user profile.
    """
    if not profile or "error" in profile:
        return "Insufficient data for user profile analysis."
    
    # Start building the summary
    summary = "USER PROFILE SUMMARY\n"
    summary += "=" * 80 + "\n\n"
    
    # Add insights if available
    if "insights" in profile and profile["insights"]:
        summary += "Key Insights:\n"
        for i, insight in enumerate(profile["insights"], 1):
            summary += f"{i}. {insight}\n"
        summary += "\n"
    
    # Add topics
    if "topics" in profile and profile["topics"]:
        summary += "Frequent Topics:\n"
        for topic in profile["topics"][:5]:  # Show top 5
            topic_name = topic.get("topic", "")
            topic_count = topic.get("count", 0)
            summary += f"- {topic_name} ({topic_count} mentions)\n"
        summary += "\n"
    
    # Add sentiment distribution
    if "sentiment_distribution" in profile and profile["sentiment_distribution"]:
        summary += "Sentiment Distribution:\n"
        for sentiment, data in profile["sentiment_distribution"].items():
            percentage = data.get("percentage", 0)
            summary += f"- {sentiment.capitalize()}: {percentage}%\n"
        summary += "\n"
    
    # Add writing style
    if "writing_style" in profile and profile["writing_style"]:
        style = profile["writing_style"]
        summary += "Writing Style:\n"
        summary += f"- Average sentence length: {style.get('average_sentence_length', 0)} words\n"
        summary += f"- Average word length: {style.get('average_word_length', 0)} characters\n"
        summary += f"- Questions per message: {style.get('questions_per_message', 0)}\n"
        summary += f"- Exclamations per message: {style.get('exclamations_per_message', 0)}\n"
        summary += "\n"
    
    # Add common entities
    if "common_entities" in profile and profile["common_entities"]:
        summary += "Frequently Mentioned Entities:\n"
        for entity in profile["common_entities"][:7]:  # Show top 7
            entity_name = entity.get("entity", "")
            entity_count = entity.get("count", 0)
            summary += f"- {entity_name} ({entity_count} mentions)\n"
        summary += "\n"
    
    # Add activity patterns
    if "activity_patterns" in profile and "by_hour" in profile["activity_patterns"]:
        hours = profile["activity_patterns"]["by_hour"]
        if hours:
            most_active_hour = max(hours.items(), key=lambda x: x[1])[0]
            summary += f"Most active hour: {most_active_hour}:00\n\n"
    
    summary += "=" * 80 + "\n"
    summary += "This profile is based on automated analysis and may not be fully accurate."
    
    return summary


async def process_messages(input_file: str, output_file: str):
    """
    Main function to process messages:
    1. Parse message.txt
    2. Extract semantic information
    3. Generate embeddings
    4. Analyze user patterns and create profile
    5. Analyze conversation dynamics
    6. Format for Neo4j
    7. Save to prepared-data.json and generate summary
    """
    logger.info(f"Starting message processing from {input_file}")
    
    # Parse messages
    messages = parse_messages(input_file)
    if not messages:
        logger.error("No messages found to process")
        return False
        
    # Sort messages by timestamp to ensure correct chronological order
    try:
        messages.sort(key=lambda m: m.get("timestamp", ""))
        logger.info("Messages sorted by timestamp")
    except Exception as e:
        logger.warning(f"Couldn't sort messages by timestamp: {e}")
        # Use the index field as backup
        messages.sort(key=lambda m: m.get("index", 0))
    
    # Process each message
    semantic_data = []
    embeddings = []
    
    # Process in batches to avoid memory issues
    batch_size = 10
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(messages)-1)//batch_size + 1}")
        
        for message in batch:
            # Extract semantic information
            semantics = await extract_semantic_info(message["content"])
            semantic_data.append(semantics)
            
            # Generate embedding
            embedding = get_embedding(message["content"])
            embeddings.append(embedding)
    
    # Analyze conversation dynamics if enabled
    conversation_dynamics = None
    if CONVERSATION_ANALYSIS and len(messages) >= 3:
        logger.info("Analyzing conversation dynamics")
        conversation_dynamics = await analyze_conversation_dynamics(messages, semantic_data)
    
    # Format for Neo4j
    neo4j_data = format_for_neo4j(messages, semantic_data, embeddings)
    
    # Add conversation dynamics to the data
    if conversation_dynamics:
        neo4j_data.append({
            "conversation_dynamics": conversation_dynamics,
            "timestamp": datetime.datetime.now().isoformat(),
            "message_count": len(messages),
            "analysis_version": "1.0"
        })
    
    # Save to file
    success = save_to_file(neo4j_data, output_file)
    
    if success:
        logger.info(f"Successfully processed {len(messages)} messages")
        
        # Generate a summary file with key insights
        try:
            # Extract user profile
            user_profile = None
            for item in neo4j_data:
                if 'user_profile' in item:
                    user_profile = item['user_profile']
                    break
                    
            if user_profile:
                summary = generate_user_profile_summary(user_profile)
                
                # Add conversation dynamics to summary if available
                if conversation_dynamics and not "error" in conversation_dynamics:
                    summary += "\n\nCONVERSATION DYNAMICS\n"
                    summary += "=" * 80 + "\n\n"
                    summary += f"Conversation Type: {conversation_dynamics.get('conversation_type', 'Unknown')}\n"
                    summary += f"Interaction Pattern: {conversation_dynamics.get('interaction_pattern', 'Unknown')}\n"
                    summary += f"Coherence (1-10): {conversation_dynamics.get('coherence', 'N/A')}\n"
                    summary += f"Engagement Level (1-10): {conversation_dynamics.get('engagement_level', 'N/A')}\n\n"
                    
                    if "topic_shifts" in conversation_dynamics:
                        summary += "Topic Shifts:\n"
                        for shift in conversation_dynamics["topic_shifts"]:
                            summary += f"- {shift}\n"
                
                summary_file = output_file.replace('.json', '_summary.txt').replace('.txt', '_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Created summary file at {summary_file}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
    
    return success


def main():
    """Entry point for the script"""
    global OLLAMA_MODEL
    print("Starting message extraction and analysis script...")
    
    parser = argparse.ArgumentParser(description='Process messages for Neo4j import with user profiling')
    parser.add_argument('--input', default=DEFAULT_INPUT_FILE, 
                        help=f'Input message file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', default=OUTPUT_FILE, 
                        help=f'Output prepared data file (default: {OUTPUT_FILE})')
    parser.add_argument('--model', default=OLLAMA_MODEL,
                        help=f'Ollama model for semantic analysis (default: {OLLAMA_MODEL})')
    parser.add_argument('--summary', action='store_true',
                        help='Generate a human-readable summary of the user profile')
    
    args = parser.parse_args()
    print(f"Arguments parsed: input={args.input}, output={args.output}")
    
    # Update model if specified
    
    if args.model:
        OLLAMA_MODEL = args.model
        logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
    
    try:
        # Run the async function
        print("Starting message processing...")
        result = asyncio.run(process_messages(args.input, args.output))
        
        print(f"\nProcessing complete. Data saved to {args.output}")
        if result:
            print("✓ Success!")
        else:
            print("✗ There were errors during processing. Check the logs for details.")
    except Exception as e:
        print(f"Error during processing: {e}")
        logger.error(f"Uncaught exception: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
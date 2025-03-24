import subprocess
from fastapi import FastAPI, Body, HTTPException, Request, Depends
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Callable, Optional, Union, Literal
import chromadb
import json
import ollama
import aiohttp
from starlette.responses import StreamingResponse, JSONResponse
import asyncio
from uuid import uuid4
from datetime import date, datetime
import logging
import sys
from modules.perplexica_module import PerplexicaClient
from datetime import date, datetime, timedelta
import logging
import redis
import json
import numpy as np
from redis.commands.search.field import (
    TextField, 
    VectorField,
    TagField
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import os
from fastapi import BackgroundTasks, APIRouter
from modules.neo4j_rag_integration import Neo4jRAGManager, get_neo4j_rag_manager
from modules.rag_api import rag_router  # Import this first
from modules.rag_web_interface import integrate_web_interface
from modules.rag_integration import integrate_rag_with_server, update_system_prompt_with_rag_info
from fastapi.middleware.cors import CORSMiddleware
from modules.timer_reminder_integration import integrate_timer_reminder_tools
from modules.neo4j_integration import get_message_store, integrate_neo4j_with_server
from modules.intent_classifier import get_intent_classifier
from modules.intent_tool_mapping import get_tools_for_intent, INTENT_CONFIDENCE_THRESHOLDS
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse, Response
import psutil
from modules.neo4j_connection import check_neo4j_connection
import tiktoken
from openai import OpenAI
import requests
import re


# Initialize the message store with Neo4j backend
message_store = get_message_store()
dashboard_router = APIRouter()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/david/sara-jarvis/Test/openai_server.log")
    ]
)
logger = logging.getLogger("openai-compatible-server")
logging.getLogger("httpx").setLevel(logging.WARNING)
app = FastAPI(title="OpenAI API Compatible Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the notes directory
NOTES_DIRECTORY = "/home/david/Sara/notes"

MESSAGE_HISTORY = []
MAX_HISTORY = 60  # Maximum number of messages (excluding system prompt)

tts_client = OpenAI(
    base_url="http://10.185.1.8:8880/v1", 
    api_key="not-needed"
)


# Ensure notes directory exists
os.makedirs(NOTES_DIRECTORY, exist_ok=True)
SKIP_TOOL_FOLLOWUP = True

# Initialize Perplexica client
perplexica = PerplexicaClient(base_url="http://localhost:3001")
SYSTEM_PROMPT = ""  # Will be loaded during startup
TOOL_SYSTEM_PROMPT_TEMPLATE = ""  # Will be loaded during startup
CORE_MEMORY_FILE = "/home/david/Sara/core_memories.txt"
# Model mapping from OpenAI to local models
MODEL_MAPPING = {
    "gpt-4": "llama3.3",
    "gpt-3.5-turbo": "gemma3:12b",
    "gpt-3.5-turbo-0125": "command-r7b",
    "gpt-3.5-turbo-1106": "llama3.2",
    "gpt-4o-mini": "llama3.1",
    "gpt-4o": "qwen2.5:32b",
    # Add more mappings as needed
    "default": "gemma3:27b"
}

# Available local models
AVAILABLE_MODELS = [
    "llama3.3:latest",
    "mistral-small:latest",
    "llama3.2:latest",
    "llama3.1:latest",
    "command-r7b",
    "gemma3:27b",
    "qwen2.5:32b",
]

# URLs for different models
MODEL_URLS = {
    "llama3.3": "http://100.104.68.115:11434/api/chat",
    "llama3.2": "http://localhost:11434/api/chat",
    "llama3.1": "http://localhost:11434/api/chat",
    "gemma3:27b": "http://localhost:11434/api/chat",
    "command-r7b": "http://localhost:11434/api/chat",
    "default": "http://localhost:11434/api/chat",
    "qwen2.5:14b": "http://localhost:11434/api/chat",
    "qwen2.5:32b": "http://100.104.68.115:11434/api/chat",
    "gemma3:12b": "http://localhost:11434/api/chat",
}




# Redis connection setup
class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.message_store = redis.Redis(
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
        self._create_indices()
    
    def _create_indices(self):
        """Create the necessary indices for vector search and conversation history"""
        # Check if index already exists
        try:
            self.message_store.ft("message_idx").info()
            print("Message index already exists")
        except:
            # Define schema for conversation messages
            schema = (
                TextField("$.role", as_name="role"),               # user or assistant
                TextField("$.content", as_name="content"),         # message content
                TextField("$.conversation_id", as_name="conversation_id"),  # to group messages
                TextField("$.conversation_date", as_name="conversation_date"),  # For time-based filtering
                TagField("$.date", as_name="date"),                # for filtering by date
                VectorField("$.embedding", 
                           "HNSW", {                                # vector index for similarity search
                               "TYPE": "FLOAT32", 
                               "DIM": 384,                          # dimension for your model
                               "DISTANCE_METRIC": "COSINE"
                           }, as_name="embedding")
            )
            
            # Create index
            try:
                self.message_store.ft("message_idx").create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=["message:"],
                        index_type=IndexType.JSON
                    )
                )
                print("Created message index")
            except Exception as e:
                print(f"Error creating index: {e}")
    
    def store_message(self, role, content, conversation_id, date, embedding=None):
        """Store a message with its metadata and embedding"""
        message_id = f"message:{conversation_id}:{role}:{int(date.timestamp())}"
        
        # Ensure content is a string
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception as e:
                logger.error(f"Failed to convert content to string: {str(e)}")
                content = "Error: Could not convert content to string"
        
        message_data = {
            "role": role,
            "content": content,
            "conversation_id": conversation_id,
            "date": date.strftime("%Y-%m-%d"),
            "timestamp": int(date.timestamp())
        }
        
        # Store the main message data first
        try:
            self.message_store.json().set(message_id, '$', message_data)
            
            # If embedding is provided, store it separately
            if embedding is not None:
                try:
                    # Store embedding in a separate key
                    embedding_key = f"{message_id}:embedding"
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    self.redis_binary.set(embedding_key, embedding_bytes)
                except Exception as e:
                    logger.error(f"Error storing embedding for message {message_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error storing message {message_id}: {str(e)}")
        
        return message_id
    
    def get_messages_by_conversation(self, conversation_id, limit=100):
        """Get messages for a conversation (improved implementation with better error handling)"""
        try:
            # Use pattern matching to find all messages for this conversation
            pattern = f"message:{conversation_id}:*"
            message_keys = self.message_store.keys(pattern)
            
            # Skip embedding keys
            filtered_keys = []
            for key in message_keys:
                # Skip embedding-related keys
                if isinstance(key, bytes):
                    key_str = key.decode('utf-8', errors='ignore')
                    if not key_str.endswith(":embedding"):
                        filtered_keys.append(key)
                elif not key.endswith(":embedding"):
                    filtered_keys.append(key)
            
            messages = []
            for key in filtered_keys[:limit]:
                try:
                    # Try to get the data, handling different Redis data types
                    key_type = self.redis_client.type(key)
                    
                    if key_type == "hash":
                        # For hash data, get all fields with proper error handling
                        raw_data = self.redis_client.hgetall(key)
                        if raw_data:
                            message_data = {}
                            for field, value in raw_data.items():
                                # Safely decode field names
                                field_name = field.decode('utf-8', errors='ignore') if isinstance(field, bytes) else field
                                
                                # Handle binary data in values
                                if isinstance(value, bytes):
                                    try:
                                        # Try to decode, but ignore errors
                                        decoded_value = value.decode('utf-8', errors='ignore')
                                        message_data[field_name] = decoded_value
                                    except Exception:
                                        # If decoding fails entirely, store as a placeholder
                                        message_data[field_name] = "(binary data)"
                                else:
                                    message_data[field_name] = value
                            
                            # Extract role and content for the message format
                            messages.append({
                                "role": message_data.get("role", "unknown"),
                                "content": message_data.get("content", ""),
                                "timestamp": int(message_data.get("timestamp", 0)) if message_data.get("timestamp", "").isdigit() else 0
                            })
                    
                    elif key_type == "ReJSON-RL" or key_type == "JSON":
                        # For JSON data
                        try:
                            json_data = self.redis_client.json().get(key)
                            if json_data and isinstance(json_data, dict):
                                messages.append({
                                    "role": json_data.get("role", "unknown"),
                                    "content": json_data.get("content", ""),
                                    "timestamp": int(json_data.get("timestamp", 0)) if isinstance(json_data.get("timestamp", ""), str) and json_data.get("timestamp", "").isdigit() else 0
                                })
                        except Exception as json_err:
                            print(f"Error getting JSON for key {key}: {json_err}")
                    
                    elif key_type == "string":
                        # For string data, try to parse as JSON
                        try:
                            string_value = self.redis_client.get(key)
                            if isinstance(string_value, bytes):
                                string_value = string_value.decode('utf-8', errors='ignore')
                                
                            # Try to parse as JSON
                            import json
                            try:
                                parsed_data = json.loads(string_value)
                                if isinstance(parsed_data, dict):
                                    messages.append({
                                        "role": parsed_data.get("role", "unknown"),
                                        "content": parsed_data.get("content", ""),
                                        "timestamp": int(parsed_data.get("timestamp", 0)) if isinstance(parsed_data.get("timestamp", ""), str) and parsed_data.get("timestamp", "").isdigit() else 0
                                    })
                            except json.JSONDecodeError:
                                # Not JSON, just use as raw content
                                messages.append({
                                    "role": "unknown",
                                    "content": string_value,
                                    "timestamp": 0
                                })
                        except Exception as str_err:
                            print(f"Error processing string for key {key}: {str_err}")
                    
                except Exception as e:
                    print(f"Error processing message key {key}: {e}")
                    continue
                    
            # Sort messages by timestamp
            messages.sort(key=lambda x: x.get("timestamp", 0))
            
            return messages
                
        except Exception as e:
            print(f"Error retrieving messages: {e}")
            return []
    
    def vector_search(self, embedding, k=3):
        """Find similar messages based on vector similarity"""
        # Convert embedding to bytes for the query
        query_vector = np.array(embedding, dtype=np.float32).tobytes()
        
        # Run the vector search query
        query = (
            f"*=>[KNN {k} @embedding $query_vector AS score]"
        )
        
        # Execute the query
        results = self.message_store.ft("message_idx").search(
            query,
            query_params={"query_vector": query_vector}
        )
        
        # Process results
        similar_docs = []
        for doc in results.docs:
            message = json.loads(doc.json)
            similar_docs.append({
                "role": message["role"],
                "content": message["content"],
                "score": doc.score
            })
        
        return similar_docs

def ping(self):
    """Test the Redis connection"""
    try:
        return self.message_store.ping()
    except Exception as e:
        logger.error(f"Redis ping failed: {e}")
        return False

def count_tokens(text, model="cl100k_base"):
    """Count the number of tokens in a text string."""
    try:
        encoder = tiktoken.get_encoding(model)
        return len(encoder.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        # Fallback simple estimate: ~4 chars per token on average
        return len(text) // 4

def save_conversation_history():
    """Save the current conversation history to Redis for persistence"""
    try:
        # Convert the conversation history to JSON
        history_json = json.dumps(MESSAGE_HISTORY)
        
        # Store in Redis - Fix: use client.redis_client
        message_store.client.redis_client.set("conversation_history", history_json)
        
        logger.info(f"Saved conversation history ({len(MESSAGE_HISTORY)} messages) to Redis")
        return True
    except Exception as e:
        logger.error(f"Error saving conversation history: {str(e)}")
        return False
    
def load_conversation_history():
    """Load the conversation history from Redis"""
    global MESSAGE_HISTORY
    
    try:
        # Get the saved history from Redis
        history_json = message_store.client.redis_client.get("conversation_history")
        
        if history_json:
            if isinstance(history_json, bytes):
                history_json = history_json.decode('utf-8')
                
            # Parse the JSON
            loaded_history = json.loads(history_json)
            
            # Validate each message has required fields
            valid_history = []
            for msg in loaded_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Add token count if missing
                    if 'tokens' not in msg:
                        msg['tokens'] = count_tokens(msg['content'])
                    valid_history.append(msg)
            
            # Update the MESSAGE_HISTORY
            MESSAGE_HISTORY = valid_history
            
            logger.info(f"Loaded conversation history from Redis ({len(MESSAGE_HISTORY)} messages)")
            return True
    except Exception as e:
        logger.error(f"Error loading conversation history: {str(e)}")
    
    # If we reach here, we either had an error or no history was found
    MESSAGE_HISTORY = []
    return False

@app.on_event("startup")
async def startup_event():
    """Log when the server starts up and check database connections"""
    global SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT_TEMPLATE
    
    logger.info("=" * 50)
    logger.info("OpenAI-compatible API server starting up")
    logger.info(f"Available models: {AVAILABLE_MODELS}")
    logger.info(f"OpenAI model mappings: {MODEL_MAPPING}")
    
    # Load system prompts
    SYSTEM_PROMPT = load_system_prompt()
    logger.info("System prompt loaded")
    
    # Include the RAG router
    app.include_router(rag_router, prefix="/rag")
    logger.info("RAG router included")
    
    # Initialize intent classifier
    intent_classifier = get_intent_classifier()
    available_intents = intent_classifier.get_available_intents()
    logger.info(f"Intent classifier loaded with {len(available_intents)} intents: {', '.join(available_intents)}")
    
    # Integrate the web interface
    integrate_web_interface(app)
    logger.info("Web interface integrated")
    
    # Update the system prompt with available notes
    SYSTEM_PROMPT = update_system_prompt_with_notes(SYSTEM_PROMPT)
    logger.info("System prompt updated with available notes")
    
    # Update the system prompt with RAG information
    SYSTEM_PROMPT = update_system_prompt_with_rag_info(SYSTEM_PROMPT)
    logger.info("System prompt updated with RAG information")
    
    TOOL_SYSTEM_PROMPT_TEMPLATE = load_tool_system_prompt()
    logger.info("Tool system prompt template loaded")
    
    load_conversation_history()
    logger.info("Conversation history loaded")
    
    asyncio.create_task(periodic_save_history())
    logger.info("Started periodic conversation history saving")
    
    # Check Redis connection
    if message_store.ping():
        logger.info("Connected to Redis database successfully")
    else:
        logger.warning("Failed to connect to Redis database - functionality may be limited")
    
    # Integrate Neo4j with the server
    integrate_neo4j_with_server(app)
    logger.info("Neo4j module integrated with server")
    
    # Integrate RAG with the server
    integrate_rag_with_server(app, AVAILABLE_TOOLS, TOOL_DEFINITIONS)
    logger.info("RAG module integrated with server")
    
    # Integrate timers and reminders with the server
    integrate_timer_reminder_tools(app, AVAILABLE_TOOLS, TOOL_DEFINITIONS)
    logger.info("Timer and reminder module integrated with server")
    
    logger.info("Server ready to accept connections")
    logger.info("=" * 50)

    # Check Neo4j connection
    neo4j_status = check_neo4j_connection()
    logger.info(f"Neo4j connection status: {neo4j_status}")

@app.on_event("shutdown")
async def shutdown_event():
    """Save conversation history when shutting down"""
    logger.info("Server shutting down, saving conversation history...")
    save_conversation_history()
    logger.info("Shutdown complete")


@app.get("/v1/conversations")
async def list_conversations(limit: int = 20, offset: int = 0):
    """List all conversations with additional metadata"""
    try:
        conversations = message_store.list_conversations(limit=limit, offset=offset)
        
        # Add additional formatting if needed
        for conv in conversations:
            # Truncate long messages for preview
            if "last_message" in conv and conv["last_message"]:
                conv["last_message"] = conv["last_message"][:50] + "..." if len(conv["last_message"]) > 50 else conv["last_message"]
        
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        # Use the client's get_messages_by_conversation method
        messages = message_store.get_messages_by_conversation(conversation_id)
        return {"conversation_id": conversation_id, "messages": messages}
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.put("/v1/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, data: dict = Body(...)):
    """Update conversation metadata"""
    try:
        if "title" in data:
            success = message_store.update_conversation_title(conversation_id, data["title"])
            if success:
                return {"status": "updated", "conversation_id": conversation_id}
            else:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            raise HTTPException(status_code=400, detail="Missing required field: title")
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    try:
        success = message_store.delete_conversation(conversation_id)
        if success:
            return {"status": "deleted", "conversation_id": conversation_id}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        messages = message_store.get_messages_by_conversation(conversation_id)
        return {"conversation_id": conversation_id, "messages": messages}
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all exceptions"""
    logger.error(f"Global exception handler caught: {str(exc)}")
    logger.exception(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Log HTTP exceptions"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.post("/v1/memory/search")
async def search_memories(request: dict = Body(...)):
    """Search for memories based on a query"""
    query = request.get("query", "")
    limit = request.get("limit", 5)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Get embedding
    embedding = get_embeddings(query)['embeddings']
    
    # Search
    results = message_store.vector_search(embedding, k=limit)
    
    return {"memories": results}

@app.get("/v1/memory/recent")
async def get_recent_memories(days: int = 7, limit: int = 5):
    """Get recent conversation summaries"""
    return {"recent_conversations": get_recent_conversations(days, limit)}

@app.get("/v1/memory/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get the user profile information"""
    profile = read_note(f"user_profile")
    if "error" in profile:
        return {"profile": "No profile information available"}
    return {"profile": profile}

@app.get("/chat", response_class=FileResponse)
async def chat_interface():
    """Serve the chat interface"""
    chat_html_path = "/home/david/Sara/static/chat.html"
    return FileResponse(chat_html_path)


@app.get("/")
async def redirect_to_chat():
    """Redirect root to chat interface"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat")

# Add this to your server.py - a new endpoint for the chat interface
@app.get("/v1/chat/current-session")
async def get_current_session():
    """Return just the current chat session data"""
    return {
        "conversation_id": "current_session",
        "messages": current_chat  # Using your existing current_chat variable
    }

def add_message_to_history(role, content):
    """
    Add a message to the in-memory conversation history
    
    Args:
        role (str): 'user' or 'assistant'
        content (str): Message content
    """
    global MESSAGE_HISTORY
    
    # Skip tool messages
    if role == 'tool':
        return
    
    # Add new message
    MESSAGE_HISTORY.append({'role': role, 'content': content})
    
    # If we exceed maximum history, remove the oldest message
    if len(MESSAGE_HISTORY) > MAX_HISTORY:
        MESSAGE_HISTORY.pop(0)  # Remove oldest message
        logger.info("Removed oldest message from conversation history")

def get_messages_with_system_prompt():
    """
    Get the full message list with system prompt for the model, with token counts
    
    Returns:
        list: Complete message list for sending to model
        dict: Token statistics
    """
    global MESSAGE_HISTORY
    
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create system message with current date
    system_prompt = f"{SYSTEM_PROMPT}\n\nCurrent date and time: {current_datetime}"
    system_tokens = count_tokens(system_prompt)
    
    # Create system message
    system_message = {
        'role': 'system', 
        'content': system_prompt,
        'tokens': system_tokens
    }
    
    # Combine system message with conversation history
    messages = [system_message] + MESSAGE_HISTORY
    
    # Calculate token statistics
    history_tokens = sum(msg.get('tokens', 0) for msg in MESSAGE_HISTORY)
    total_tokens = system_tokens + history_tokens
    
    token_stats = {
        'system_tokens': system_tokens,
        'history_tokens': history_tokens,
        'total_tokens': total_tokens
    }
    
    # Log token usage
    logger.info(f"Token usage - System: {system_tokens}, History: {history_tokens}, Total: {total_tokens}")
    
    # Store token stats in Redis for dashboard
    try:
        token_stats_json = json.dumps(token_stats)
        
        # Use a more careful approach to avoid attribute errors
        try:
            # Try different possible Redis client locations
            if hasattr(message_store, 'client') and hasattr(message_store.client, 'redis_client'):
                message_store.client.redis_client.set("token_stats", token_stats_json)
            elif hasattr(message_store, 'message_store'):
                message_store.message_store.set("token_stats", token_stats_json)
            elif hasattr(message_store, 'redis_client'):
                message_store.redis_client.set("token_stats", token_stats_json)
            else:
                # Skip Redis storage with a warning
                logger.warning("Couldn't find appropriate Redis client attribute in message_store")
        except Exception as e:
            logger.warning(f"Failed to store token stats in Redis: {str(e)}")
    except Exception as e:
        logger.error(f"Error preparing token stats for Redis: {str(e)}")
    
    # Return the messages without token counts for the model
    clean_messages = []
    for msg in messages:
        clean_msg = {'role': msg['role'], 'content': msg['content']}
        clean_messages.append(clean_msg)
    
    return clean_messages, token_stats


def clear_conversation_history():
    """Reset the conversation history"""
    global MESSAGE_HISTORY
    MESSAGE_HISTORY = []

@app.get("/")
async def root():
    """Root endpoint to help diagnose connection issues"""
    logger.info("Root endpoint called")
    return {
        "status": "online",
        "server": "OpenAI-compatible API",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "legacy_chat": "/api/chat",
            "health": "/health"
        },
        "timestamp": datetime.now().isoformat()
    }

def get_recent_conversations(days=7, limit=5):
    """Get summaries of recent conversations within the specified time period"""
    today = datetime.now()
    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    
    query = f"@conversation_id:summary-* @date:[{start_date} +inf]"
    results = message_store.message_store.ft("message_idx").search(
        query,
        sort_by="date",
        sortby_desc=True,
        limit=0, limit_num=limit
    )
    
    summaries = []
    for doc in results.docs:
        message = json.loads(doc.json)
        summaries.append({
            "date": message["date"],
            "summary": message["content"]
        })
    
    return summaries

async def periodic_save_history():
    """Periodically save conversation history to ensure it's not lost"""
    while True:
        try:
            # Wait for 5 minutes
            await asyncio.sleep(300)
            
            # Save the conversation history
            save_conversation_history()
        except Exception as e:
            logger.error(f"Error in periodic history save: {str(e)}")

async def classify_intent_and_select_tools(user_query):
    """
    Classify the user's intent and select appropriate tools.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        tuple: (prediction_result, selected_tools)
    """
    try:
        # Get the intent classifier
        intent_classifier = get_intent_classifier()
        
        # Predict intent
        prediction_result = intent_classifier.predict_intent(user_query)
        
        if "error" in prediction_result:
            logger.error(f"Error predicting intent: {prediction_result['error']}")
            return prediction_result, None
        
        # Log the prediction
        if prediction_result["predictions"]:
            top_prediction = prediction_result["predictions"][0]
            logger.info(f"Top intent for query '{user_query}': {top_prediction['intent']} ({top_prediction['percentage']:.2f}%)")
        
        # Select tools based on the intent
        selected_tools = get_tools_for_intent(prediction_result, TOOL_DEFINITIONS)
        
        return prediction_result, selected_tools
    
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return {"error": str(e), "predictions": []}, None

def load_system_prompt():
    """Load the system prompt from a file"""
    prompt_file = "/home/david/Sara/system_prompt.txt"
    
    try:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                system_prompt = f.read().strip()
                logger.info(f"Loaded system prompt from file ({len(system_prompt)} chars)")
                
                # Load core memories and append to system prompt if they exist
                core_memories = load_core_memories()
                if core_memories:
                    system_prompt += f"\n\n Core Memories:\n{core_memories}"
                    logger.info(f"Added {len(core_memories)} chars of core memories to system prompt")
                
                return system_prompt
        else:
            # Fallback prompt if file doesn't exist
            default_prompt = "You are a personal assistant named Sara. You are witty, flirty and always ready to learn and chat."
            logger.warning(f"System prompt file not found at {prompt_file}, using default")
            return default_prompt
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        # Return a basic default if there's an error
        return "You are a personal assistant named Sara. You are witty, flirty and always ready to learn and chat."

def load_tool_system_prompt():
    """Load the tool system prompt template from a file"""
    prompt_file = "/home/david/Sara/tool_system_prompt.txt"
    
    try:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                template = f.read().strip()
                logger.info(f"Loaded tool system prompt template from file ({len(template)} chars)")
                return template
        else:
            # Fallback template if file doesn't exist
            default_template = (
                "You are continuing a conversation with the user. You previously decided to use a tool to help answer their question.\n\n"
                "User's original query: \"{{user_query}}\"\n\n"
                "Information obtained from tool(s):\n{{tool_results}}\n\n"
                "Use this information to craft a helpful, direct response that addresses the user's question."
            )
            logger.warning(f"Tool system prompt file not found at {prompt_file}, using default")
            return default_template
    except Exception as e:
        logger.error(f"Error loading tool system prompt: {e}")
        # Return a basic default if there's an error
        return "The following is information obtained by the tool call, use this in your response: {{tool_results}}"


def load_core_memories() -> str:
    """Load core memories from file and return as a formatted string."""
    try:
        if os.path.exists(CORE_MEMORY_FILE):
            with open(CORE_MEMORY_FILE, 'r') as f:
                content = f.read().strip()
                logger.info(f"Loaded core memories ({len(content)} chars)")
                return content
        else:
            # Create an empty file if it doesn't exist
            with open(CORE_MEMORY_FILE, 'w') as f:
                f.write("")
            logger.info("Created new empty core memories file")
            return ""
    except Exception as e:
        logger.error(f"Error loading core memories: {e}")
        return ""

def append_core_memory(memory: str) -> Dict[str, Any]:
    """Append a new memory to the core memories file."""
    try:
        # First load existing memories to validate
        existing_memories = load_core_memories()
        
        # Check if this memory already exists (to avoid duplicates)
        if memory in existing_memories:
            return {
                "success": False,
                "message": "This memory already exists in core memories.",
                "current_size": len(existing_memories)
            }
        
        # Append the new memory
        with open(CORE_MEMORY_FILE, 'a') as f:
            # Add a newline before the memory if file is not empty
            if existing_memories:
                f.write(f"\n- {memory}")
            else:
                f.write(f"- {memory}")
        
        # Get updated file size
        updated_memories = load_core_memories()
        
        logger.info(f"Added new core memory: {memory[:50]}...")
        return {
            "success": True,
            "message": "Memory added successfully.",
            "current_size": len(updated_memories)
        }
    except Exception as e:
        logger.error(f"Error appending core memory: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "current_size": 0
        }
    
def rewrite_core_memories(memories: List[str]) -> Dict[str, Any]:
    """Rewrite the entire core memories file with the provided list of memories."""
    try:
        # Format the memories
        formatted_memories = "\n".join([f"- {memory}" for memory in memories])
        
        # Write to file
        with open(CORE_MEMORY_FILE, 'w') as f:
            f.write(formatted_memories)
        
        logger.info(f"Rewrote core memories with {len(memories)} items")
        return {
            "success": True,
            "message": f"Successfully rewrote core memories with {len(memories)} items.",
            "current_size": len(formatted_memories)
        }
    except Exception as e:
        logger.error(f"Error rewriting core memories: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "current_size": 0
        }
    
def get_core_memory_stats() -> Dict[str, Any]:
    """Get statistics about the core memories file."""
    try:
        content = load_core_memories()
        memories = [m.strip() for m in content.split('\n') if m.strip()]
        
        return {
            "total_memories": len(memories),
            "total_chars": len(content),
            "sample": memories[:3] if memories else []
        }
    except Exception as e:
        logger.error(f"Error getting core memory stats: {e}")
        return {
            "total_memories": 0,
            "total_chars": 0,
            "sample": []
        }

class ConversationBuffer:
    def __init__(self, max_buffer_size=15):
        self.buffer = {}  # Dictionary of conversation_id -> list of messages
        self.max_buffer_size = max_buffer_size
        self.summarization_tasks = set()  # Track ongoing summarization tasks
    
    def add_message(self, conversation_id, message):
        """Add a message to the buffer for a specific conversation"""
        if conversation_id not in self.buffer:
            self.buffer[conversation_id] = []
        
        self.buffer[conversation_id].append(message)
        
        # Check if we need to summarize
        if len(self.buffer[conversation_id]) >= self.max_buffer_size:
            return True
        return False
    
    def get_messages(self, conversation_id):
        """Get all messages in the buffer for a conversation"""
        return self.buffer.get(conversation_id, [])
    
    def clear_buffer(self, conversation_id):
        """Clear the buffer for a conversation after summarization"""
        if conversation_id in self.buffer:
            self.buffer[conversation_id] = []
    
    def is_summarization_in_progress(self, conversation_id):
        """Check if summarization is already in progress"""
        return conversation_id in self.summarization_tasks
    
    def mark_summarization_started(self, conversation_id):
        """Mark that summarization has started for this conversation"""
        self.summarization_tasks.add(conversation_id)
    
    def mark_summarization_completed(self, conversation_id):
        """Mark that summarization has completed for this conversation"""
        if conversation_id in self.summarization_tasks:
            self.summarization_tasks.remove(conversation_id)


# Initialize the conversation buffer
conversation_buffer = ConversationBuffer(max_buffer_size=10)







def list_notes() -> List[Dict[str, Any]]:
    """List all available notes with metadata"""
    try:
        notes = []
        for filename in os.listdir(NOTES_DIRECTORY):
            if filename.endswith('.json'):
                file_path = os.path.join(NOTES_DIRECTORY, filename)
                with open(file_path, 'r') as f:
                    note_data = json.load(f)
                    note_data['filename'] = filename
                    notes.append(note_data)
        
        # Sort notes by last modified time, newest first
        notes.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        return notes
    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        return []

def get_note_names_for_prompt() -> str:
    """Get a formatted list of note names for the system prompt"""
    try:
        notes = list_notes()
        if not notes:
            return "No notes available."
        
        note_list = ["Available notes:"]
        for note in notes:
            title = note.get('title', 'Untitled note')
            created = note.get('created', 'Unknown date')
            note_list.append(f"- {title} (created: {created})")
        
        return "\n".join(note_list)
    except Exception as e:
        logger.error(f"Error getting note names for prompt: {e}")
        return "Error retrieving notes."

def create_note(title: str, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new note"""
    try:
        # Generate a filename based on the title
        safe_title = "".join(c if c.isalnum() else "_" for c in title).lower()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{safe_title}_{timestamp}.json"
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Create the note data
        now = datetime.now().isoformat()
        note_data = {
            "title": title,
            "content": content,
            "tags": tags or [],
            "created": now,
            "last_modified": now
        }
        
        # Write the note to file
        with open(file_path, 'w') as f:
            json.dump(note_data, f, indent=2)
        
        logger.info(f"Created new note: {title} ({filename})")
        
        # Return the note data with the filename
        note_data['filename'] = filename
        return note_data
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        return {"error": str(e)}

async def summarize_conversation(conversation_id):
    """Summarize a conversation and store it as a higher-level memory"""
    # Get all messages for this conversation
    messages = message_store.get_messages_by_conversation(conversation_id)
    
    # Format the conversation for summarization
    formatted_convo = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Prepare system prompt for summarization
    system_prompt = "Summarize the key points of this conversation in 2-3 sentences. Focus on what information was shared and what was learned about the user."
    
    # Create summarization request
    summary_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Here's the conversation to summarize:\n\n{formatted_convo}"}
    ]
    
    # Get summary
    response = await fetch_ollama_response(summary_messages, "llama3.1", stream=False)
    if 'message' in response and 'content' in response['message']:
        summary = response['message']['content']
        
        # Store this summary with a special tag for easier retrieval
        today = datetime.now()
        message_store.store_message(
            role='system',
            content=summary,
            conversation_id=f"summary-{conversation_id}",
            date=today,
            embedding=get_embeddings(summary)['embeddings']
        )
        
        return summary
    return None

async def read_note(identifier: str) -> Dict[str, Any]:
    """Read a note by title or filename"""
    try:
        # Properly await the coroutine
        notes = await list_notes()
        
        # Try to find the note by exact title match first
        for note in notes:
            if note.get('title') == identifier:
                logger.info(f"Read note by title: {identifier}")
                return note
        
        # Then try to find by filename
        for note in notes:
            if note.get('filename') == identifier:
                logger.info(f"Read note by filename: {identifier}")
                return note
        
        # Lastly, try fuzzy matching on title
        for note in notes:
            if identifier.lower() in note.get('title', '').lower():
                logger.info(f"Read note by fuzzy match: {identifier} -> {note.get('title')}")
                return note
        
        logger.warning(f"Note not found: {identifier}")
        return {"error": f"Note not found: {identifier}"}
    except Exception as e:
        logger.error(f"Error reading note: {e}")
        return {"error": str(e)}

def append_note(identifier: str, content: str) -> Dict[str, Any]:
    """Append content to an existing note"""
    try:
        # First, find the note
        note = read_note(identifier)
        if "error" in note:
            return note
        
        # Get the file path
        filename = note.get('filename')
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Append content and update timestamp
        note['content'] = note.get('content', '') + '\n\n' + content
        note['last_modified'] = datetime.now().isoformat()
        
        # Write the updated note
        with open(file_path, 'w') as f:
            json.dump(note, f, indent=2)
        
        logger.info(f"Appended to note: {note.get('title')} ({filename})")
        return note
    except Exception as e:
        logger.error(f"Error appending to note: {e}")
        return {"error": str(e)}

def delete_note(identifier: str) -> Dict[str, Any]:
    """Delete a note by title or filename"""
    try:
        # First, find the note
        note = read_note(identifier)
        if "error" in note:
            return note
        
        # Get the file path
        filename = note.get('filename')
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Delete the file
        os.remove(file_path)
        
        logger.info(f"Deleted note: {note.get('title')} ({filename})")
        return {"success": True, "message": f"Note '{note.get('title')}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        return {"error": str(e)}

def update_system_prompt_with_notes(original_prompt: str) -> str:
    """Update the system prompt to include available notes"""
    # Use a synchronous version of get_note_names_for_prompt
    notes_info = get_note_names_for_prompt_sync()
    
    # Check if the prompt already has notes information
    if "Available notes:" in original_prompt:
        # Replace the existing notes section
        lines = original_prompt.split('\n')
        new_lines = []
        skip_mode = False
        
        for line in lines:
            if line.strip() == "Available notes:":
                new_lines.append(notes_info)
                skip_mode = True
            elif skip_mode and line.strip().startswith('-'):
                continue  # Skip existing note entries
            else:
                skip_mode = False
                new_lines.append(line)
        
        return '\n'.join(new_lines)
    else:
        # Append notes information at the end
        return f"{original_prompt}\n\n{notes_info}"

def get_note_names_for_prompt_sync() -> str:
    """Get a formatted list of note names for the system prompt (synchronous version)"""
    try:
        # Make sure we're using the synchronous version, not the async one
        notes = []
        for filename in os.listdir(NOTES_DIRECTORY):
            if filename.endswith('.json'):
                file_path = os.path.join(NOTES_DIRECTORY, filename)
                with open(file_path, 'r') as f:
                    note_data = json.load(f)
                    note_data['filename'] = filename
                    notes.append(note_data)
        
        if not notes:
            return "No notes available."
        
        note_list = ["Available notes:"]
        for note in notes:
            title = note.get('title', 'Untitled note')
            created = note.get('created', 'Unknown date')
            note_list.append(f"- {title} (created: {created})")
        
        return "\n".join(note_list)
    except Exception as e:
        logger.error(f"Error getting note names for prompt: {e}")
        return "Error retrieving notes."


def format_tool_system_prompt(template, user_query, tool_results):
    """Replace template variables with actual values"""
    prompt = template.replace("{{user_query}}", user_query)
    prompt = prompt.replace("{{tool_results}}", tool_results)
    return prompt

async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = datetime.now()
    
    # Get request details
    method = request.method
    url = request.url
    client = request.client.host if request.client else "unknown"
    
    # Log the incoming request
    logger.info(f"Request received: {method} {url} from {client}")
    
    # Process the request
    try:
        response = await call_next(request)
        
        # Log the response
        process_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed: {method} {url} - Status: {response.status_code} - Time: {process_time:.3f}s")
        
        return response
    except Exception as e:
        # Log any exceptions
        logger.error(f"Request failed: {method} {url} - Error: {str(e)}")
        raise

async def handle_request_based_on_intent(messages, model, user_query, prediction_result, stream=True):
    """
    Handle the API request differently based on the classified intent.
    
    Args:
        messages (list): The messages to send to the model
        model (str): The model name
        user_query (str): The user's query
        prediction_result (dict): The intent classification result
        stream (bool): Whether to stream the response
        
    Returns:
        async generator: A generator that yields formatted response chunks
    """
    # Map OpenAI model to local model
    local_model = MODEL_MAPPING.get(model, MODEL_MAPPING["default"])
    
    # Get the URL for this model
    url = MODEL_URLS.get(local_model, MODEL_URLS["default"])
    
    # Determine if we should skip tools based on intent
    skip_tools = False
    if prediction_result and "predictions" in prediction_result and prediction_result["predictions"]:
        top_intent = prediction_result["predictions"][0]["intent"]
        confidence = prediction_result["predictions"][0]["probability"]
        
        # Skip tools for "none" or "thinking" intents or low confidence
        if top_intent in ["none", "thinking"] or confidence < INTENT_CONFIDENCE_THRESHOLDS.get(top_intent, 0.3):
            skip_tools = True
            logger.info(f"Skipping tools for intent '{top_intent}' with confidence {confidence:.4f}")

    async with aiohttp.ClientSession() as session:
        # Prepare request data
        data = {
            'model': local_model,
            'messages': messages,
            'stream': True  # Always stream in this function
        }
        
        # Add tools only if not skipping
        if not skip_tools:
            # Get selected tools based on intent
            selected_tools = get_tools_for_intent(prediction_result, TOOL_DEFINITIONS)
            
            if selected_tools:
                logger.info(f"Adding {len(selected_tools)} tools to request based on intent")
                data['tools'] = selected_tools
            else:
                # Use default tools if no specific tools selected
                logger.info(f"Adding default tools to request")
                data['tools'] = TOOL_DEFINITIONS
        else:
            logger.info("No tools included in request based on intent")
        
        logger.info(f"Sending request to {url} with model {local_model}, skip_tools={skip_tools}")
        
        try:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                
                # For streaming responses
                full_response = ""
                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    chunk_str = chunk.decode('utf-8').strip()
                    if not chunk_str:
                        continue
                        
                    try:
                        response_data = json.loads(chunk_str)
                        
                        # Check if this is a tool call and we're not in skip_tools mode
                        if not skip_tools and 'message' in response_data and 'tool_calls' in response_data['message']:
                            logger.info(f"Tool call detected in response")
                            
                            # Process tool calls
                            tool_outputs = await process_tool_calls(response_data)
                            
                            if tool_outputs:
                                logger.info(f"Processed tool outputs: {tool_outputs}")
                                
                                # Format tool results for system prompt
                                tool_results = ""
                                for output in tool_outputs:
                                    if output.get('role') == 'tool' and 'content' in output:
                                        tool_name = output.get('name', 'unknown_tool')
                                        tool_results += f"Tool: {tool_name}\n"
                                        tool_results += f"Result: {output['content']}\n\n"
                                
                                # Create system prompt with tool results
                                system_prompt = format_tool_system_prompt(
                                    TOOL_SYSTEM_PROMPT_TEMPLATE,
                                    user_query,
                                    tool_results
                                )
                                
                                # Create follow-up messages
                                follow_up_messages = [
                                    {'role': 'system', 'content': system_prompt}
                                ]
                                
                                # Send a follow-up request with the tool results
                                follow_up_data = {
                                    'model': local_model,
                                    'messages': follow_up_messages,
                                    'stream': True
                                }
                                
                                async with session.post(url, json=follow_up_data) as follow_up_response:
                                    follow_up_response.raise_for_status()
                                    async for follow_up_chunk in follow_up_response.content:
                                        if not follow_up_chunk:
                                            continue
                                        
                                        follow_up_str = follow_up_chunk.decode('utf-8').strip()
                                        if follow_up_str:
                                            yield format_sse_message(follow_up_str)
                                            
                                            # Extract content for full response
                                            try:
                                                follow_up_data = json.loads(follow_up_str)
                                                if 'message' in follow_up_data and 'content' in follow_up_data['message']:
                                                    full_response += follow_up_data['message']['content']
                                            except:
                                                pass
                            else:
                                # No tool outputs, just yield the original response
                                yield format_sse_message(chunk_str)
                                
                                # Update full response
                                if 'message' in response_data and 'content' in response_data['message']:
                                    full_response += response_data['message']['content']
                        else:
                            # Regular message (or we're in skip_tools mode)
                            yield format_sse_message(chunk_str)
                            
                            # Update full response
                            if 'message' in response_data and 'content' in response_data['message']:
                                full_response += response_data['message']['content']
                    except json.JSONDecodeError:
                        logger.warning(f"JSON decode error for chunk: {chunk_str}")
                        continue
                
                # Add the full response to history
                if full_response:
                    add_message_to_history('assistant', full_response)
                    
                    # Store in Redis for vector search
                    if message_store:
                        try:
                            embed_and_save(full_response, "primary_conversation", "assistant")
                        except Exception as e:
                            logger.error(f"Error storing response in Redis: {str(e)}")
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg)
            yield format_sse_message(json.dumps({"error": error_msg}))

async def retrieve_relevant_memories(user_query, conversation_id=None, k=5):
    try:
        # Replace this:
        # query_embedding = text_to_embedding_ollama(user_query)
        
        # With this:
        query_embedding = get_embeddings(user_query)
        
        # Check if vector_search method exists
        if hasattr(message_store, 'vector_search'):
            # Replace any references to query_embedding['embeddings'] with the same format
            similar_messages = message_store.vector_search(query_embedding['embeddings'], k=k+5)
            
            # Filter out messages from the current conversation if provided
            filtered_messages = []
            if conversation_id:
                # Get the current conversation messages
                current_convo_messages = message_store.get_messages_by_conversation(conversation_id)
                
                # Extract content from current conversation to compare
                current_convo_contents = set()
                for msg in current_convo_messages:
                    content = msg.get('content', '')
                    if content:
                        # Add a simplified version of content for comparison (first 100 chars)
                        current_convo_contents.add(content[:100].lower().strip())
                
                # Filter out messages that are part of the current conversation
                for msg in similar_messages:
                    content = msg.get('content', '')
                    # Skip if content is empty
                    if not content:
                        continue
                        
                    # Check if this content appears to match anything in the current conversation
                    simplified_content = content[:100].lower().strip()
                    if simplified_content not in current_convo_contents:
                        filtered_messages.append(msg)
                    
                    # Break if we have enough filtered messages
                    if len(filtered_messages) >= k:
                        break
            else:
                # No conversation ID provided, use all results
                filtered_messages = similar_messages[:k]
            
            # Format into conversational context
            if filtered_messages:
                memory_context = "Relevant past conversations:\n\n"
                for i, msg in enumerate(filtered_messages):
                    memory_context += f"Memory {i+1}:\n"
                    memory_context += f"Role: {msg.get('role', 'unknown')}\n"
                    memory_context += f"Content: {msg.get('content', '')}\n\n"
                return memory_context
        else:
            logger.warning("vector_search method not available in RedisClient")
        
        return ""
    except Exception as e:
        logger.error(f"Error retrieving relevant memories: {str(e)}")
        return ""



# Define tool functions
def send_message(message: str, user: str = "default") -> str:
    """Send a message to a user"""
    print(f"\nSending message to user {user}: {message}")
    return message

def search_perplexica(query: str, focus_mode: str = "webSearch", optimization_mode: str = "balanced") -> str:
    """Search using Perplexica"""
    print(f"\nSearching Perplexica with query: {query}")
    try:
        result = perplexica.search(
            query=query,
            focus_mode=focus_mode,
            optimization_mode=optimization_mode
        )
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

# Define available tools
AVAILABLE_TOOLS: Dict[str, Callable] = {
    'send_message': send_message,
    'search_perplexica': search_perplexica,
    'append_core_memory': append_core_memory,
    'rewrite_core_memories': rewrite_core_memories,
    'create_note': create_note,
    'read_note': read_note,
    'append_note': append_note,
    'delete_note': delete_note,
    'list_notes': list_notes,
}

# Define tool schemas that match OpenAI's format
TOOL_DEFINITIONS = [
        {
        'type': 'function',
        'function': {
            'name': 'send_message',
            'description': 'Use this tool to think about what the user said and the conversation context before formulating your response. An example would be "the user is saying good morning, I should respond in a cute funny manner" or "the user seems to be struggling with something, I should try and cheer him up". ',
            'parameters': {
                'type': 'object',
                'required': ['message'],
                'properties': {
                    'message': {
                        'type': 'string', 
                        'description': 'Your thinking process about how to respond to the user'
                    },
                    'user': {
                        'type': 'string',
                        'description': 'The ID of the user to send the message to (optional)',
                        'default': 'default'
                    },
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_perplexica',
            'description': 'Search the web and other sources using Perplexica. Use this when you need to find information.',
            'parameters': {
                'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query'
                    },
                    'focus_mode': {
                        'type': 'string',
                        'description': 'The type of search to perform',
                        'enum': ['webSearch', 'academicSearch', 'writingAssistant', 
                                'wolframAlphaSearch', 'youtubeSearch', 'redditSearch'],
                        'default': 'webSearch'
                    },
                    'optimization_mode': {
                        'type': 'string',
                        'description': 'Whether to optimize for speed or balanced results',
                        'enum': ['speed', 'balanced'],
                        'default': 'balanced'
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'append_core_memory',
            'description': 'Add a new important memory to the assistant\'s core memories. Use this when the user shares something important that you think should be saved for future reference. For example, "The users favorite book series is the KingKiller Chronicle", "The user loves eating pizza."',
            'parameters': {
                'type': 'object',
                'required': ['memory'],
                'properties': {
                    'memory': {
                        'type': 'string',
                        'description': 'The important information to remember (keep it concise and focused on a single fact or detail)'
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'rewrite_core_memories',
            'description': 'Rewrite all core memories. Use this to reorganize, consolidate, or update the entire set of memories.',
            'parameters': {
                'type': 'object',
                'required': ['memories'],
                'properties': {
                    'memories': {
                        'type': 'array',
                        'description': 'The complete list of memories to save',
                        'items': {
                            'type': 'string'
                        }
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'create_note',
            'description': 'Create a new note with a title and content. Use this for saving important information that needs to be referenced later.',
            'parameters': {
                'type': 'object',
                'required': ['title', 'content'],
                'properties': {
                    'title': {
                        'type': 'string',
                        'description': 'A descriptive title for the note'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'The content of the note'
                    },
                    'tags': {
                        'type': 'array',
                        'description': 'Optional tags to categorize the note',
                        'items': {
                            'type': 'string'
                        }
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_note',
            'description': 'Read a note by its title or filename. Use this to retrieve information from a previously saved note.',
            'parameters': {
                'type': 'object',
                'required': ['identifier'],
                'properties': {
                    'identifier': {
                        'type': 'string',
                        'description': 'The title or filename of the note to read'
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'append_note',
            'description': 'Append content to an existing note. Use this to add information to a note without replacing its existing content.',
            'parameters': {
                'type': 'object',
                'required': ['identifier', 'content'],
                'properties': {
                    'identifier': {
                        'type': 'string',
                        'description': 'The title or filename of the note to append to'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'The content to append to the note'
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'delete_note',
            'description': 'Delete a note by its title or filename.',
            'parameters': {
                'type': 'object',
                'required': ['identifier'],
                'properties': {
                    'identifier': {
                        'type': 'string',
                        'description': 'The title or filename of the note to delete'
                    }
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_notes',
            'description': 'List all available notes with their metadata.',
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    },
]


# Conversation history
current_chat = []




# File operations
def write_to_file(role, content, dates):
    try:
        with open(f"/home/david/Sara/logs/{dates}.txt",'x+') as f:
            pass
    except FileExistsError:
        pass
    with open(f"/home/david/Sara/logs/{dates}.txt", 'a+') as f:
        f.write(f"{role}:\n")
        f.write(content + "\n")

# Embedding operations
def get_embeddings(text, model_name="bge-m3"):
    """
    Generate embeddings for a text using Ollama models.
    
    Args:
        text (str): The text to generate embeddings for
        model_name (str): The name of the embedding model to use
        
    Returns:
        dict: A dictionary containing the embedding vectors and metadata
        
    Example:
        result = get_embeddings("This is a sample text")
        embeddings = result['embeddings']
    """
    try:
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text provided for embedding: {type(text)}")
            return {"embeddings": [], "error": "Invalid input text"}
        
        # Clean/prepare text if needed
        cleaned_text = text.strip()
        
        # Generate embeddings using Ollama
        response = ollama.embed(model=model_name, input=cleaned_text)
        
        # Log success if debug logging is enabled
        if logger.isEnabledFor(logging.DEBUG):
            embedding_length = len(response.get('embedding', [])) 
            logger.debug(f"Generated embedding with {embedding_length} dimensions")
        
        return response
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return {"embeddings": [], "error": str(e)}

def query_vector_database(query_vectors, k=3):
    try:
        results = message_store.vector_search(query_vectors, k=k)
        # Extract content from results
        content_list = [item.get('content', '') for item in results if item.get('content')]
        return content_list
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        return []

def embed_and_save(content, conversation_id, role="assistant"):
    if not content or not content.strip():
        return None
        
    today = datetime.now()
    
    try:
        # Replace this:
        # embeddings = text_to_embedding_ollama(content)
        # embedding = embeddings.get('embeddings')
        
        # With this:
        embedding_result = get_embeddings(content)
        embedding = embedding_result.get('embeddings')
        
        if embedding:
            # Store in Redis
            message_id = message_store.store_message(
                role=role,
                content=content,
                conversation_id=conversation_id,
                date=today,
                embedding=embedding
            )
            
            # Still write to file for backup
            formatted_date = today.strftime("%m-%d-%Y")
            write_to_file(role.capitalize(), content, formatted_date)
            
            # Update current chat (if needed)
            current_chat.append({'role': role, 'content': content})
            
            return embedding
    except Exception as e:
        logger.error(f"Error in embed_and_save: {e}")
        return None

async def async_embed_and_save(content, id):
    await asyncio.to_thread(embed_and_save, content, id)

# Tool processing
async def process_tool_calls(response_data: dict) -> dict:
    """Process any tool calls in the response and return the tool outputs"""
    logger.info("Processing tool calls in response")
    
    if not isinstance(response_data, dict):
        logger.warning("Response data is not a dictionary")
        return None
        
    message = response_data.get('message', {})
    if 'tool_calls' not in message:
        logger.debug("No tool calls found in message")
        return None
        
    logger.info(f"Found tool calls: {message['tool_calls']}")
    
    tool_outputs = []
    for tool_call in message['tool_calls']:
        function_name = tool_call['function']['name']
        if function_name not in AVAILABLE_TOOLS:
            logger.warning(f"Tool {function_name} not available in AVAILABLE_TOOLS")
            continue
            
        function = AVAILABLE_TOOLS.get(function_name)
        try:
            arguments = tool_call['function']['arguments']
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing arguments for {function_name}: {arguments}")
                    continue
            
            logger.info(f"Executing tool {function_name} with arguments: {arguments}")
            
            # Check if the function is a coroutine function (async)
            if asyncio.iscoroutinefunction(function):
                # If it's async, await it
                result = await function(**arguments)
            else:
                # If it's not async, call it normally
                result = function(**arguments)
                
            logger.info(f"Tool {function_name} executed successfully")
            
            # Store the tool output with standard role/name format
            tool_outputs.append({
                'role': 'tool',
                'name': function_name,
                'content': str(result)
            })
        except Exception as e:
            error_msg = f"Error executing tool {function_name}: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
    
    return tool_outputs

# API request processing
async def send_to_ollama_api(messages, model, tools=None, stream=True, user_query=None, is_follow_up=False):
    """Streaming API function - use only for streaming responses"""
    async with aiohttp.ClientSession() as session:
        # Map OpenAI model to local model
        local_model = MODEL_MAPPING.get(model, MODEL_MAPPING["default"])
        
        # Prepare request data
        data = {
            'model': local_model,
            'messages': messages,
            'stream': True  # Always stream in this function
        }
        
        # Always include tools if available
        if tools:
            logger.info(f"Adding {len(tools)} tools to streaming request")
            data['tools'] = tools
        else:
            # Always include default tools for consistent behavior
            logger.info(f"Adding default tools to streaming request")
            data['tools'] = TOOL_DEFINITIONS
        try:
            # Make sure tools are properly serializable
            json.dumps(data['tools'])  # This will raise an error if there's an issue
        except Exception as e:
            logger.error(f"Error in request data formatting: {str(e)}")
            # Try with simplified tools if there's an error
            if 'tools' in data:
                data['tools'] = data['tools'][:3]  # Just use the first 3 tools as a test

        # Get the appropriate URL based on the model
        url = MODEL_URLS.get(local_model, MODEL_URLS["default"])
        
        logger.info(f"Sending {'follow-up' if is_follow_up else 'initial'} streaming request to Ollama at {url} with model {local_model}")
        logger.debug(f"Request data: {json.dumps(data)}")
        
        try:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                
                # For streaming responses
                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    chunk_str = chunk.decode('utf-8').strip()
                    if not chunk_str:
                        continue
                        
                    try:
                        response_data = json.loads(chunk_str)
                        logger.debug(f"Received chunk: {chunk_str}")
                        
                        # If this is already a follow-up response, just yield it without further processing
                        if is_follow_up:
                            yield format_sse_message(chunk_str)
                            continue
                        
                        # Handle different types of responses
                        # Case 1: Tool calls in the response
                        if 'message' in response_data and 'tool_calls' in response_data['message']:
                            logger.info(f"Tool call detected in response")
                            
                            # Don't yield the tool call message directly
                            # Instead, process it and make a follow-up request with the results
                            
                            tool_outputs = await process_tool_calls(response_data)
                            
                            if tool_outputs:
                                logger.info(f"Processed tool outputs: {tool_outputs}")
                                
                                # Get the most recent user query if it wasn't passed explicitly
                                original_query = user_query
                                if not original_query:
                                    # Reverse the messages list to find the most recent user message
                                    for msg in reversed(messages):
                                        if msg.get('role') == 'user':
                                            original_query = msg.get('content', '')
                                            logger.info(original_query)
                                            break
                                
                                # Format tool results for the system prompt
                                tool_results = ""
                                for output in tool_outputs:
                                    if output.get('role') == 'tool' and 'content' in output:
                                        tool_name = output.get('tool_call_id', 'unknown_tool')
                                        tool_results += f"Tool: {tool_name}\n"
                                        tool_results += f"Result: {output['content']}\n\n"
                                
                                # Read the tool system prompt template
                                try:
                                    with open('tool_system_prompt.txt', 'r') as f:
                                        system_prompt_template = f.read()
                                    
                                    # Replace placeholders in the template
                                    system_prompt = system_prompt_template.replace("{{user_query}}", original_query)
                                    system_prompt = system_prompt.replace("{{tool_results}}", tool_results)
                                    
                                except Exception as e:
                                    logger.error(f"Error reading tool system prompt: {str(e)}")
                                    system_prompt = f"Continue based on the tool results for query: {original_query}\nTool results: {tool_results}\nDo not repeat the initial response."
                                
                                # Create follow-up messages with the proper system prompt
                                follow_up_messages = [
                                    {'role': 'system', 'content': system_prompt}
                                ]
                                
                                # Send a follow-up request with the tool context, passing is_follow_up=True
                                # This will prevent recursive follow-up requests
                                async for follow_up_chunk in send_to_ollama_api(
                                    follow_up_messages, 
                                    model, 
                                    tools=None, 
                                    stream=True, 
                                    user_query=original_query,
                                    is_follow_up=True  # Mark this as a follow-up request
                                ):
                                    yield follow_up_chunk
                            else:
                                # No tool outputs, so just yield the original response
                                yield format_sse_message(chunk_str)
                        
                        # Case 2: Regular responses (non-tool responses)
                        else:
                            logger.info("Regular message response detected - sending follow-up request for consistency")
                            
                            # Use the same approach as with tool calls for consistency:
                            # Make a new request with the same messages for proper streaming
                            follow_up_data = {
                                'model': local_model,
                                'messages': messages,
                                'stream': True
                            }
                            
                            async with session.post(url, json=follow_up_data) as follow_up_response:
                                follow_up_response.raise_for_status()
                                async for follow_up_chunk in follow_up_response.content:
                                    if not follow_up_chunk:
                                        continue
                                    
                                    follow_up_str = follow_up_chunk.decode('utf-8').strip()
                                    if follow_up_str:
                                        yield format_sse_message(follow_up_str)
                            
                    except json.JSONDecodeError:
                        logger.warning(f"JSON decode error for chunk: {chunk_str}")
                        continue
                        
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg)
            yield format_sse_message(json.dumps({"error": error_msg}))
# Separate function for non-streaming API calls
async def fetch_ollama_response(messages, model, tools=None):
    """Non-streaming API function for direct responses"""
    async with aiohttp.ClientSession() as session:
        # Map OpenAI model to local model
        local_model = MODEL_MAPPING.get(model, MODEL_MAPPING["default"])
        
        # Prepare request data
        data = {
            'model': local_model,
            'messages': messages,
            'stream': False
        }
        
        # Always include tools if available
        if tools:
            logger.info(f"Adding tools to request: {json.dumps(tools)}")
            data['tools'] = tools
        else:
            # Always include default tools for consistent behavior
            logger.info(f"Adding default tools to request")
            data['tools'] = TOOL_DEFINITIONS
        
        # Get the appropriate URL based on the model
        url = MODEL_URLS.get(local_model, MODEL_URLS["default"])
        
        logger.info(f"Sending non-streaming request to {url} with model {local_model}")
        logger.debug(f"Request data: {json.dumps(data)}")
        
        try:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                response_text = await response.text()
                response_json = json.loads(response_text)
                logger.info(f"Received response from Ollama")
                logger.debug(f"Response: {json.dumps(response_json)}")
                return response_json
        except Exception as e:
            error_msg = f"Error in non-streaming request: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

# Define a new function to manage the conversation window
def manage_conversation_window(conversation_id, max_history=60):
    """
    Retrieve the last N messages from a conversation to use as context.
    If there are more than max_history messages, oldest messages will be excluded.
    
    Args:
        conversation_id: The ID of the conversation to retrieve
        max_history: Maximum number of messages to include in the context window
        
    Returns:
        List of messages in the conversation window
    """
    try:
        # Get all messages for this conversation
        all_messages = message_store.get_messages_by_conversation(conversation_id)
        
        # If we have more messages than our max_history, truncate
        if len(all_messages) > max_history:
            logger.info(f"Truncating conversation history from {len(all_messages)} to {max_history} messages")
            return all_messages[-max_history:]  # Keep only the most recent messages
        else:
            logger.info(f"Using full conversation history ({len(all_messages)} messages)")
            return all_messages
    except Exception as e:
        logger.error(f"Error retrieving conversation window: {str(e)}")
        return []

def format_sse_message(data):
    """Format a response as a Server-Sent Event message"""
    try:
        # If it's already a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data_obj = json.loads(data)
            except json.JSONDecodeError:
                # If it's not valid JSON, just wrap it in an SSE format
                return f"data: {data}\n\n"
        else:
            data_obj = data
        
        # Generate a unique ID for each message
        message_id = f"chatcmpl-{uuid4()}"
        current_time = int(datetime.now().timestamp())
        
        # Special handling for "done" messages
        if isinstance(data_obj, dict) and data_obj.get('done', False):
            # First send the final chunk
            done_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": current_time,
                "model": data_obj.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            return f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n"
        
        # Handle error messages
        if isinstance(data_obj, dict) and 'error' in data_obj:
            error_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": current_time,
                "model": data_obj.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": data_obj['error']
                        },
                        "finish_reason": "error"
                    }
                ]
            }
            return f"data: {json.dumps(error_chunk)}\n\n"
        
        # Handle message content
        if isinstance(data_obj, dict) and 'message' in data_obj:
            message = data_obj['message']
            
            # Handle content
            if 'content' in message:
                # Make sure content is a string
                content = str(message['content'])
                
                openai_chunk = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": data_obj.get('model', 'unknown'),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": content
                            },
                            "finish_reason": None
                        }
                    ]
                }
                return f"data: {json.dumps(openai_chunk)}\n\n"
                
            # Handle tool calls
            elif 'tool_calls' in message:
                openai_chunk = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": data_obj.get('model', 'unknown'),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": message['tool_calls']
                            },
                            "finish_reason": None
                        }
                    ]
                }
                return f"data: {json.dumps(openai_chunk)}\n\n"
        
        # Handle cases where we have a complete OpenAI-style chunk already
        if (isinstance(data_obj, dict) and 
            data_obj.get('object') == 'chat.completion.chunk' and
            'choices' in data_obj):
            # It's already formatted correctly, just wrap in SSE
            return f"data: {json.dumps(data_obj)}\n\n"
        
        # Default case - just wrap in SSE format with basic structure
        default_chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": current_time,
            "model": data_obj.get('model', 'unknown') if isinstance(data_obj, dict) else 'unknown',
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": str(data_obj) if not isinstance(data_obj, dict) else json.dumps(data_obj)
                    },
                    "finish_reason": None
                }
            ]
        }
        return f"data: {json.dumps(default_chunk)}\n\n"
    except Exception as e:
        # If any error occurs, format a basic error message
        error_chunk = {
            "id": f"chatcmpl-{uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "unknown",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": f"Error formatting message: {str(e)}"
                    },
                    "finish_reason": "error"
                }
            ]
        }
        return f"data: {json.dumps(error_chunk)}\n\n"
# Pydantic models for OpenAI API
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class FunctionParameter(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None

class FunctionObject(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: FunctionParameter

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str
    function: ToolFunction

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    tools: Optional[List[Dict[str, Any]]] = None
    user: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None




@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Extract request parameters
    client_messages = request.messages.copy()
    model = request.model
    stream = request.stream
    user_provided_tools = request.tools
    
    # Extract latest user message
    latest_user_message = None
    for msg in reversed(client_messages):
        if msg['role'] == 'user':
            latest_user_message = msg
            break
    
    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found in request")
    
    user_query = latest_user_message['content']
    
    # Add user message to history
    add_message_to_history('user', user_query)
    
    # Classify intent
    prediction_result, _ = await classify_intent_and_select_tools(user_query)
    
    # Log the intent classification result
    if prediction_result and "predictions" in prediction_result and prediction_result["predictions"]:
        top_prediction = prediction_result["predictions"][0]
        logger.info(f"Classified intent: {top_prediction['intent']} with confidence {top_prediction['probability']:.4f}")
    
    # Get messages for model
    messages, token_stats = get_messages_with_system_prompt()
    
    # Store user query in Redis for vector search
    if message_store:
        try:
            embed_and_save(user_query, "primary_conversation", "user")
        except Exception as e:
            logger.error(f"Error storing message in Redis: {str(e)}")
    
    # Handle streaming response
    if stream:
        # Use the intent-based handler for streaming responses
        return StreamingResponse(
            handle_request_based_on_intent(messages, model, user_query, prediction_result, stream=True),
            media_type="text/event-stream"
        )
    else:
        # For non-streaming requests
        try:
            # Determine if we should skip tools based on intent
            skip_tools = False
            if prediction_result and "predictions" in prediction_result and prediction_result["predictions"]:
                top_intent = prediction_result["predictions"][0]["intent"]
                confidence = prediction_result["predictions"][0]["probability"]
                
                # Skip tools for "none" or "thinking" intents or low confidence
                if top_intent in ["none", "thinking"] or confidence < INTENT_CONFIDENCE_THRESHOLDS.get(top_intent, 0.3):
                    skip_tools = True
                    logger.info(f"Skipping tools for intent '{top_intent}' with confidence {confidence:.4f}")
            
            # Use user-provided tools if specified, otherwise use intent-based tools
            tools = None
            if not skip_tools:
                if user_provided_tools is not None:
                    tools = user_provided_tools
                else:
                    # Get tools based on intent
                    tools = get_tools_for_intent(prediction_result, TOOL_DEFINITIONS)
                    if not tools:
                        tools = TOOL_DEFINITIONS
            
            # Log what we're doing
            if skip_tools:
                logger.info(f"Sending request without tools due to intent classification")
            else:
                logger.info(f"Sending request with {len(tools) if tools else 0} tools")
            
            # Send request
            response = await fetch_ollama_response(messages, model, tools)
            
            # Process tool calls if present and tools were not skipped
            if not skip_tools and 'message' in response and 'tool_calls' in response['message']:
                logger.info("Processing tool calls in non-streaming response")
                
                # Process tool calls
                tool_outputs = await process_tool_calls(response)
                
                if tool_outputs:
                    # Format tool results
                    tool_results = ""
                    for output in tool_outputs:
                        if output.get('role') == 'tool' and 'content' in output:
                            tool_name = output.get('name', 'unknown_tool')
                            tool_results += f"Tool: {tool_name}\n"
                            tool_results += f"Result: {output['content']}\n\n"
                    
                    # Create system prompt with tool results
                    system_prompt = format_tool_system_prompt(
                        TOOL_SYSTEM_PROMPT_TEMPLATE,
                        user_query,
                        tool_results
                    )
                    
                    # Create follow-up messages
                    follow_up_messages = [
                        {'role': 'system', 'content': system_prompt}
                    ]
                    
                    # Send follow-up request with tool results
                    follow_up_response = await fetch_ollama_response(follow_up_messages, model, None)
                    
                    # Use this as our final response
                    response = follow_up_response
            
            # Extract content from response
            assistant_content = ""
            if 'message' in response and 'content' in response['message']:
                assistant_content = response['message']['content']
                add_message_to_history('assistant', assistant_content)
                
                # Store in Redis for vector search
                if message_store:
                    embed_and_save(assistant_content, "primary_conversation", "assistant")
            
            # Format response to match OpenAI's format
            openai_response = {
                "id": f"chatcmpl-{str(uuid4())}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": assistant_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": token_stats.get('total_tokens', 0),
                    "completion_tokens": len(assistant_content) // 4,  # Rough estimate
                    "total_tokens": token_stats.get('total_tokens', 0) + len(assistant_content) // 4
                }
            }
            
            return JSONResponse(content=openai_response)
            
        except Exception as e:
            logger.error(f"Error in completion: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in completion: {str(e)}")

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        embeddings_list = []
        for i, text in enumerate(input_texts):
            embedding_result = get_embeddings(text)
            embeddings_list.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding_result['embeddings']
            })
        
        response = {
            "object": "list",
            "data": embeddings_list,
            "model": request.model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in input_texts),
                "total_tokens": sum(len(text.split()) for text in input_texts)
            }
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

@app.get("/v1/models")
async def list_models():
    # Log the models request
    logger.info("Models endpoint called - returning available models")
    
    # Get current timestamp for "created" field
    timestamp = int(datetime.now().timestamp())
    
    # Create models list with both OpenAI model names and actual local models
    models = []
    
    # Add OpenAI model names (for compatibility with OpenAI clients)
    for openai_model in MODEL_MAPPING.keys():
        if openai_model != "default":
            models.append({
                "id": openai_model,
                "object": "model",
                "created": timestamp,
                "owned_by": "local"
            })
    
    # Add actual local model names
    for local_model in AVAILABLE_MODELS:
        models.append({
            "id": local_model,
            "object": "model",
            "created": timestamp,
            "owned_by": "local"
        })
    
    # Log the response
    logger.info(f"Returning {len(models)} models")
    
    response = {"object": "list", "data": models}
    return JSONResponse(content=response)

# Backward compatibility with the original API
@app.post("/api/chat")
async def receive_chat_message(data: dict = Body(...)):
    try:
        message = data["message"]
        model = data["model"]
        msg_content = message["content"]
        
        logger.info(f"Legacy chat API called")
        
        # Add user message to history
        add_message_to_history('user', msg_content)
        
        # Optionally store in Redis for vector search
        if message_store:
            embed_and_save(msg_content, "primary_conversation", "user")
        
        # Classify intent
        prediction_result, _ = await classify_intent_and_select_tools(msg_content)
        
        # Log the intent classification result
        if prediction_result and "predictions" in prediction_result and prediction_result["predictions"]:
            top_prediction = prediction_result["predictions"][0]
            logger.info(f"Legacy API - classified intent: {top_prediction['intent']} with confidence {top_prediction['probability']:.4f}")
        
        # Get messages for model
        messages = get_messages_with_system_prompt()
        
        logger.info(f"Sending {len(messages)} messages to model in legacy endpoint")

        async def stream_response():
            # Use the intent-based handler
            full_response = ""
            async for chunk in handle_request_based_on_intent(messages, model, msg_content, prediction_result, stream=True):
                yield chunk
                
                # Extract content for saving
                try:
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if 'choices' in chunk_data and chunk_data['choices'] and 'delta' in chunk_data['choices'][0]:
                        delta = chunk_data['choices'][0]['delta']
                        if 'content' in delta and delta['content']:
                            full_response += delta['content']
                except Exception as e:
                    pass
            
            # Return fixed conversation ID in the response
            yield f"\ndata: {{\"conversation_id\": \"primary_conversation\"}}\n\n"
            
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Error in legacy chat endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/debug/messages/{conversation_id}")
async def debug_messages(conversation_id: str):
    """Debug endpoint to check message retrieval for a conversation"""
    try:
        # Try direct key listing first
        direct_keys = message_store.message_store.keys(f"message:{conversation_id}:*")
        
        # Try to get a sample of raw data
        sample_data = []
        for key in direct_keys[:3]:  # Just look at first 3 keys
            try:
                # Get the raw data
                raw_json = message_store.message_store.json().get(key)
                sample_data.append({
                    "key": key,
                    "data": raw_json
                })
            except Exception as e:
                sample_data.append({
                    "key": key,
                    "error": str(e)
                })
        
        # Try normal retrieval
        messages = []
        try:
            messages = message_store.get_messages_by_conversation(conversation_id)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving messages: {str(e)}",
                "direct_keys_count": len(direct_keys),
                "direct_keys_sample": [k for k in direct_keys[:5]],
                "sample_data": sample_data
            }
        
        return {
            "status": "success",
            "message_count": len(messages),
            "messages": messages[:5],  # Return first 5 messages
            "direct_keys_count": len(direct_keys),
            "direct_keys_sample": [k for k in direct_keys[:15]],
            "sample_data": sample_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

class NoteCreate(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = None

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None

# Helper function to list all notes
async def list_notes():
    """List all notes from the notes directory."""
    try:
        notes = []
        for filename in os.listdir(NOTES_DIRECTORY):
            if filename.endswith(".json"):
                file_path = os.path.join(NOTES_DIRECTORY, filename)
                with open(file_path, "r") as f:
                    note_data = json.load(f)
                    notes.append({
                        "filename": filename,
                        **note_data
                    })
        return notes
    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to get note names for the prompt
async def get_note_names_for_prompt():
    """Get note names for the prompt."""
    try:
        notes = await list_notes()  # Await the coroutine
        note_names = [note["title"] for note in notes]  # Extract note titles
        return note_names
    except Exception as e:
        logger.error(f"Error getting note names for prompt: {e}")
        raise

# Example usage of get_note_names_for_prompt
async def some_other_function():
    try:
        note_names = await get_note_names_for_prompt()  # Await the coroutine
        logger.info(f"Note names: {note_names}")
    except Exception as e:
        logger.error(f"Error in some_other_function: {e}")
        raise

# API Endpoints
@dashboard_router.get("/v1/notes")
async def list_notes_endpoint(limit: int = 100, offset: int = 0):
    """List all notes with pagination"""
    try:
        notes = await list_notes()  # Await the coroutine
        
        # Apply pagination
        paginated_notes = notes[offset:offset + limit]
        
        return {"notes": paginated_notes, "total": len(notes)}
    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/v1/notes/count")
async def get_notes_count():
    """Get the total count of notes"""
    try:
        notes = await list_notes()  # Await the coroutine
        return {"count": len(notes)}
    except Exception as e:
        logger.error(f"Error counting notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.post("/v1/notes")
async def create_note_endpoint(note: NoteCreate):
    """Create a new note"""
    try:
        result = create_note(
            title=note.title,
            content=note.content,
            tags=note.tags or []
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/v1/notes/{note_id}")
async def get_note_endpoint(note_id: str):
    """Get a note by its identifier"""
    try:
        # Await the async function
        note = await read_note(note_id)
        
        if "error" in note:
            raise HTTPException(status_code=404, detail=note["error"])
            
        return note
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.put("/v1/notes/{note_id}")
async def update_note_endpoint(note_id: str, note: NoteUpdate):
    """Update a note"""
    try:
        # First get the existing note
        existing = read_note(note_id)
        
        if "error" in existing:
            raise HTTPException(status_code=404, detail=existing["error"])
        
        # Update the fields that are provided
        updated_content = existing.get('content', '')
        
        if note.content is not None:
            updated_content = note.content
        
        # For a complete update, use append_note with a complete replacement
        result = append_note(note_id, updated_content, replace=True)
        
        # Update title and tags if needed
        if note.title is not None or note.tags is not None:
            # Get the file path
            filename = existing.get('filename')
            file_path = os.path.join(NOTES_DIRECTORY, filename)
            
            with open(file_path, 'r') as f:
                note_data = json.load(f)
            
            if note.title is not None:
                note_data['title'] = note.title
                
            if note.tags is not None:
                note_data['tags'] = note.tags
                
            note_data['last_modified'] = datetime.now().isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(note_data, f, indent=2)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return {"status": "success", "message": "Note updated successfully", "note_id": note_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.delete("/v1/notes/{note_id}")
async def delete_note_endpoint(note_id: str):
    """Delete a note"""
    try:
        result = delete_note(note_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return {"status": "success", "message": "Note deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Memory Endpoints ---

class MemoryCreate(BaseModel):
    memory: str

class MemoriesRewrite(BaseModel):
    memories: List[str]

@dashboard_router.get("/v1/memory/core")
async def get_core_memories():
    """Get all core memories"""
    try:
        content = load_core_memories()
        
        # Parse the content into a list of memories
        memories = []
        if content:
            # Split by lines and remove any empty lines
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # Remove leading dash if present
                    if line.startswith('- '):
                        line = line[2:]
                    memories.append(line)
        
        return {"memories": memories}
    except Exception as e:
        logger.error(f"Error loading core memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.post("/v1/memory/core")
async def add_core_memory(memory: MemoryCreate):
    """Add a new core memory"""
    try:
        result = append_core_memory(memory.memory)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to add memory"))
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding core memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.post("/v1/memory/core/rewrite")
async def rewrite_core_memories_endpoint(memories: MemoriesRewrite):
    """Rewrite all core memories"""
    try:
        result = rewrite_core_memories(memories.memories)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to rewrite memories"))
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rewriting core memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/v1/memory/core/stats")
async def get_core_memory_stats_endpoint():
    """Get statistics about core memories"""
    try:
        result = get_core_memory_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting core memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- System Endpoints ---

@dashboard_router.get("/v1/models/mapping")
async def get_model_mapping():
    """Get the mapping between OpenAI models and local models"""
    return MODEL_MAPPING

@app.get("/v1/system/gpu")
async def get_gpu_usage():
    """Get GPU usage information (supports multiple GPUs)"""
    try:
        # Use nvidia-smi command to get GPU info
        try:
            # Command to get all GPUs' data
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8').strip()
            
            # Split by newlines to get each GPU's data
            gpu_lines = output.strip().split('\n')
            
            # Process all GPUs
            gpus = []
            for line in gpu_lines:
                values = [val.strip() for val in line.split(',')]
                
                if len(values) >= 4:
                    try:
                        gpu_index = int(values[0])
                        memory_used = float(values[1])
                        memory_total = float(values[2])
                        utilization = float(values[3])
                        
                        gpus.append({
                            "index": gpu_index,
                            "memory_used": memory_used / 1024,  # Convert to GB
                            "memory_total": memory_total / 1024,  # Convert to GB
                            "utilization": utilization
                        })
                    except ValueError:
                        logger.warning(f"Could not convert GPU values to float: {values}")
                        continue
            
            # Return information about all GPUs
            return {
                "gpus": gpus,
                "total_gpus": len(gpus),
                "primary_gpu": gpus[0] if gpus else {
                    "memory_used": 0,
                    "memory_total": 0,
                    "utilization": 0
                }
            }
                
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            logger.warning(f"Failed to get GPU info from nvidia-smi: {e}")
            return {
                "gpus": [],
                "total_gpus": 0,
                "primary_gpu": {
                    "memory_used": 0,
                    "memory_total": 0,
                    "utilization": 0
                },
                "error": str(e)
            }
    except Exception as e:
        logger.error(f"Error getting GPU usage: {e}")
        return {
            "gpus": [],
            "total_gpus": 0,
            "primary_gpu": {
                "memory_used": 0,
                "memory_total": 0,
                "utilization": 0
            },
            "error": str(e)
        }

def use_default_values(error_message, gpu_count=1):
    """Helper function to return default GPU values"""
    return {
        "memory_used": 4,  # Default value
        "memory_total": 16,  # Default value
        "utilization": 30,  # Default value
        "total_gpus": gpu_count,
        "error": error_message
    }
@dashboard_router.get("/v1/system/log", response_class=PlainTextResponse)
async def get_system_log():
    """Get the last 100 lines of the system log"""
    try:
        log_file = "/home/david/sara-jarvis/Test/openai_server.log"
        
        if not os.path.exists(log_file):
            return "Log file not found"
            
        # Get last 100 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-100:])
    except Exception as e:
        logger.error(f"Error reading system log: {e}")
        return f"Error reading system log: {str(e)}"

# --- Dashboard API ---

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard page"""
    dashboard_html_path = "/home/david/Sara/static/dashboard.html"
    return FileResponse(dashboard_html_path)

# Add the dashboard router to the main app
app.include_router(dashboard_router)

# Helper function to append note with replace option
def append_note(identifier: str, content: str, replace: bool = False) -> Dict[str, Any]:
    """Append content to an existing note or replace its content"""
    try:
        # First, find the note
        note = read_note(identifier)
        if "error" in note:
            return note
        
        # Get the file path
        filename = note.get('filename')
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Update content and timestamp
        if replace:
            note['content'] = content
        else:
            note['content'] = note.get('content', '') + '\n\n' + content
            
        note['last_modified'] = datetime.now().isoformat()
        
        # Write the updated note
        with open(file_path, 'w') as f:
            json.dump(note, f, indent=2)
        
        logger.info(f"{'Updated' if replace else 'Appended to'} note: {note.get('title')} ({filename})")
        return note
    except Exception as e:
        logger.error(f"Error {'updating' if replace else 'appending to'} note: {e}")
        return {"error": str(e)}

@app.get("/v1/system/resources")
async def get_system_resources():
    """Get system CPU and RAM usage information"""
    try:
        # Get CPU usage percentage
        cpu_usage = psutil.cpu_percent(interval=0.5)
        
        # Get memory information
        memory = psutil.virtual_memory()
        ram_used = memory.used / (1024 ** 3)  # Convert to GB
        ram_total = memory.total / (1024 ** 3)  # Convert to GB
        
        return {
            "cpu_usage": cpu_usage,
            "ram_used": round(ram_used, 2),
            "ram_total": round(ram_total, 2)
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return {
            "cpu_usage": 0,
            "ram_used": 0,
            "ram_total": 0,
            "error": str(e)
        }


@app.get("/v1/system/gpu")
async def get_gpu_usage():
    """Get GPU usage information"""
    try:
        # Use nvidia-smi command to get GPU info
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8').strip()
            
            # Clean the output - handle potential newlines or extra whitespace
            cleaned_output = output.replace('\n', ',').replace('\r', '')
            values = [val.strip() for val in cleaned_output.split(',') if val.strip()]
            
            if len(values) >= 3:
                try:
                    memory_used = float(values[0])
                    memory_total = float(values[1])
                    utilization = float(values[2])
                    
                    return {
                        "memory_used": memory_used / 1024,  # Convert to GB
                        "memory_total": memory_total / 1024,  # Convert to GB
                        "utilization": utilization
                    }
                except ValueError as ve:
                    logger.warning(f"Could not convert GPU values to float: {values}, error: {ve}")
                    # Fall back to default values
                    return {
                        "memory_used": 4,  # Default value
                        "memory_total": 16,  # Default value
                        "utilization": 30,  # Default value
                        "error": f"Value conversion error: {ve}"
                    }
            else:
                raise ValueError(f"Not enough values returned from nvidia-smi: {values}")
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            logger.warning(f"Failed to get GPU info from nvidia-smi: {e}")
            # Return default values
            return {
                "memory_used": 4,  # Default value
                "memory_total": 16,  # Default value
                "utilization": 30,  # Default value
                "error": str(e)
            }
    except Exception as e:
        logger.error(f"Error getting GPU usage: {e}")
        return {
            "memory_used": 4,  # Default value
            "memory_total": 16,  # Default value
            "utilization": 30,  # Default value
            "error": str(e)
        }

@app.post("/v1/clear-conversation")
async def clear_conversation(request: dict = Body(...)):
    """Clear the conversation history for a specific conversation ID"""
    try:
        conversation_id = request.get("conversation_id", "current_session")
        logger.info(f"Clearing conversation history for conversation_id: {conversation_id}")
        
        # Use the existing clear_conversation_history function if clearing the in-memory history
        if conversation_id == "current_session" or conversation_id == "primary_conversation":
            clear_conversation_history()
            logger.info("In-memory conversation history cleared")
        
        # If using Redis/message store, attempt to delete messages for the conversation
        if message_store:
            try:
                # This will use the delete_conversation method if implemented
                if hasattr(message_store, 'delete_conversation') and callable(getattr(message_store, 'delete_conversation')):
                    success = message_store.delete_conversation(conversation_id)
                    logger.info(f"Deleted conversation {conversation_id} from message store: {success}")
                # Alternative approach using direct Redis commands if available
                elif hasattr(message_store, 'message_store') and hasattr(message_store.message_store, 'keys'):
                    # Get all keys for this conversation
                    pattern = f"message:{conversation_id}:*"
                    keys = message_store.message_store.keys(pattern)
                    if keys:
                        # Delete all keys
                        message_store.message_store.delete(*keys)
                        logger.info(f"Deleted {len(keys)} messages for conversation {conversation_id}")
            except Exception as e:
                logger.error(f"Error deleting conversation from message store: {str(e)}")
                # We continue even if there's an error with the message store
        
        return {"status": "success", "message": f"Conversation {conversation_id} cleared"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.get("/v1/neo4j/check-direct")
async def check_neo4j_direct():
    """Direct endpoint to check Neo4j connection"""
    try:
        # Get the Neo4j RAG manager
        neo4j_manager = get_neo4j_rag_manager()
        
        if not neo4j_manager or not hasattr(neo4j_manager, 'driver'):
            return {"status": "Not Configured"}
        
        # Try a direct query
        try:
            with neo4j_manager.driver.session(database=neo4j_manager.neo4j_db) as session:
                result = session.run("RETURN 1 as num")
                record = result.single()
                status = "Connected" if record and record["num"] == 1 else "Disconnected"
                return {"status": status}
        except Exception as e:
            logger.error(f"Direct Neo4j check failed: {e}")
            return {"status": "Error"}
    except Exception as e:
        logger.error(f"Error in direct Neo4j check: {e}")
        return {"status": "Error"}

# Add this code to your server.py file

def clean_text_for_tts(text):
    """Clean text for TTS by removing emojis, asterisks, and other special characters"""
    # Remove emoji pattern
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove markdown formatting characters
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold: **text** to text
    text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic: *text* to text
    text = re.sub(r'__(.+?)__', r'\1', text)      # Underline: __text__ to text
    text = re.sub(r'_(.+?)_', r'\1', text)        # Italic with underscore: _text_ to text
    
    # Remove code blocks and inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'`(.+?)`', r'\1', text)        # Inline code: `text` to text
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Convert common symbols to their spoken form
    text = text.replace('&', ' and ')
    text = text.replace('%', ' percent ')
    text = text.replace('/', ' slash ')
    text = text.replace('=', ' equals ')
    
    # Remove excess whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

@app.post("/v1/tts/generate")
async def generate_speech(request: dict = Body(...)):
    """Generate speech from text using TTS API"""
    try:
        text = request.get("text", "")
        voice = request.get("voice", "af_bella")  # Default to af_bella only if not provided
        speed = request.get("speed", 1.0)
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Clean the text for better TTS rendering
        cleaned_text = clean_text_for_tts(text)
        
        logger.info(f"Generating speech for text of length {len(cleaned_text)} with voice {voice}")
        
        # Use the simpler request approach
        try:
            # Make a direct request to the TTS API
            response = requests.post(
                "http://10.185.1.8:8880/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": cleaned_text,
                    "voice": voice,  # Use the voice provided in the request
                    "response_format": "mp3",
                    "speed": speed
                }
            )
            
            # Raise an error if the request failed
            response.raise_for_status()
            
            # Return the audio data directly
            return Response(
                content=response.content,
                media_type="audio/mpeg"
            )
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

# Add a voice list endpoint that actually queries the TTS service
@app.get("/v1/tts/voices")
async def list_voices():
    """List available TTS voices from the TTS service"""
    try:
        # Try to query the TTS service for voices
        try:
            response = requests.get("http://10.185.1.8:8880/v1/audio/voices", timeout=3)
            
            if response.status_code == 200:
                # Log successful response
                voices_data = response.json()
                logger.info(f"Successfully fetched voices from TTS service: {voices_data}")
                return voices_data
            else:
                # Log error response
                logger.warning(f"TTS service returned non-200 status: {response.status_code}")
        except requests.RequestException as e:
            logger.warning(f"Error connecting to TTS service: {str(e)}")
        
        # Fallback voices - return these if we couldn't connect to the service
        # or if the service didn't return valid data
        fallback_voices = [
            {"id": "af_bella", "name": "Bella (African)"},
            {"id": "en_jony", "name": "Jony (English)"},
            {"id": "en_rachel", "name": "Rachel (English)"},
            {"id": "en_emma", "name": "Emma (English)"},
            {"id": "en_antoni", "name": "Antoni (English)"}
        ]
        
        logger.info("Using fallback voices list")
        return {"voices": fallback_voices}
        
    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error in list_voices: {str(e)}")
        
        # Always return something usable, even if there's an error
        fallback_voices = [
            {"id": "af_bella", "name": "Bella (African)"},
            {"id": "en_jony", "name": "Jony (English)"}
        ]
        return {"voices": fallback_voices}

# Add a health check endpoint for the TTS service
@app.get("/v1/tts/status")
async def tts_status():
    """Check TTS service status"""
    try:
        # Try to get voices as a status check
        response = requests.get("http://10.185.1.8:8880/v1/audio/voices")
        
        if response.status_code == 200:
            return {"status": "online", "message": "TTS service is available"}
        else:
            return {"status": "error", "message": f"TTS service returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "offline", "message": f"TTS service is unavailable: {str(e)}"}

@app.post("/v1/debug/classify-intent")
async def debug_classify_intent(request: dict = Body(...)):
    """Debug endpoint to test intent classification"""
    query = request.get("query", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        # Classify intent
        prediction_result, selected_tools = await classify_intent_and_select_tools(query)
        
        # Format response
        response = {
            "query": query,
            "prediction": prediction_result,
            "selected_tools": [tool['function']['name'] for tool in selected_tools] if selected_tools else []
        }
        
        return response
    except Exception as e:
        logger.error(f"Error in debug intent classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error classifying intent: {str(e)}")

@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint that also checks Redis and Neo4j connections"""
    logger.info("Health check endpoint called")
    
    # Check Redis connection
    redis_status = "connected" if message_store.ping() else "disconnected"
    
    # Check Neo4j connection
    neo4j_status = check_neo4j_connection()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_status,
            "neo4j": neo4j_status.lower(),
            "server": "online"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 7009")
    uvicorn.run(app, host="0.0.0.0", port=7009, log_level="info")

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
from fastapi import BackgroundTasks
from modules.neo4j_rag_integration import Neo4jRAGManager, get_neo4j_rag_manager
from modules.rag_api import rag_router  # Import this first
from modules.rag_web_interface import integrate_web_interface
from modules.rag_integration import integrate_rag_with_server, update_system_prompt_with_rag_info
from fastapi.middleware.cors import CORSMiddleware
from modules.timer_reminder_integration import integrate_timer_reminder_tools
from modules.neo4j_integration import get_message_store, integrate_neo4j_with_server



# Initialize the message store with Neo4j backend
message_store = get_message_store(use_neo4j=True)

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

# Ensure notes directory exists
os.makedirs(NOTES_DIRECTORY, exist_ok=True)


# Initialize Perplexica client
perplexica = PerplexicaClient(base_url="http://localhost:3001")
SYSTEM_PROMPT = ""  # Will be loaded during startup
TOOL_SYSTEM_PROMPT_TEMPLATE = ""  # Will be loaded during startup
CORE_MEMORY_FILE = "/home/david/Sara/core_memories.txt"
# Model mapping from OpenAI to local models
MODEL_MAPPING = {
    "gpt-4": "llama3.3",
    "gpt-3.5-turbo": "mistral-small",
    "gpt-3.5-turbo-0125": "llama3.1",
    "gpt-3.5-turbo-1106": "llama3.2",
    # Add more mappings as needed
    "default": "llama3.3"
}

# Available local models
AVAILABLE_MODELS = [
    "llama3.3:latest",
    "mistral-small:latest",
    "llama3.2:latest",
    "llama3.1:latest"
]

# URLs for different models
MODEL_URLS = {
    "llama3.3": "http://100.82.117.46:11434/api/chat",
    "llama3.2": "http://localhost:11434/api/chat",
    "llama3.1": "http://localhost:11434/api/chat",
    "mistral-small": "http://localhost:11434/api/chat",
    "default": "http://localhost:11434/api/chat"
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
        """Retrieve all messages for a specific conversation, ordered by timestamp"""
        try:
            # Get all message keys for this conversation
            raw_keys = self.message_store.keys(f"message:{conversation_id}:*")
            
            # Filter out embedding keys
            message_keys = []
            for key in raw_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Skip embedding keys
                if not key_str.endswith(":embedding"):
                    message_keys.append(key_str)
            
            if not message_keys:
                return []
            
            messages = []
            for key in message_keys:
                try:
                    # Try to get JSON data safely
                    try:
                        json_data = self.message_store.json().get(key)
                        
                        if json_data and isinstance(json_data, dict):
                            # Extract just the data we need
                            messages.append({
                                "role": json_data.get("role", "unknown"),
                                "content": json_data.get("content", ""),
                                "timestamp": json_data.get("timestamp", 0)
                            })
                    except Exception as json_err:
                        logger.warning(f"Error getting JSON for key {key}: {json_err}")
                        # Try alternative retrieval method
                        try:
                            raw_data = self.message_store.get(key)
                            if raw_data and isinstance(raw_data, bytes):
                                try:
                                    # Try to decode and parse as JSON
                                    json_str = raw_data.decode('utf-8')
                                    json_data = json.loads(json_str)
                                    messages.append({
                                        "role": json_data.get("role", "unknown"),
                                        "content": json_data.get("content", ""),
                                        "timestamp": json_data.get("timestamp", 0)
                                    })
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    # Not valid UTF-8 or JSON, skip this
                                    continue
                        except Exception:
                            # Skip this message
                            continue
                except Exception as e:
                    logger.warning(f"Error processing message for key {key}: {e}")
                    continue
            
            # Sort messages by timestamp
            messages.sort(key=lambda x: x.get("timestamp", 0))
            
            return messages
        except Exception as e:
            logger.error(f"Error retrieving messages: {str(e)}")
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

# Update the startup_event function
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
    app.include_router(rag_router, prefix="/rag")
    integrate_web_interface(app)
    # Update the system prompt with available notes
    SYSTEM_PROMPT = update_system_prompt_with_notes(SYSTEM_PROMPT)
    logger.info("System prompt updated with available notes")
    
    # Update the system prompt with RAG information
    SYSTEM_PROMPT = update_system_prompt_with_rag_info(SYSTEM_PROMPT)
    logger.info("System prompt updated with RAG information")
    
    TOOL_SYSTEM_PROMPT_TEMPLATE = load_tool_system_prompt()
    logger.info("Tool system prompt template loaded")
    
    # Check database connection
    if message_store.ping():
        logger.info("Connected to database successfully")
    else:
        logger.warning("Failed to connect to database - functionality may be limited")
    
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

@app.get("/v1/conversations")
async def list_conversations(limit: int = 20, offset: int = 0):
    """List all conversations"""
    try:
        conversations = message_store.list_conversations(limit=limit, offset=offset)
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
    embedding = text_to_embedding_ollama(query)['embeddings']
    
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
    }@app.middleware("http")

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

def update_user_profile(user_id, conversation_id):
    """Update user profile based on conversation"""
    try:
        # Use a consistent identifier for the user profile
        profile_id = "user_profile"
        profile_file_path = os.path.join(NOTES_DIRECTORY, "user_profile.json")
        
        # Check if profile exists, create if not
        if not os.path.exists(profile_file_path):
            # Create the user profile file directly with a fixed filename
            logger.info("Creating new user profile")
            
            # Create note data
            now = datetime.now().isoformat()
            note_data = {
                "title": "User Profile",
                "content": "Initial user profile.",
                "tags": ["user_profile"],
                "created": now,
                "last_modified": now,
                "filename": "user_profile.json"
            }
            
            # Write directly to the specified path
            with open(profile_file_path, 'w') as f:
                json.dump(note_data, f, indent=2)
        
        # Get messages using the client's built-in method
        messages = message_store.get_messages_by_conversation(conversation_id)
        
        logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
        
        # Filter to get only user messages
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        logger.info(f"Found {len(user_messages)} user messages from {len(messages)} total messages")
        
        # Skip if fewer than required messages
        if len(user_messages) < 3:  # Changed from 15 to 3 for testing
            logger.info(f"Not enough messages in current conversation: Only {len(user_messages)} user messages")
            return None
            
        # Prepare prompt to extract user information
        extract_prompt = "Based on this conversation, what new information did we learn about the user? List any preferences, interests, or personal details mentioned."
        
        # Format conversation for the model
        formatted_convo = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Make model request
        extract_messages = [
            {'role': 'system', 'content': extract_prompt},
            {'role': 'user', 'content': formatted_convo}
        ]
        
        # Get extraction
        try:
            response = asyncio.run(fetch_ollama_response(extract_messages, "llama3.1", stream=False))
            if 'message' in response and 'content' in response['message']:
                new_info = response['message']['content']
                
                # Read the current profile
                with open(profile_file_path, 'r') as f:
                    profile_data = json.load(f)
                
                # Update content
                profile_data['content'] += f"\nUpdated {datetime.now().strftime('%Y-%m-%d')}:\n{new_info}"
                profile_data['last_modified'] = datetime.now().isoformat()
                
                # Write updated profile
                with open(profile_file_path, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                
                logger.info("Updated user profile with new information")
                return new_info
        except Exception as e:
            logger.error(f"Error in profile extraction: {str(e)}")
            
        return None
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        return None

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
    def __init__(self, max_buffer_size=10):
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

async def summarize_buffer_and_update_profile(conversation_id):
    """
    Summarize the conversation buffer and update the user profile
    This function should be called as a background task
    """
    # Skip if summarization is already in progress for this conversation
    if conversation_buffer.is_summarization_in_progress(conversation_id):
        logger.info(f"Summarization already in progress for conversation {conversation_id}")
        return
    
    # Mark summarization as started
    conversation_buffer.mark_summarization_started(conversation_id)
    
    try:
        # Get messages from buffer
        buffer_messages = conversation_buffer.get_messages(conversation_id)
        
        if not buffer_messages or len(buffer_messages) < 3:  # Minimum threshold for summarization
            logger.info(f"Not enough messages in buffer for summarization: {len(buffer_messages)}")
            conversation_buffer.mark_summarization_completed(conversation_id)
            return
        
        logger.info(f"Summarizing buffer with {len(buffer_messages)} messages for conversation {conversation_id}")
        
        # Format conversation for the model
        formatted_convo = "\n".join([f"{msg['role']}: {msg['content']}" for msg in buffer_messages])
        
        # Create a prompt specific to user profile extraction
        extract_prompt = """
        Based on this conversation segment, extract key information about the user such as:
        1. Preferences and interests
        2. Personal details they've shared
        3. Opinions or values they've expressed
        4. Any other relevant information for building a user profile
        
        Provide this information in a structured list format that could be added to a user profile.
        Focus only on new information, not what was already known.
        """
        
        # Make model request for extraction
        extract_messages = [
            {'role': 'system', 'content': extract_prompt},
            {'role': 'user', 'content': formatted_convo}
        ]
        
        # Get extraction using non-streaming request
        response = await fetch_ollama_response(extract_messages, "llama3.1", stream=False)
        
        if 'message' in response and 'content' in response['message']:
            new_info = response['message']['content']
            
            # Check if new information was extracted
            if "no new information" in new_info.lower() or "not enough information" in new_info.lower():
                logger.info(f"No new user information extracted from conversation {conversation_id}")
            else:
                # Update the single user profile in the notes directory
                profile_id = "user_profile"
                existing_profile = read_note(profile_id)
                
                if "error" in existing_profile:
                    # Create new profile if it doesn't exist
                    logger.info("Creating new user profile")
                    create_note(
                        title="User Profile",
                        content=f"# User Profile\nCreated: {datetime.now().strftime('%Y-%m-%d')}\n\n{new_info}",
                        tags=["user_profile"]
                    )
                else:
                    # Append to existing profile
                    logger.info("Updating user profile")
                    append_note(
                        profile_id,
                        f"\n\n## Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{new_info}"
                    )
                
                logger.info("User profile updated with new information")
        
        # Also generate a summary of this conversation segment
        summary_prompt = "Summarize the key points of this conversation segment in 2-3 sentences."
        summary_messages = [
            {'role': 'system', 'content': summary_prompt},
            {'role': 'user', 'content': formatted_convo}
        ]
        
        summary_response = await fetch_ollama_response(summary_messages, "llama3.1", stream=False)
        
        if 'message' in summary_response and 'content' in summary_response['message']:
            summary = summary_response['message']['content']
            
            # Store this summary with a special tag
            today = datetime.now()
            summary_id = f"summary-segment-{conversation_id}-{today.strftime('%Y%m%d%H%M%S')}"
            
            message_store.store_message(
                role='system',
                content=summary,
                conversation_id=summary_id,
                date=today,
                embedding=text_to_embedding_ollama(summary)['embeddings']
            )
            
            logger.info(f"Stored segment summary: {summary_id}")
    except Exception as e:
        logger.error(f"Error in summarization process: {str(e)}")
        logger.exception(e)
    finally:
        # Clear the buffer and mark summarization as completed
        conversation_buffer.clear_buffer(conversation_id)
        conversation_buffer.mark_summarization_completed(conversation_id)

async def maybe_summarize_conversation(conversation_id):
    """Summarize only if the conversation is substantial"""
    messages = message_store.get_messages_by_conversation(conversation_id)
    if len(messages) >= 10:  # Only summarize conversations with 10+ messages
        return await summarize_conversation(conversation_id)
    return None

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
            embedding=text_to_embedding_ollama(summary)['embeddings']
        )
        
        return summary
    return None

def read_note(identifier: str) -> Dict[str, Any]:
    """Read a note by title or filename"""
    try:
        notes = list_notes()
        
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

# Example of how to update the system prompt with note information
def update_system_prompt_with_notes(original_prompt: str) -> str:
    """Update the system prompt to include available notes"""
    notes_info = get_note_names_for_prompt()
    
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

async def retrieve_relevant_memories(user_query, conversation_id=None, k=5):
    """Retrieve relevant past conversations based on the user query, excluding recent messages from current conversation"""
    try:
        # Get embedding for the current query
        query_embedding = text_to_embedding_ollama(user_query)
        
        # Check if vector_search method exists
        if hasattr(message_store, 'vector_search'):
            # Search for similar past messages
            similar_messages = message_store.vector_search(query_embedding['embeddings'], k=k+5)  # Get more than needed to allow for filtering
            
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
            'description': 'Think about a response to the user. Use this tool to think about what the user said and the conversation context before formulating your response. An example would be "the user is saying good morning, I should respond in a cute funny manner" or "the user seems to be struggling with something, I should try and cheer him up". ',
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

def get_message_by_key(self, key):
    """Retrieve a message by key, handling different Redis data types"""
    try:
        # Check if key is bytes and decode if needed
        if isinstance(key, bytes):
            try:
                key = key.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"Could not decode key: {key}")
                return None
                
        # Check what type the key is
        key_type = self.message_store.type(key)
        
        if key_type == "ReJSON-RL":
            # It's a JSON object
            data = self.message_store.json().get(key)
            return data
        elif key_type == "hash":
            # It's a hash - get all fields
            data = self.message_store.hgetall(key)
            return data
        elif key_type == "string":
            # It's a string - try to parse as JSON
            try:
                raw_data = self.message_store.get(key)
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode('utf-8')
                return json.loads(raw_data)
            except:
                # If can't parse as JSON, return as is
                if isinstance(raw_data, bytes):
                    try:
                        raw_data = raw_data.decode('utf-8')
                    except UnicodeDecodeError:
                        # If it's binary data and can't be decoded, return a placeholder
                        return {"content": "(binary data)", "role": "unknown"}
                return {"content": raw_data, "role": "unknown"}
        else:
            return None
    except Exception as e:
        logger.warning(f"Error retrieving message {key}: {e}")
        return None

# File operations
def write_to_file(role, content, dates):
    try:
        with open(f"/home/david/sara-jarvis/Test/{dates}.txt",'x+') as f:
            pass
    except FileExistsError:
        pass
    with open(f"/home/david/sara-jarvis/Test/{dates}.txt", 'a+') as f:
        f.write(f"{role}:\n")
        f.write(content + "\n")

# Embedding operations
def text_to_embedding_ollama(text):
    model_name = "bge-m3"
    response = ollama.embed(model=model_name, input=text)
    return response

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
        embeddings = text_to_embedding_ollama(content)
        embedding = embeddings.get('embeddings')
        
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
async def send_to_ollama_api(messages, model, tools=None, stream=True):
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
            # Pretty print the request data to see exactly what's being sent
            import json
            logger.info(f"JSON request data: {json.dumps(data, indent=2)}")
            
            # Make sure tools are properly serializable
            json.dumps(data['tools'])  # This will raise an error if there's an issue
        except Exception as e:
            logger.error(f"Error in request data formatting: {str(e)}")
            # You might want to still try the request, but with simpler tools
            if 'tools' in data:
                # Try with simplified tools if there's an error
                data['tools'] = data['tools'][:3]  # Just use the first 3 tools as a test


        # Get the appropriate URL based on the model
        url = MODEL_URLS.get(local_model, MODEL_URLS["default"])
        
        logger.info(f"Sending streaming request to Ollama at {url} with model {local_model}")
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
                        
                        # Process tool calls if present
                        if 'message' in response_data and 'tool_calls' in response_data['message']:
                            logger.info(f"Tool call detected in response")
                            tool_outputs = await process_tool_calls(response_data)
                            
                            if tool_outputs:
                                logger.info(f"Processed tool outputs: {tool_outputs}")
                                
                                # Add tool outputs to conversation history
                                for tool_output in tool_outputs:
                                    messages.append(tool_output)
                                
                                # Create a combined tool results message for the assistant
                                tool_results = "\n\n".join([
                                    f"Tool: {output['name']}\nResult: {output['content']}"
                                    for output in tool_outputs
                                ])
                                
                                # Add a special system message for tool results
                                tool_system_message = {
                                    'role': 'system',
                                    'content': f"The following is information obtained by the tool call, use this in your response: {tool_results}. If a detailed report or analysis is provided, reword the response to fit the users query."
                                }
                                
                                # Create a new messages array with original messages and the tool system message
                                follow_up_messages = messages.copy()
                                # Insert the tool system message before the last user message
                                for i in range(len(follow_up_messages) - 1, -1, -1):
                                    if follow_up_messages[i]['role'] == 'user':
                                        follow_up_messages.insert(i + 1, tool_system_message)
                                        break
                                
                                # Send a follow-up request with the tool outputs and special system message
                                follow_up_data = {
                                    'model': local_model,
                                    'messages': follow_up_messages,
                                    'stream': True
                                }
                                
                                logger.info(f"Sending follow-up request with tool outputs and system message")
                                async with session.post(url, json=follow_up_data) as follow_up_response:
                                    async for follow_up_chunk in follow_up_response.content:
                                        if not follow_up_chunk:
                                            continue
                                        
                                        follow_up_str = follow_up_chunk.decode('utf-8').strip()
                                        if follow_up_str:
                                            yield format_sse_message(follow_up_str)
                            else:
                                # No tool outputs, just yield the chunk
                                yield format_sse_message(chunk_str)
                        else:
                            # For non-tool responses, yield the chunk in SSE format
                            yield format_sse_message(chunk_str)
                            
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
def manage_conversation_window(conversation_id, max_history=10):
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

# Format chunk as SSE message (Server-Sent Events)
def format_sse_message(data):
    # Convert Ollama format to OpenAI format for streaming
    try:
        ollama_data = json.loads(data)
        
        # Extract the content from Ollama response
        if 'message' in ollama_data and 'content' in ollama_data['message']:
            content = ollama_data['message']['content']
            
            # Create OpenAI-compatible delta structure
            openai_chunk = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": ollama_data.get('model', 'unknown'),
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
        
        # Handle tool calls if present
        elif 'message' in ollama_data and 'tool_calls' in ollama_data['message']:
            tool_calls = ollama_data['message']['tool_calls']
            
            openai_chunk = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": ollama_data.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": tool_calls
                        },
                        "finish_reason": None
                    }
                ]
            }
            
            return f"data: {json.dumps(openai_chunk)}\n\n"
            
        # For the final chunk
        elif ollama_data.get('done', False):
            done_chunk = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": ollama_data.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            return f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n"
        
        # For other cases, pass through as is
        return f"data: {data}\n\n"
        
    except json.JSONDecodeError:
        # If it's not valid JSON, just wrap it in an SSE format
        return f"data: {data}\n\n"

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

# OpenAI API Compatible Endpoints
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    global current_chat
    
    # Process the request
    client_messages = request.messages.copy()  # Create a copy to avoid modifying the original
    model = request.model
    stream = request.stream
    tools = request.tools or TOOL_DEFINITIONS  # Always use tools
    
    # Get conversation ID from user field if provided, otherwise generate new
    conversation_id = request.user if request.user and request.user.startswith("conv_") else f"conv_{str(uuid4())}"
    
    today = datetime.now()
    formatted_date = today.strftime("%m-%d-%Y")
    
    logger.info(f"Chat completion request: model={model}, stream={stream}, conversation_id={conversation_id}")
    
    # Extract the latest user message from the client's messages
    latest_user_message = None
    for msg in reversed(client_messages):
        if msg['role'] == 'user':
            latest_user_message = msg
            break
    
    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found in request")
    
    # Store the latest user message
    try:
        msg_content = latest_user_message['content']
        embeddings = text_to_embedding_ollama(msg_content)
        embedding = embeddings['embeddings']
        
        # Store using the message store
        message_store.store_message(
            role='user',
            content=msg_content,
            conversation_id=conversation_id,
            date=today,
            embedding=embedding
        )
        
        # Add to conversation buffer
        need_summarization = conversation_buffer.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'user',
                'content': msg_content,
                'timestamp': today.timestamp()
            }
        )
        
        # If buffer is full, schedule summarization
        if need_summarization:
            background_tasks.add_task(
                summarize_buffer_and_update_profile,
                conversation_id=conversation_id
            )
        
        # Write to file (if needed)
        write_to_file('User', msg_content, formatted_date)
    except Exception as e:
        logger.error(f"Error storing user message: {str(e)}")
        logger.exception(e)
    
    # Get conversation history (last 10 messages) - USING MESSAGE STORE INSTEAD OF REDIS
    conversation_history = message_store.get_messages_by_conversation(conversation_id, limit=10)
    
    
    # Prepare messages for the AI with system prompt and conversation history
    messages = []
    
    # Add system prompt
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get user profile
    user_profile = read_note("user_profile")
    user_profile_content = ""
    if "error" not in user_profile:
        user_profile_content = f"\n\nUser Profile:\n{user_profile.get('content', '')}"
        logger.info(f"Adding user profile ({len(user_profile_content)} chars) to system prompt")
    
    # Add system prompt with current date/time and user profile
    system_prompt_with_context = f"{SYSTEM_PROMPT}\n\nCurrent date and time: {current_datetime}{user_profile_content}"
    messages.append({'role': 'system', 'content': system_prompt_with_context})
    
    # Add memory context if available
    memory_context = await retrieve_relevant_memories(latest_user_message['content'], conversation_id)
    if memory_context:
        memory_system_message = {
            'role': 'system',
            'content': f"The following are relevant memories from past conversations that may help with this request:\n\n{memory_context}\n\nUse these memories if they're relevant to the current conversation."
        }
        messages.append(memory_system_message)
        logger.info(f"Added relevant memories to conversation context")
    
    # Add conversation history (converts Redis format to OpenAI message format)
    for msg in conversation_history:
        if 'role' in msg and 'content' in msg:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
    
    # Add background tasks
    background_tasks.add_task(maybe_summarize_conversation, conversation_id)
    
    # Update current chat
    current_chat = messages.copy()
    
    logger.info(f"Sending conversation with {len(messages)} messages to model")
    
    # Process the completion request
    if stream:
        # Streaming response
        async def stream_response():
            full_response = ""
            async for chunk in send_to_ollama_api(messages, model, tools, stream=True):
                yield chunk
                
                # Try to extract content for saving
                try:
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if 'choices' in chunk_data and chunk_data['choices'] and 'delta' in chunk_data['choices'][0]:
                        delta = chunk_data['choices'][0]['delta']
                        if 'content' in delta and delta['content']:
                            full_response += delta['content']
                except Exception as e:
                    logger.warning(f"Error extracting content from chunk: {str(e)}")
            
            # Save the complete response
            if full_response:
                logger.info(f"Saving complete response ({len(full_response)} chars)")
                # Store assistant response in the same conversation
                await async_embed_and_save(full_response, conversation_id)
                
                # Add assistant response to conversation buffer
                conversation_buffer.add_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': full_response,
                        'timestamp': datetime.now().timestamp()
                    }
                )
                
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        try:
            logger.info(f"Making non-streaming request with {len(tools)} tools")
            response = await fetch_ollama_response(messages, model, tools)
            
            # Extract content
            assistant_content = ""
            if 'message' in response and 'content' in response['message']:
                assistant_content = response['message']['content']
                logger.info(f"Received response with {len(assistant_content)} chars")
            
            # Check for tool calls in the response
            if 'message' in response and 'tool_calls' in response['message']:
                logger.info(f"Response contains tool calls, processing...")
                tool_outputs = await process_tool_calls(response)
                
                if tool_outputs:
                    logger.info(f"Tool outputs received, sending follow-up request")
                    # Add tool outputs to the conversation
                    for tool_output in tool_outputs:
                        messages.append(tool_output)
                    
                    # Make a follow-up request to get the final response
                    follow_up_response = await fetch_ollama_response(messages, model, tools)
                    
                    if 'message' in follow_up_response and 'content' in follow_up_response['message']:
                        assistant_content = follow_up_response['message']['content']
                        logger.info(f"Received follow-up response with {len(assistant_content)} chars")
            
            # Save assistant response in the same conversation
            if assistant_content:
                await async_embed_and_save(assistant_content, conversation_id)
                
                # Add assistant response to conversation buffer
                conversation_buffer.add_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': assistant_content,
                        'timestamp': datetime.now().timestamp()
                    }
                )
            
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
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(assistant_content),
                    "total_tokens": len(str(messages)) + len(assistant_content)
                }
            }
            
            return JSONResponse(content=openai_response)
            
        except Exception as e:
            logger.error(f"Error in completion: {str(e)}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Error in completion: {str(e)}")

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        embeddings_list = []
        for i, text in enumerate(input_texts):
            embedding_result = text_to_embedding_ollama(text)
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
async def receive_chat_message(data: dict = Body(...), background_tasks: BackgroundTasks = None):
    try:
        # Extract conversation_id from data if provided, otherwise generate new
        conversation_id = data.get("conversation_id", f"conv_{str(uuid4())}")
        today = datetime.now()
        formatted_date = today.strftime("%m-%d-%Y")
        message = data["message"]
        model = data["model"]
        msg_content = message["content"]
        
        logger.info(f"Legacy chat API called with conversation_id={conversation_id}")
        
        # Get embeddings
        embeddings = text_to_embedding_ollama(msg_content)
        embedding = embeddings['embeddings']
        
        # Store in Redis
        message_store.store_message(
            role='user',
            content=msg_content,
            conversation_id=conversation_id,
            date=today,
            embedding=embedding
        )
        
        # Add to conversation buffer
        need_summarization = conversation_buffer.add_message(
            conversation_id=conversation_id,
            message={
                'role': 'user',
                'content': msg_content,
                'timestamp': today.timestamp()
            }
        )
        
        # If buffer is full, schedule summarization
        if need_summarization and background_tasks is not None:
            background_tasks.add_task(
                summarize_buffer_and_update_profile,
                conversation_id=conversation_id
            )
        
        # Write to file
        write_to_file('User', msg_content, formatted_date)
        
        # Get conversation history (last 10 messages)
        conversation_history = manage_conversation_window(conversation_id, max_history=10)
        
        # Prepare messages for the model
        messages = []
        
        # Get user profile
        user_profile = read_note("user_profile")
        user_profile_content = ""
        if "error" not in user_profile:
            user_profile_content = f"\n\nUser Profile:\n{user_profile.get('content', '')}"
            logger.info(f"Adding user profile ({len(user_profile_content)} chars) to system prompt")
        
        # Add system prompt with user profile
        system_prompt_with_context = f"{SYSTEM_PROMPT}{user_profile_content}"
        messages.append({'role': 'system', 'content': system_prompt_with_context})
        
        # Add conversation history
        for msg in conversation_history:
            if 'role' in msg and 'content' in msg:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Add memory context if available
        memory_context = await retrieve_relevant_memories(msg_content, conversation_id)
        if memory_context:
            memory_system_message = {
                'role': 'system',
                'content': f"The following are relevant memories from past conversations that may help with this request:\n\n{memory_context}\n\nUse these memories if they're relevant to the current conversation."
            }
            # Insert after system prompt but before conversation
            messages.insert(1, memory_system_message)
            logger.info(f"Added relevant memories to conversation context")
        
        # Update current_chat
        current_chat.clear()
        current_chat.extend(messages)
        
        logger.info(f"Sending {len(messages)} messages to model in legacy endpoint")

        async def stream_response():
            full_response = ""
            async for chunk in send_to_ollama_api(messages, model, TOOL_DEFINITIONS, stream=True):
                yield chunk
                
                # Try to extract content for saving
                try:
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if 'choices' in chunk_data and chunk_data['choices'] and 'delta' in chunk_data['choices'][0]:
                        delta = chunk_data['choices'][0]['delta']
                        if 'content' in delta and delta['content']:
                            full_response += delta['content']
                except Exception as e:
                    logger.warning(f"Error extracting content from chunk: {str(e)}")
            
            # Save the complete response
            if full_response:
                logger.info(f"Saving complete response ({len(full_response)} chars)")
                await async_embed_and_save(full_response, conversation_id)
                
                # Add assistant response to conversation buffer
                conversation_buffer.add_message(
                    conversation_id=conversation_id,
                    message={
                        'role': 'assistant',
                        'content': full_response,
                        'timestamp': datetime.now().timestamp()
                    }
                )
                
                # Return the conversation ID in the response
                yield f"\ndata: {{\"conversation_id\": \"{conversation_id}\"}}\n\n"
            
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Error in legacy chat endpoint: {str(e)}")
        logger.exception(e)
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
                "direct_keys_sample": [k for k in direct_keys[:10]],
                "sample_data": sample_data
            }
        
        return {
            "status": "success",
            "message_count": len(messages),
            "messages": messages[:5],  # Return first 5 messages
            "direct_keys_count": len(direct_keys),
            "direct_keys_sample": [k for k in direct_keys[:10]],
            "sample_data": sample_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint that also checks Redis connection"""
    logger.info("Health check endpoint called")
    
    # Check Redis connection
    redis_status = "connected" if message_store.ping() else "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_status,
            "server": "online"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 7009")
    uvicorn.run(app, host="0.0.0.0", port=7009, log_level="info")

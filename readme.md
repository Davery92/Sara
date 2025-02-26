# OpenAI-Compatible API Server

This server provides an OpenAI-compatible API interface to local LLM models using Ollama. It includes features for vector storage, conversation history, RAG (Retrieval-Augmented Generation), and various tools like web search, note management, and memory features. The server can be customized to work with different Ollama models and deployment configurations.

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API endpoints
- **Local LLM Integration**: Run models locally through Ollama
- **Vector Storage**: Store and retrieve conversation history using Redis
- **RAG Support**: Enhance responses with relevant document retrieval
- **Conversation Memory**: Long-term memory for personalized interactions
- **Tool Integration**: Web search, notes management, and more
- **Streaming Responses**: Support for streaming completions
- **Model Mapping**: Map OpenAI model names to local models

## Prerequisites

- Python 3.8+
- Docker (for Redis)
- Ollama with required models installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/openai-compatible-server.git
cd openai-compatible-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Redis using Docker:
```bash
docker run --name redis-vector -p 6379:6379 -d redis/redis-stack:latest
```

4. Ensure Ollama is installed with your required models:
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.3
ollama pull mistral-small
ollama pull llama3.2
ollama pull llama3.1
```

## Project Structure

pip install requirements.txt

## Directory Structure Setup

```bash
# Assuming you're in the project root directory
mkdir -p ./data/notes
touch ./data/system_prompt.txt
touch ./data/tool_system_prompt.txt
touch ./data/core_memories.txt
```

### System Prompt

Edit `./data/system_prompt.txt` with your preferred system prompt. This sets the personality and capabilities of the assistant.

### Tool System Prompt

Edit `./data/tool_system_prompt.txt` with your preferred tool system prompt template. This defines how tool results are formatted.

### Model Configuration

Modify the model mappings in the code to match your setup:

```python
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
    # Add or remove models as needed
]

# URLs for different models
MODEL_URLS = {
    "llama3.3": "http://x.x.x.x:11434/api/chat",
    "llama3.2": "http://localhost:11434/api/chat",
    "llama3.1": "http://localhost:11434/api/chat",
    "mistral-small": "http://localhost:11434/api/chat",
    "default": "http://localhost:11434/api/chat"
}
```

### Custom Configuration

You need to update the hardcoded paths in the code to match your environment. The main file paths to update are:

```python
# Logging path (line ~45-46)
logging.FileHandler("/home/david/sara-jarvis/Test/openai_server.log")
# Change to:
logging.FileHandler("./logs/openai_server.log")

# Notes directory (line ~92)
NOTES_DIRECTORY = "/home/david/Sara/notes"
# Change to:
NOTES_DIRECTORY = "./data/notes"

# Core memory file (line ~99)
CORE_MEMORY_FILE = "/home/david/Sara/core_memories.txt"
# Change to:
CORE_MEMORY_FILE = "./data/core_memories.txt"

# System prompt file path (line ~509)
prompt_file = "/home/david/Sara/system_prompt.txt"
# Change to:
prompt_file = "./data/system_prompt.txt"

# Tool system prompt file path (line ~533)
prompt_file = "/home/david/Sara/tool_system_prompt.txt"
# Change to:
prompt_file = "./data/tool_system_prompt.txt"
```

Make sure to create the `logs` directory in your project folder:
```bash
mkdir -p ./logs
```

## Customizing for Your Setup

### 1. Model Mapping

To change which local models map to OpenAI model names, edit the `MODEL_MAPPING` dictionary:

```python
MODEL_MAPPING = {
    "gpt-4": "your-preferred-model", 
    "gpt-3.5-turbo": "your-other-model",
    # Add more mappings as needed
    "default": "your-default-model" 
}
```

### 2. Available Models

Update the `AVAILABLE_MODELS` list to include the models you have installed through Ollama:

```python
AVAILABLE_MODELS = [
    "your-model-1:latest",
    "your-model-2:latest",
    # Add more models as needed
]
```

### 3. Model URLs

If you're running Ollama on different machines or ports, update the `MODEL_URLS` dictionary:

```python
MODEL_URLS = {
    "your-model-1": "http://ip-address-1:11434/api/chat",
    "your-model-2": "http://ip-address-2:11434/api/chat",
    "default": "http://localhost:11434/api/chat"
}
```

### 4. File Paths

Update the file paths throughout the code to match your directory structure:

```python
# Update these paths to match your environment
NOTES_DIRECTORY = "./data/notes"
CORE_MEMORY_FILE = "./data/core_memories.txt"
logging.FileHandler("./logs/openai_server.log")
```

## Usage

### Starting the Server

Run the server with:

```bash
uvicorn main:app --host 0.0.0.0 --port 7009 --reload
```

### API Endpoints

The server provides these main endpoints:

- `/v1/chat/completions` - OpenAI-compatible chat completions
- `/v1/embeddings` - Generate embeddings
- `/v1/models` - List available models
- `/api/chat` - Legacy chat endpoint
- `/health` - Health check endpoint
- `/v1/conversations` - Manage conversations
- `/rag/*` - RAG API endpoints

### Example Request to Chat Completions

```python
import requests
import json

url = "http://localhost:7009/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

## Tools and Features

### Available Tools

The server includes several built-in tools:

- `send_message`: Formulate response thinking
- `search_perplexica`: Web search integration
- `append_core_memory`: Add important information to memory
- `rewrite_core_memories`: Update the entire memory set
- `create_note`, `read_note`, `append_note`, `delete_note`, `list_notes`: Note management
- RAG integration for document retrieval

### Conversation Management

- Conversations are stored in Redis
- Access conversation history via `/v1/conversations` endpoints
- Embeddings for semantic search

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**:
   - Verify Redis Docker container is running: `docker ps | grep redis-vector`
   - Check Redis connection parameters in the code
   
2. **Ollama Model Issues**:
   - Verify models are installed: `ollama list`
   - Check Ollama is running: `curl http://localhost:11434/api/version`
   
3. **API Endpoint Errors**:
   - Check server logs for detailed error messages
   - Verify request format matches OpenAI API specifications

### Logs

Check the server logs for detailed information:

```bash
tail -f ./logs/openai_server.log
```

## Code Modification Guide

### Updating Folder Locations

To change all hardcoded paths in the codebase, you should search for and replace these patterns:

1. Find all instances of `/home/david/sara-jarvis/Test/` and replace with `./logs/`
2. Find all instances of `/home/david/Sara/` and replace with `./data/`

Here's a bash command to find all paths that might need changing:

```bash
grep -r "/home/" . --include="*.py"
```

You can make these replacements using your code editor's search and replace feature with regex support.

### Steps to Update the Code

1. **Create a backup of your original code**:
   ```bash
   cp main.py main.py.backup
   ```

2. **Update import paths** if needed:
   ```python
   # Look for lines like:
   from modules.perplexica_module import PerplexicaClient
   
   # If module locations have changed, update accordingly
   ```

3. **Update file handler locations**:
   ```python
   # Find logging setup (around line 41-47)
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.StreamHandler(sys.stdout),
           logging.FileHandler("./logs/openai_server.log")  # Updated path
       ]
   )
   ```

4. **Update data directories**:
   ```python
   # Find NOTES_DIRECTORY definition (around line 92)
   NOTES_DIRECTORY = "./data/notes"  # Updated path
   
   # Find CORE_MEMORY_FILE definition (around line 99)
   CORE_MEMORY_FILE = "./data/core_memories.txt"  # Updated path
   ```

5. **Update system prompt loading functions**:
   ```python
   # Find load_system_prompt function (around line 509)
   def load_system_prompt():
       """Load the system prompt from a file"""
       prompt_file = "./data/system_prompt.txt"  # Updated path
       # ... rest of function ...
   
   # Find load_tool_system_prompt function (around line 533)
   def load_tool_system_prompt():
       """Load the tool system prompt template from a file"""
       prompt_file = "./data/tool_system_prompt.txt"  # Updated path
       # ... rest of function ...
   ```


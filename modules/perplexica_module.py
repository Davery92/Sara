import json
import requests
import sys
import uuid
import time
import re
from typing import List, Tuple, Dict, Any, Optional


class PerplexicaClient:
    """Client for the Perplexica API with Ollama integration."""

    def __init__(
        self, 
        api_url: str,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_model: str = "llama3.2:latest",
        provider: str = "ollama",
        embedding_provider: str = "ollama",
        embedding_model: str = "bge-m3:latest",
        focus_mode: str = "webSearch",
        optimization_mode: str = "balanced",
        debug: bool = False
    ):
        """
        Initialize the Perplexica client.

        Args:
            api_url: The base URL for the API
            ollama_base_url: Base URL for Ollama API, default is local instance
            ollama_model: Ollama model to use (llama3.2:latest, mistral, etc.)
            provider: Provider for the chat model
            embedding_provider: Provider for embeddings
            embedding_model: Model for embeddings
            focus_mode: Search focus mode
            optimization_mode: 'speed', 'balanced', or 'quality'
            debug: Enable debug output
        """
        self.api_url = api_url.rstrip('/')
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.provider = provider
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.focus_mode = focus_mode
        self.optimization_mode = optimization_mode
        self.debug = debug
        self.history: List[Tuple[str, str]] = []
        self.chat_id = str(uuid.uuid4())

    def send_message(self, query: str) -> Dict[str, Any]:
        """
        Send a message to the API and get a streaming response.

        Args:
            query: The user query to send

        Returns:
            Dict containing the response message and sources
        """
        # Format history for the API
        formatted_history = []
        for role, content in self.history:
            formatted_history.append([role, content])
            
        # Create a unique message ID
        message_id = str(uuid.uuid4())
        
        payload = {
            "message": {
                "chatId": self.chat_id,
                "messageId": message_id,
                "content": query
            },
            "optimizationMode": self.optimization_mode,
            "focusMode": self.focus_mode,
            "chatModel": {
                "provider": self.provider,
                "name": self.ollama_model
            },
            "embeddingModel": {
                "provider": self.embedding_provider,
                "name": self.embedding_model
            },
            "history": formatted_history,
            "files": []
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }

        if self.debug:
            print(f"Sending request to: {self.api_url}/api/chat")
            print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            # This will be a streaming response
            response = requests.post(
                f"{self.api_url}/api/chat",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )

            if response.status_code != 200:
                error_message = f"Error: {response.status_code} - {response.text}"
                print(error_message, file=sys.stderr)
                return {"message": error_message, "sources": []}
            
            # Process the streaming response manually
            full_message = ""
            sources = []
            ai_message_id = None
            
            # Regular expression to match JSON objects in the event stream
            json_pattern = re.compile(r'({.*?})(?:\n|$)')
            
            buffer = ""
            # Only show this once at the beginning of the response
            print("\nAI: ", end="", flush=True)
            
            # Process the stream in chunks
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                    
                chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                buffer += chunk_str
                
                # Only print debug if requested
                if self.debug and '{"type":' in chunk_str:
                    print(f"\n[Debug] Raw chunk: {chunk_str}", file=sys.stderr)
                
                # Find all JSON objects in the current buffer
                matches = json_pattern.finditer(buffer)
                for match in matches:
                    try:
                        json_str = match.group(1)
                        if self.debug:
                            print(f"\n[Debug] Found JSON: {json_str}", file=sys.stderr)
                            
                        data = json.loads(json_str)
                        
                        if self.debug:
                            print(f"\n[Debug] Event type: {data['type']}", file=sys.stderr)
                            
                        if data.get('type') == 'message':
                            if ai_message_id is None:
                                ai_message_id = data.get('messageId', '')
                            fragment = data.get('data', '')
                            full_message += fragment
                            # Print ONLY the fragment text as it arrives
                            print(fragment, end="", flush=True)
                            
                        elif data.get('type') == 'sources':
                            sources = data.get('data', [])
                            if self.debug:
                                print("\n[Debug] Sources:", sources, file=sys.stderr)
                                
                        elif data.get('type') == 'error':
                            error_msg = data.get('data', 'Unknown error')
                            print(f"\nError: {error_msg}", file=sys.stderr)
                            
                        elif data.get('type') == 'messageEnd':
                            # Add this exchange to history
                            self.history.append(("human", query))
                            self.history.append(("ai", full_message))
                            
                            # Return full response without additional output
                            return {
                                "message": full_message,
                                "messageId": ai_message_id,
                                "sources": sources
                            }
                            
                    except json.JSONDecodeError as e:
                        if self.debug:
                            print(f"\n[Debug] JSON decode error: {e} for: {match.group(1)}", file=sys.stderr)
                    except Exception as e:
                        if self.debug:
                            print(f"\n[Debug] Error processing event: {str(e)}", file=sys.stderr)
                
                # Keep only the unprocessed part of the buffer
                last_newline = buffer.rfind('\n')
                if last_newline >= 0:
                    buffer = buffer[last_newline + 1:]
            
            # If we get here, the stream ended without a messageEnd event
            if self.debug:
                print("\n[Debug] Response complete - no messageEnd event received", file=sys.stderr)
            
            # Add this exchange to history if we got any content
            if full_message:
                self.history.append(("human", query))
                self.history.append(("ai", full_message))
            
            return {
                "message": full_message,
                "messageId": ai_message_id,
                "sources": sources
            }
                
        except requests.exceptions.Timeout:
            print("\nError: Request timed out after 60 seconds", file=sys.stderr)
            return {"message": "Request timed out", "sources": []}
        except Exception as e:
            print(f"\nError: {str(e)}", file=sys.stderr)
            return {"message": f"Request failed: {str(e)}", "sources": []}

    def search(self, 
               query: str, 
               focus_mode: Optional[str] = None, 
               optimization_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a search using the Perplexica API.
        
        Args:
            query: The search query
            focus_mode: Optional override for the search focus mode
            optimization_mode: Optional override for the optimization mode
            
        Returns:
            Dict containing the search results and sources
        """
        search_focus = focus_mode if focus_mode else self.focus_mode
        search_optimization = optimization_mode if optimization_mode else self.optimization_mode
        
        # Validate focus mode
        valid_focus_modes = {
            "webSearch", "academicSearch", "writingAssistant",
            "wolframAlphaSearch", "youtubeSearch", "redditSearch"
        }
        if search_focus not in valid_focus_modes:
            raise ValueError(f"Invalid focus mode. Must be one of: {valid_focus_modes}")

        # Validate optimization mode
        if search_optimization not in {"speed", "balanced", "quality"}:
            raise ValueError("optimization_mode must be one of: 'speed', 'balanced', or 'quality'")
        
        payload = {
            "focusMode": search_focus,
            "optimizationMode": search_optimization,
            "query": query,
            "chatModel": {
                "provider": self.provider,
                "name": self.ollama_model  # Changed from "model" to "name" to match send_message format
            },
            "embeddingModel": {
                "provider": self.embedding_provider,
                "name": self.embedding_model
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.debug:
            print(f"Sending search request to: {self.api_url}/api/search")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
        try:
            response = requests.post(
                f"{self.api_url}/api/search",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Format the response if debug is enabled
            if self.debug:
                print("\nSearch result:")
                print(f"Answer: {result.get('message', '')}")
                print("\nSources:")
                if "sources" in result:
                    for idx, source in enumerate(result["sources"], 1):
                        metadata = source.get("metadata", {})
                        print(f"{idx}. {metadata.get('title', 'Untitled')}: {metadata.get('url', 'No URL')}")
            
            return result
            
        except requests.exceptions.Timeout:
            error_message = "Search timed out after 60 seconds"
            if self.debug:
                print(f"\nError: {error_message}", file=sys.stderr)
            return {"message": error_message, "sources": []}
        except requests.exceptions.RequestException as e:
            error_message = f"Search failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_message += f"\nDetails: {error_detail}"
                except:
                    error_message += f"\nStatus code: {e.response.status_code}"
            
            if self.debug:
                print(f"\nError: {error_message}", file=sys.stderr)
            return {"message": error_message, "sources": []}
        except Exception as e:
            error_message = f"Unexpected error during search: {str(e)}"
            if self.debug:
                print(f"\nError: {error_message}", file=sys.stderr)
            return {"message": error_message, "sources": []}

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
        # Generate a new chat ID for a fresh conversation
        self.chat_id = str(uuid.uuid4())
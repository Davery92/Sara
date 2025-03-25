import requests
import json
from typing import List, Tuple, Optional, Dict, Any

class PerplexicaClient:
    def __init__(self, api_url: str, **kwargs):
        """
        Initialize with enhanced validation.
        
        Args:
            api_url: URL of the Perplexica API endpoint
            kwargs: See below for other parameters
        """
        self.api_url = api_url
        self.focus_mode = kwargs.get('focus_mode', 'all')
        self.optimization_mode = kwargs.get('optimization_mode', 'balanced')
        self.chat_model_provider = kwargs.get('chat_model_provider', 'ollama')
        self.chat_model_name = kwargs.get('chat_model_name', 'llama3.2:latest')
        self.embedding_model_provider = kwargs.get('embedding_model_provider')
        self.embedding_model_name = kwargs.get('embedding_model_name')
        self.history: List[Tuple[str, str]] = []
        
        # Validate initialization parameters
        if self.optimization_mode not in ['speed', 'balanced']:
            raise ValueError("optimization_mode must be either 'speed' or 'balanced'")
        if not isinstance(self.chat_model_provider, str) or not self.chat_model_provider:
            raise ValueError("chat_model_provider must be a non-empty string")

    def _validate_history(self, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Ensure history is in the correct format."""
        validated = []
        for i, msg in enumerate(history):
            if not isinstance(msg, (list, tuple)) or len(msg) != 2:
                raise ValueError(f"Message {i} must be a tuple of (role, content)")
            role, content = msg
            if role not in ['human', 'ai']:
                raise ValueError(f"Message {i} role must be 'human' or 'ai'")
            if not isinstance(content, str):
                raise ValueError(f"Message {i} content must be a string")
            validated.append((role, content))
        return validated

    def chat(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """Enhanced version with better error handling."""
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
                
            history = self._validate_history(history if history is not None else self.history)
            
            request_body = {
                "optimizationMode": self.optimization_mode,
                "focusMode": self.focus_mode,
                "query": query,
                "history": history,
                "chatModel": {
                    "provider": self.chat_model_provider,
                    "name": self.chat_model_name
                }
            }

            if self.embedding_model_provider and self.embedding_model_name:
                request_body["embeddingModel"] = {
                    "provider": self.embedding_model_provider,
                    "name": self.embedding_model_name
                }

            response = requests.post(
                self.api_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=60  # 60-second timeout
            )

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get('message', error_detail)
                except:
                    pass
                raise Exception(f"API error {response.status_code}: {error_detail}")

            message = ""
            sources = []
            
            for line in response.iter_lines():
                if line:
                    try:
                        parsed = json.loads(line.decode('utf-8'))
                        if parsed.get('type') == 'response':
                            message += parsed.get('data', '')
                        elif parsed.get('type') == 'sources':
                            sources = parsed.get('data', [])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line: {line}, error: {str(e)}")
                        continue

            # Update history only after successful completion
            self.history.append(("human", query))
            self.history.append(("ai", message))

            return {
                "message": message,
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "sources": []
            }

    def reset_history(self):
        """Reset conversation history."""
        self.history = []


# Example usage with error handling
if __name__ == "__main__":
    try:
        client = PerplexicaClient(
            api_url="http://localhost:3000/api/chat",
            focus_mode="all",
            chat_model_name="llama3.2:latest"
        )
        
        # First query
        response = client.chat("Explain quantum computing in simple terms")
        if response["status"] == "success":
            print("Response:", response["message"])
            print("Sources:", response["sources"])
        else:
            print("Error:", response["message"])
        
        # Follow-up with bad history to test validation
        try:
            bad_response = client.chat("What about Germany?", history=[("wrong-role", 123)])
        except ValueError as e:
            print(f"Caught expected validation error: {str(e)}")
            
    except Exception as e:
        print(f"Initialization error: {str(e)}")
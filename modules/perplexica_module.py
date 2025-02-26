from dataclasses import dataclass
import requests
from typing import Optional, Dict, List, Tuple

@dataclass
class ModelConfig:
    provider: str
    model: str
    custom_openai_base_url: Optional[str] = None
    custom_openai_key: Optional[str] = None

class PerplexicaClient:
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url.rstrip('/')
        self.search_endpoint = f"{self.base_url}/api/search"
        
    def search(self,
              query: str,
              focus_mode: str = "webSearch",
              optimization_mode: str = "balanced") -> Dict:
        """
        Simplified search function for use as a tool.
        """
        valid_focus_modes = {
            "webSearch", "academicSearch", "writingAssistant",
            "wolframAlphaSearch", "youtubeSearch", "redditSearch"
        }
        if focus_mode not in valid_focus_modes:
            raise ValueError(f"Invalid focus mode. Must be one of: {valid_focus_modes}")

        if optimization_mode not in {"speed", "balanced"}:
            raise ValueError("optimization_mode must be either 'speed' or 'balanced'")

        # Default chat model configuration for Ollama
        chat_model = ModelConfig(
            provider="ollama",
            model="llama3.2:latest",
            custom_openai_base_url="http://host.docker.internal:11434"
        )

        payload = {
            "focusMode": focus_mode,
            "optimizationMode": optimization_mode,
            "query": query,
            "chatModel": {
                "provider": chat_model.provider,
                "model": chat_model.model,
                "customOpenAIBaseURL": chat_model.custom_openai_base_url
            }
        }

        try:
            response = requests.post(self.search_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Format the response nicely
            formatted_response = f"Answer: {result['message']}\n\nSources:\n"
            if "sources" in result:
                for source in result["sources"]:
                    formatted_response += f"- {source['metadata']['title']}: {source['metadata']['url']}\n"
            
            return formatted_response
            
        except requests.exceptions.RequestException as e:
            error_message = f"Search failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_message += f"\nDetails: {error_detail}"
                except:
                    error_message += f"\nStatus code: {e.response.status_code}"
            return error_message
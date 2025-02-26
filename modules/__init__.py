# modules/__init__.py
"""
RAG system modules package
"""
# Import key components for easy access
from .rag_module import RAGManager
from .rag_api import rag_router
from .rag_web_interface import integrate_web_interface
from .redis_client import RedisClient
from .perplexica_module import PerplexicaClient
from .rag_integration import integrate_rag_with_server, update_system_prompt_with_rag_info
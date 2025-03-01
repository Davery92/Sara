"""
Integration of RAG module with the main FastAPI server
"""

from fastapi import FastAPI
import logging
from .rag_module import RAGManager
from .rag_api import rag_router
from .rag_web_interface import integrate_web_interface

# Configure logging
logger = logging.getLogger("rag-integration")

# RAG tool definition for OpenAI API
RAG_TOOL_DEFINITION = {
    'type': 'function',
    'function': {
        'name': 'search_knowledge_base',
        'description': 'Search the knowledge base for information related to the user\'s query. Use this tool when the user asks questions that might be answered using the stored documents.',
        'parameters': {
            'type': 'object',
            'required': ['query'],
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query'
                },
                'top_k': {
                    'type': 'integer',
                    'description': 'Number of results to return',
                    'default': 3
                },
                'filter_tags': {
                    'type': 'array',
                    'description': 'Optional tags to filter documents by',
                    'items': {
                        'type': 'string'
                    }
                }
            },
        },
    },
}

def search_knowledge_base(query: str, top_k: int = 3, filter_tags: list = None):
    try:
        # Import here to avoid circular imports
        from .neo4j_rag_integration import get_neo4j_rag_manager
        
        # Initialize RAG manager if needed
        global rag_manager
        if not rag_manager:
            rag_manager = get_neo4j_rag_manager()
        
        # Perform search
        results = rag_manager.search(
            query=query,
            top_k=top_k,
            filter_tags=filter_tags
        )
        
        # Format results
        if results:
            context_text = "Here is relevant information found in the knowledge base:\n\n"
            
            for i, result in enumerate(results, 1):
                title = result.get("title", "Untitled document")
                text = result.get("text", "").strip()
                context_text += f"[{i}] From '{title}':\n{text}\n\n"
                
            return context_text
        else:
            return "No relevant information found in the knowledge base."
        
    except Exception as e:
        logger.error(f"Error in search_knowledge_base: {e}")
        return f"Error searching knowledge base: {str(e)}"

# Initialize RAG manager (will be lazy-loaded)
rag_manager = None

def integrate_rag_with_server(app: FastAPI, available_tools: dict, tool_definitions: list):
    """
    Integrate RAG with the main server
    """
    try:
        # Include RAG router
        app.include_router(rag_router)
        logger.info("RAG API endpoints added to server")
        
        # Add RAG tool to available tools
        available_tools['search_knowledge_base'] = search_knowledge_base
        logger.info("RAG tool added to available tools")
        
        # Add RAG tool definition
        tool_definitions.append(RAG_TOOL_DEFINITION)
        logger.info("RAG tool definition added")
        
        # Add health check for RAG
        @app.get("/rag/health")
        async def rag_health():
            return {"status": "healthy", "service": "RAG"}
        
        # Integrate web interface
        integrate_web_interface(app)
        logger.info("RAG web interface integrated")
        
        logger.info("RAG integration complete")
        
    except Exception as e:
        logger.error(f"Error integrating RAG with server: {e}")
        raise

def update_system_prompt_with_rag_info(system_prompt: str) -> str:
    """
    Update system prompt to include RAG capabilities
    """
    try:
        # Don't add if already present
        if "Knowledge Base Access:" in system_prompt:
            return system_prompt
            
        # Add RAG capabilities to system prompt
        rag_info = """
Knowledge Base Access:
You have access to a knowledge base containing documents. When the user asks something that might be in these documents, use the search_knowledge_base tool to find relevant information before responding. Always use the knowledge base for questions about specific topics, policies, or custom information that the user might have uploaded.
"""
        
        return system_prompt + rag_info
        
    except Exception as e:
        logger.error(f"Error updating system prompt with RAG info: {e}")
        return system_prompt
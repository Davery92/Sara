import asyncio
import logging
from fastapi import APIRouter
from typing import List, Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
import json

# Local imports (assuming the conversations_processor.py file is in the modules directory)
from modules.conversation_processor import get_conversation_monitor

# Configure logging
logger = logging.getLogger("conversation-integration")

# Create a middleware class that can be added at app creation time
class ConversationProcessingMiddleware(BaseHTTPMiddleware):
    """Middleware to capture chat completion requests for processing"""
    
    def __init__(self, app, max_messages=40):
        """Initialize with app and configuration"""
        super().__init__(app)
        # Get conversation monitor
        self.monitor = get_conversation_monitor(max_messages=max_messages)
        logger.info(f"Initialized conversation processing middleware (threshold: {max_messages} messages)")
    
    async def dispatch(self, request, call_next):
        """Process the request and capture chat completions"""
        # Only process for chat completions endpoint
        if request.url.path == "/v1/chat/completions" and request.method == "POST":
            try:
                # Get a copy of the request body before it's consumed
                body_bytes = await request.body()
                
                # Create a modified request with the same body to avoid consumption issues
                request = await self.set_body(request, body_bytes)
                
                # Store a copy of the body for processing after the response
                body_copy = body_bytes
            except Exception as e:
                logger.error(f"Error copying request body: {e}")
                body_copy = None
        else:
            body_copy = None
        
        # Process the request normally
        response = await call_next(request)
        
        # Process the stored body copy after response if available
        if body_copy:
            try:
                # Process in a background task to avoid blocking
                asyncio.create_task(self._process_chat_completion(body_copy))
            except Exception as e:
                logger.error(f"Error in conversation processing middleware: {e}")
        
        return response
    
    async def set_body(self, request, body):
        """Set the request body so it can be read again"""
        async def receive():
            return {"type": "http.request", "body": body}
        
        request._receive = receive
        return request
    
    async def _process_chat_completion(self, body_bytes):
        """Process a chat completion request (runs in background)"""
        try:
            body = json.loads(body_bytes)
            
            # Extract messages
            messages = body.get("messages", [])
            
            # Process if we have messages
            if messages:
                await self.monitor.process_if_needed(messages)
        except Exception as e:
            logger.error(f"Error processing chat completion: {e}")


def add_conversation_middleware(app, max_messages=40):
    """
    Add middleware for conversation processing during app creation
    
    This function should be called before app.add_middleware() becomes unavailable
    """
    # Add the middleware
    app.add_middleware(ConversationProcessingMiddleware, max_messages=max_messages)
    logger.info(f"Added conversation processing middleware (threshold: {max_messages} messages)")


def integrate_conversation_memory_with_server(app, message_store=None, max_messages=40, add_middleware=False):
    """
    Integrate the conversation memory system with the server
    
    Args:
        app: FastAPI application
        message_store: Optional message store for fetching messages
        max_messages: Number of messages before processing occurs
        
    Returns:
        The conversation monitor instance
    """
    # Get conversation monitor
    monitor = get_conversation_monitor(max_messages=max_messages)
    
    # Create router for conversation memory endpoints
    router = APIRouter()
    
    @router.post("/process")
    async def process_current_conversation():
        """Process the current conversation immediately"""
        # Use message store to get current conversation if available
        messages = []
        
        if message_store:
            try:
                messages = message_store.get_messages_by_conversation("primary_conversation")
            except Exception as e:
                logger.error(f"Error getting messages from store: {e}")
                messages = []
        
        # If we couldn't get messages from store, try to access MESSAGE_HISTORY
        if not messages:
            try:
                # Try to import MESSAGE_HISTORY
                import sys
                # Look through all modules to find MESSAGE_HISTORY
                for module_name, module in sys.modules.items():
                    if hasattr(module, 'MESSAGE_HISTORY'):
                        messages = module.MESSAGE_HISTORY
                        logger.info(f"Found MESSAGE_HISTORY in module {module_name}")
                        break
            except Exception as e:
                logger.error(f"Error accessing MESSAGE_HISTORY: {e}")
                messages = []
        
        # Process if we have messages
        if messages:
            result = await monitor.process_if_needed(messages, "primary_conversation", force=True)
            return {
                "status": "success", 
                "message_count": len(messages),
                "processing_result": result
            }
        else:
            return {
                "status": "error",
                "reason": "No messages found in current conversation"
            }
    
    @router.get("/status")
    async def get_memory_status():
        """Get status of the conversation memory system"""
        # Count total conversations in Neo4j
        count = 0
        try:
            conversations = await monitor.processor.list_conversations(limit=1)
            count = conversations.get("total", 0)
        except Exception as e:
            logger.error(f"Error getting conversation count: {e}")
        
        return {
            "status": "active",
            "max_messages": monitor.max_messages,
            "conversation_count": count,
            "message_counts": monitor.message_counts,
            "last_processed": {
                k: v.isoformat() for k, v in monitor.last_processed.items()
            } if hasattr(monitor, 'last_processed') else {}
        }
    
    # Include router
    app.include_router(router, prefix="/v1/memory", tags=["memory"])
    
    # Add event handler to process message history on server startup
    @app.on_event("startup")
    async def process_history_on_startup():
        """Process conversation history when server starts"""
        # Use message store to get current conversation if available
        if message_store:
            try:
                messages = message_store.get_messages_by_conversation("primary_conversation")
                
                if messages and len(messages) >= 5:  # Only process if we have meaningful history
                    logger.info(f"Processing conversation history with {len(messages)} messages on startup")
                    result = await monitor.process_if_needed(messages, "primary_conversation", force=True)
                    logger.info(f"Conversation processing result: {result}")
            except Exception as e:
                logger.error(f"Error processing history on startup: {e}")
    
    # Instead of trying to hook the function, add a background task 
    # that periodically checks for new messages
    
    async def check_messages_periodically():
        """Periodically check if conversations need processing"""
        while True:
            try:
                # Wait for a bit to avoid excessive checking
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get current messages from store if available
                current_messages = []
                
                if message_store:
                    try:
                        current_messages = message_store.get_messages_by_conversation("primary_conversation")
                    except Exception as e:
                        logger.error(f"Error getting messages from store: {e}")
                
                # If no messages from store, try to find MESSAGE_HISTORY
                if not current_messages:
                    try:
                        # Try to find MESSAGE_HISTORY in loaded modules
                        import sys
                        for module_name, module in sys.modules.items():
                            if hasattr(module, 'MESSAGE_HISTORY'):
                                current_messages = module.MESSAGE_HISTORY
                                break
                    except Exception as e:
                        logger.error(f"Error accessing MESSAGE_HISTORY: {e}")
                
                # Process if we have enough messages
                if current_messages and len(current_messages) >= monitor.max_messages // 2:
                    # Only log if we're going to process
                    if len(current_messages) >= monitor.max_messages:
                        logger.info(f"Checking messages: found {len(current_messages)} messages")
                    
                    # Check if processing is needed (don't force it)
                    await monitor.process_if_needed(current_messages, "primary_conversation")
            
            except Exception as e:
                logger.error(f"Error in periodic message check: {e}")
                await asyncio.sleep(60)  # Wait longer after an error
    
    # Start the background task
    asyncio.create_task(check_messages_periodically())
    
    # Use our own implementation without calling the other integration function
    # This avoids double middleware registration
    
    # Create a router for conversation endpoints
    processor_router = APIRouter()
    
    # Add endpoint to process a conversation
    @processor_router.post("/process")
    async def process_conversation(data: dict):
        """Force processing of a conversation"""
        messages = data.get("messages", [])
        conversation_id = data.get("conversation_id", "primary_conversation")
        
        if not messages:
            return {"error": "No messages provided"}
        
        result = await monitor.processor.process_conversation(messages, conversation_id)
        return result or {"status": "no_processing_needed"}
    
    # Add endpoint to list processed conversations
    @processor_router.get("/list")
    async def list_conversations(limit: int = 10, offset: int = 0, 
                           start_date: str = None, end_date: str = None):
        """List processed conversations"""
        return monitor.processor.list_conversations(
            limit=limit, 
            offset=offset,
            filter_date_start=start_date,
            filter_date_end=end_date
        )
    
    # Add endpoint to get a specific conversation
    @processor_router.get("/{conversation_id}")
    async def get_conversation(conversation_id: str):
        """Get a specific conversation by ID"""
        return monitor.processor.get_conversation_by_id(conversation_id)
    
    # Add endpoint to search conversations
    @processor_router.post("/search")
    async def search_conversations(data: dict):
        """Search conversations"""
        query = data.get("query", "")
        limit = data.get("limit", 5)
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        
        if not query:
            return {"error": "No query provided"}
        
        results = await monitor.processor.search_conversations(
            query=query,
            limit=limit,
            filter_date_start=start_date,
            filter_date_end=end_date
        )
        
        return {"results": results}
    
    # Add endpoint to delete a conversation
    @processor_router.delete("/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        """Delete a conversation"""
        return monitor.processor.delete_conversation(conversation_id)
    
    # Include the router
    app.include_router(processor_router, prefix="/v1/conversations", tags=["conversations"])
    
    logger.info(f"Conversation memory system integrated with server (threshold: {max_messages} messages)")
    
    return monitor
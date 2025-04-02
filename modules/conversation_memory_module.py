"""
Simple Conversation Memory Module

This module periodically processes conversation history and stores it in Neo4j
with semantic analysis, completely separate from document storage.

Usage:
- Import and call setup_conversation_memory in your startup event
- No middleware, no request interception, just periodic processing
"""

import asyncio
import logging
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, FastAPI, BackgroundTasks

# Import Neo4j RAG manager
try:
    from modules.neo4j_rag_integration import get_neo4j_rag_manager
except ImportError:
    get_neo4j_rag_manager = None

# Configure logging
logger = logging.getLogger("conversation-memory")

# Configuration
MAX_MESSAGES = 40  # Number of messages before processing
CHECK_INTERVAL = 60  # Seconds between checks

# Global message store reference
_message_store = None

# Global message history reference
_message_history_ref = None

# Track processing status
_last_processed = None
_is_processing = False

class ConversationProcessor:
    """Simple processor for conversation history"""
    
    def __init__(self, neo4j_manager=None):
        """Initialize with Neo4j connection"""
        self.neo4j_manager = neo4j_manager
        if not self.neo4j_manager and get_neo4j_rag_manager:
            # Try to get existing manager
            self.neo4j_manager = get_neo4j_rag_manager()
        
        if not self.neo4j_manager:
            logger.warning("No Neo4j manager available - conversation memory will be limited")
        
        logger.info("Conversation processor initialized")
    
    async def process_conversation(self, messages: List[Dict], 
                              conversation_id: str = "primary_conversation",
                              title: str = None) -> Dict[str, Any]:
        """
        Process a conversation and store in Neo4j
        
        Args:
            messages: List of message dictionaries with role and content
            conversation_id: ID for this conversation
            title: Optional title (generated if not provided)
            
        Returns:
            Status dictionary
        """
        global _is_processing, _last_processed
        
        # Set processing flag
        _is_processing = True
        
        try:
            if not messages:
                logger.warning("No messages to process")
                _is_processing = False
                return {"status": "error", "reason": "No messages provided"}
                
            if not self.neo4j_manager:
                logger.warning("No Neo4j manager available for processing")
                _is_processing = False
                return {"status": "error", "reason": "Neo4j manager not available"}
            
            # Generate a simple title if not provided
            if not title:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                # Try to extract a better title from first few messages
                if len(messages) >= 2:
                    # Get the first user message content
                    for msg in messages[:3]:
                        if msg.get('role') == 'user':
                            # Use first line as title, truncated
                            first_line = msg.get('content', '').split('\n')[0]
                            if first_line:
                                title = (first_line[:40] + '...') if len(first_line) > 40 else first_line
                                break
            
            # Filter out system messages
            filtered_messages = [
                msg for msg in messages 
                if msg.get('role') != 'system'
            ]
            
            # Format the messages for storage
            message_text = ""
            for msg in filtered_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                message_text += f"{role.capitalize()}: {content}\n\n"
            
            # Store in Neo4j
            # Use process_document to avoid duplicating code
            result = await self.neo4j_manager.process_document(
                content=message_text,
                doc_id=conversation_id,
                title=title,
                content_type="conversation",  # Special content type to distinguish from documents
                metadata={"source": "conversation", "message_count": len(filtered_messages)}
            )
            
            # Reset message count by cleaning MESSAGE_HISTORY if we have access to it
            self._reset_message_count()
            
            # Update last processed time
            _last_processed = datetime.now()
            _is_processing = False
            
            # Return status
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "title": title,
                "message_count": len(filtered_messages),
                "processed_at": _last_processed.isoformat(),
                "message_count_reset": True
            }
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            _is_processing = False
            return {"status": "error", "reason": str(e)}
    
    def _reset_message_count(self):
        """Attempt to reset the message count by clearing MESSAGE_HISTORY"""
        global _message_history_ref
        
        try:
            # If we have a direct reference to MESSAGE_HISTORY
            if _message_history_ref is not None:
                # Keep the most recent 10 messages
                if len(_message_history_ref) > 10:
                    _message_history_ref[:] = _message_history_ref[-10:]
                    logger.info("Reset MESSAGE_HISTORY, keeping the 10 most recent messages")
            
            # Try to find MESSAGE_HISTORY in modules
            else:
                import sys
                for module_name, module in sys.modules.items():
                    if hasattr(module, 'MESSAGE_HISTORY'):
                        # Found it, now modify it
                        message_history = getattr(module, 'MESSAGE_HISTORY')
                        if isinstance(message_history, list) and len(message_history) > 10:
                            message_history[:] = message_history[-10:]
                            _message_history_ref = message_history
                            logger.info(f"Reset MESSAGE_HISTORY in module {module_name}, keeping the 10 most recent messages")
                            break
        
        except Exception as e:
            logger.error(f"Error resetting message count: {e}")
            logger.info("Will rely on time-based throttling instead of message count reset")

# Simple function to get current messages
def get_current_messages() -> List[Dict]:
    """Get current messages from store or global variable"""
    global _message_store, _message_history_ref
    
    try:
        # Try message store first
        if _message_store:
            try:
                messages = _message_store.get_messages_by_conversation("primary_conversation")
                if messages:
                    return messages
            except Exception as e:
                logger.error(f"Error getting messages from store: {e}")
        
        # Try global MESSAGE_HISTORY
        if _message_history_ref:
            return _message_history_ref
        
        # Try to find MESSAGE_HISTORY in modules
        import sys
        for module_name, module in sys.modules.items():
            if hasattr(module, 'MESSAGE_HISTORY'):
                _message_history_ref = module.MESSAGE_HISTORY
                return _message_history_ref
    
    except Exception as e:
        logger.error(f"Error getting current messages: {e}")
    
    return []

# Background task for periodic checking
async def check_messages_periodically(processor: ConversationProcessor):
    """Periodically check for messages and process if needed"""
    global _last_processed, _is_processing, MAX_MESSAGES
    
    while True:
        try:
            # Wait between checks
            await asyncio.sleep(CHECK_INTERVAL)
            
            # Skip if already processing
            if _is_processing:
                continue
                
            # Skip if too recent
            if _last_processed and (datetime.now() - _last_processed).total_seconds() < 3600:
                continue
                
            # Get current messages
            messages = get_current_messages()
            
            # Process if we have enough
            if messages and len(messages) >= MAX_MESSAGES:
                logger.info(f"Processing {len(messages)} messages after periodic check")
                _is_processing = True
                
                # Process in the background
                asyncio.create_task(
                    processor.process_conversation(messages)
                )
                
        except Exception as e:
            logger.error(f"Error in periodic check: {e}")
            await asyncio.sleep(120)  # Wait longer after error

# Main setup function
def setup_conversation_memory(app: FastAPI, message_store=None, max_messages=40):
    """
    Set up the conversation memory system
    
    Args:
        app: FastAPI application
        message_store: Message store for retrieving conversation history
        max_messages: Number of messages before processing
    """
    global _message_store, MAX_MESSAGES
    
    # Store configuration
    _message_store = message_store
    MAX_MESSAGES = max_messages
    
    # Create processor
    processor = ConversationProcessor()
    
    # Create router for API endpoints
    router = APIRouter()
    
    @router.post("/process")
    async def force_process_conversation(background_tasks: BackgroundTasks):
        """Force processing of the current conversation"""
        messages = get_current_messages()
        
        if not messages:
            return {"status": "error", "reason": "No messages found"}
            
        # Process in background
        background_tasks.add_task(
            processor.process_conversation, 
            messages
        )
        
        return {
            "status": "processing",
            "message": "Processing started for conversation",
            "message_count": len(messages)
        }
    
    @router.get("/status")
    async def get_memory_status():
        """Get status of the conversation memory system"""
        messages = get_current_messages()
        
        return {
            "status": "active" if processor.neo4j_manager else "limited",
            "max_messages": MAX_MESSAGES,
            "current_messages": len(messages),
            "last_processed": _last_processed.isoformat() if _last_processed else None,
            "is_processing": _is_processing
        }
    
    # Include the router
    app.include_router(router, prefix="/v1/memory", tags=["memory"])
    
    # Add startup event to start the background task
    @app.on_event("startup")
    async def start_background_task():
        asyncio.create_task(check_messages_periodically(processor))
        logger.info(f"Started conversation memory background task (threshold: {MAX_MESSAGES} messages)")
    
    logger.info(f"Conversation memory system set up (threshold: {MAX_MESSAGES} messages)")
    
    return processor
"""
FastAPI endpoints for the RAG module.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import shutil
import os
import tempfile
import logging
import uuid
from pydantic import BaseModel

# Import the RAG module
from .rag_module import RAGManager

# Setup logging
logger = logging.getLogger("rag-api")

# Create router
rag_router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize RAG manager
rag_manager = None

# These classes define the request and response models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filter_tags: Optional[List[str]] = None

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    tags: Optional[List[str]] = None

# Function to get RAG manager instance (dependency injection)
def get_rag_manager():
    global rag_manager
    if rag_manager is None:
        # Import the Redis client here to avoid circular imports
        from redis_client import RedisClient
        redis_client = RedisClient(host='localhost', port=6379, db=0)
        rag_manager = RAGManager(redis_client)
    return rag_manager

@rag_router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """
    Upload a document for RAG processing
    - File is required
    - Title is optional (defaults to filename)
    - Tags are optional, comma-separated
    """
    try:
        rag = get_rag_manager()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Copy uploaded file to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Get content type
        content_type = file.content_type or "application/octet-stream"
        
        # Process in background to avoid timeout
        def process_document_task():
            try:
                result = rag.process_document(
                    file_path=temp_path,
                    filename=file.filename,
                    content_type=content_type,
                    title=title,
                    tags=tag_list
                )
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                logger.info(f"Document processing completed: {result.get('title', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Background document processing failed: {e}")
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Add task to background
        background_tasks.add_task(process_document_task)
        
        return {
            "status": "processing",
            "message": "Document uploaded and processing started",
            "filename": file.filename,
            "content_type": content_type,
            "title": title or file.filename,
            "tags": tag_list
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@rag_router.get("/documents")
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tags: Optional[str] = None
):
    """
    List all documents with pagination and optional tag filtering
    """
    try:
        rag = get_rag_manager()
        
        # Process tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        result = rag.list_documents(
            limit=limit,
            offset=offset,
            filter_tags=tag_list
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@rag_router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """
    Get a document by ID
    """
    try:
        rag = get_rag_manager()
        
        result = rag.get_document_by_id(doc_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@rag_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks
    """
    try:
        rag = get_rag_manager()
        
        result = rag.delete_document(doc_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@rag_router.post("/chat")
async def rag_chat(request: SearchRequest):
    """
    Search and format results for chat integration
    """
    try:
        rag = get_rag_manager()
        
        results = rag.search(
            query=request.query,
            top_k=request.top_k,
            filter_tags=request.filter_tags
        )
        
        # Format results for chat context integration
        if results:
            context_text = "I found the following relevant information:\n\n"
            
            for i, result in enumerate(results, 1):
                title = result.get("title", "Untitled document")
                text = result.get("text", "").strip()
                context_text += f"[{i}] From '{title}':\n{text}\n\n"
                
            return {
                "query": request.query,
                "context": context_text,
                "count": len(results)
            }
        else:
            return {
                "query": request.query,
                "context": "No relevant information found.",
                "count": 0
            }
        
    except Exception as e:
        logger.error(f"Error in RAG chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error in RAG chat: {str(e)}")


@rag_router.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search documents using RAG
    """
    try:
        rag = get_rag_manager()
        
        results = rag.search(
            query=request.query,
            top_k=request.top_k,
            filter_tags=request.filter_tags
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger
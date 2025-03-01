"""
FastAPI endpoints for the RAG module.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends, Body
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
import shutil
import os
import tempfile
import logging
import uuid
from pydantic import BaseModel
import traceback
from datetime import datetime

# Import the RAG module

from .neo4j_rag_integration import Neo4jRAGManager, get_neo4j_rag_manager

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
        # Import the Neo4j RAG manager instead of Redis client
        from .neo4j_rag_integration import get_neo4j_rag_manager
        rag_manager = get_neo4j_rag_manager()
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
        # Initialize response structure first
        response = {
            "documents": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "status": "success"
        }
        
        # Process tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            response["applied_filters"] = {"tags": tag_list}
        
        # Try to get the RAG manager
        try:
            rag = get_rag_manager()
        except Exception as e:
            logger.error(f"Error getting RAG manager: {e}")
            response["status"] = "error"
            response["error"] = f"RAG manager initialization failed: {str(e)}"
            # Return early but with valid JSON
            return JSONResponse(
                status_code=500,
                content=response
            )
        
        # Check Redis connection
        try:
            redis_connected = rag.redis_client.redis_client.ping()
            if not redis_connected:
                response["status"] = "error"
                response["error"] = "Redis connection failed (ping returned False)"
                return JSONResponse(
                    status_code=500,
                    content=response
                )
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            response["status"] = "error"
            response["error"] = f"Redis connection error: {str(e)}"
            return JSONResponse(
                status_code=500,
                content=response
            )
        
        # Try to get documents
        try:
            result = rag.list_documents(
                limit=limit,
                offset=offset,
                filter_tags=tag_list
            )
            
            # Update response with results
            if "documents" in result:
                response["documents"] = result.get("documents", [])
            if "total" in result:
                response["total"] = result.get("total", 0)
            if "error" in result:
                response["warning"] = result.get("error")
            
            return response
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            response["status"] = "error"
            response["error"] = f"Error listing documents: {str(e)}"
            # Still return a valid JSON response
            return JSONResponse(
                status_code=500,
                content=response
            )
        
    except Exception as e:
        # Catch absolutely any error and still return valid JSON
        logger.error(f"Unexpected error in documents API: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "documents": [],
                "total": 0,
                "status": "error",
                "error": f"Unexpected error: {str(e)}"
            }
        )

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

@rag_router.get("/documents/{doc_id}/download")
async def download_document(doc_id: str):
    """
    Download a document by ID
    """
    try:
        rag = get_rag_manager()
        
        # Get document metadata
        doc_data = rag.get_document_by_id(doc_id)
        
        if "error" in doc_data:
            raise HTTPException(status_code=404, detail=doc_data["error"])
        
        # Get file path from metadata or construct it
        if "file_path" in doc_data:
            file_path = doc_data["file_path"]
        else:
            # Fallback for older documents
            doc_dir = os.path.join(rag.DOCUMENTS_DIRECTORY, doc_id)
            storage_filename = doc_data.get("storage_filename")
            
            if storage_filename:
                file_path = os.path.join(doc_dir, storage_filename)
            else:
                # Very old format fallback
                content_type = doc_data.get("content_type", "")
                extension = content_type.split('/')[-1]
                file_path = os.path.join(rag.DOCUMENTS_DIRECTORY, f"{doc_id}.{extension}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Determine filename for the download
        filename = doc_data.get("storage_filename", doc_data.get("filename", f"{doc_id}"))
        
        # Return the file
        return FileResponse(
            path=file_path, 
            filename=filename,
            media_type=doc_data.get("content_type", "application/octet-stream")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading document: {str(e)}")


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

@rag_router.get("/debug/documents")
@rag_router.get("/debug/documents")
async def debug_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Debug endpoint to check the documents API response
    """
    try:
        rag = get_rag_manager()
        
        # Get the documents directory path (handling the case where it might be missing)
        docs_dir = getattr(rag, "DOCUMENTS_DIRECTORY", "/home/david/Sara/documents")
        
        # Get Redis connection status
        redis_status = {
            "connected": rag.redis_client.redis_client.ping(),
            "indices": []
        }
        
        # Check indices
        try:
            docs_info = rag.redis_client.redis_client.ft(rag.docs_index_name).info()
            redis_status["indices"].append({
                "name": rag.docs_index_name,
                "docs_count": docs_info["num_docs"] if "num_docs" in docs_info else "unknown"
            })
        except Exception as e:
            redis_status["indices"].append({
                "name": rag.docs_index_name,
                "error": str(e)
            })
        
        # Get document keys directly
        doc_keys = rag.redis_client.redis_client.keys("doc:*")
        doc_count = len(doc_keys)
        
        # Try to get a sample document
        sample_doc = None
        if doc_keys:
            try:
                sample_doc = rag.redis_client.redis_client.json().get(doc_keys[0])
            except Exception as e:
                sample_doc = {"error": str(e)}
        
        return {
            "redis_status": redis_status,
            "doc_count": doc_count,
            "documents_dir_exists": os.path.exists(docs_dir),
            "documents_dir_path": docs_dir,
            "sample_doc": sample_doc,
            "sample_keys": doc_keys[:5] if doc_keys else []
        }
        
    except Exception as e:
        logger.error(f"Error in debug documents: {e}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@rag_router.get("/simple-documents")
async def simple_documents_list():
    """
    Simple endpoint that just returns valid JSON with documents
    Used as a fallback when the main endpoint isn't working
    """
    try:
        rag = get_rag_manager()
        redis_client = rag.redis_client.redis_client
        
        # Direct key fetch approach
        doc_keys = redis_client.keys("doc:*")
        documents = []
        
        for key in doc_keys:
            try:
                doc_data = redis_client.json().get(key)
                documents.append(doc_data)
            except Exception as e:
                logger.warning(f"Error getting document {key}: {e}")
                continue
        
        # Force content type to application/json
        return JSONResponse(
            content={
                "documents": documents,
                "total": len(documents),
                "source": "simple-documents"
            },
            status_code=200,
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error in simple documents endpoint: {e}")
        # Always return valid JSON even on error
        return JSONResponse(
            content={
                "documents": [],
                "total": 0,
                "error": str(e)
            },
            status_code=500,
            media_type="application/json"
        )


@rag_router.get("/health/documents")
async def documents_health_check():
    """
    Health check specifically for the documents API
    """
    try:
        rag = get_rag_manager()
        
        # Get the documents directory path (handling the case where it might be missing)
        docs_dir = getattr(rag, "DOCUMENTS_DIRECTORY", "/home/david/Sara/documents")
        
        # Check Redis connection
        redis_connected = False
        try:
            redis_connected = rag.redis_client.redis_client.ping()
        except Exception as redis_error:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Redis connection failed",
                    "error": str(redis_error)
                }
            )
            
        if not redis_connected:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Redis connection ping failed"
                }
            )
        
        # Check document index
        index_ok = False
        index_error = None
        try:
            rag.redis_client.redis_client.ft(rag.docs_index_name).info()
            index_ok = True
        except Exception as idx_error:
            index_error = str(idx_error)
        
        # Get document count directly
        doc_count = len(rag.redis_client.redis_client.keys("doc:*"))
        
        # Check documents directory
        docs_dir_exists = os.path.exists(docs_dir)
        
        return {
            "status": "healthy" if index_ok else "degraded",
            "redis_connected": redis_connected,
            "index_ok": index_ok,
            "index_error": index_error,
            "doc_count": doc_count,
            "docs_dir_exists": docs_dir_exists,
            "docs_dir": docs_dir,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in documents health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }
        )

@rag_router.post("/clear-cache")
async def clear_document_cache(request: dict = Body(...)):
    """
    Clear all document data from Redis
    """
    # Check confirmation
    if not request.get("confirmation", False):
        return JSONResponse(
            status_code=400,
            content={"message": "Confirmation required"}
        )
    
    try:
        rag = get_rag_manager()
        redis_client = rag.redis_client.redis_client
        
        # Document-related key patterns
        key_patterns = [
            "doc:*",         # Document metadata
            "chunk:*",       # Chunk metadata and content
            "chunk:json:*",  # Chunk JSON data
            "chunk:vector:*" # Chunk vector data
        ]
        
        # Delete keys for each pattern
        total_deleted = 0
        for pattern in key_patterns:
            keys = redis_client.keys(pattern)
            if keys:
                deleted = redis_client.delete(*keys)
                total_deleted += deleted
                logger.info(f"Deleted {deleted} keys matching pattern {pattern}")
        
        # Reset index if needed
        try:
            redis_client.ft(rag.docs_index_name).dropindex()
            logger.info(f"Dropped index {rag.docs_index_name}")
        except:
            logger.info(f"Index {rag.docs_index_name} not found or could not be dropped")
        
        # Re-create the index
        try:
            rag._ensure_indices()
            logger.info("Re-created indices")
        except Exception as e:
            logger.error(f"Error re-creating indices: {e}")
        
        return {
            "message": f"Document cache cleared successfully. Deleted {total_deleted} keys.",
            "deleted_count": total_deleted
        }
        
    except Exception as e:
        logger.error(f"Error clearing document cache: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Error clearing document cache: {str(e)}",
                "error": str(e)
            }
        )
"""
FastAPI endpoints for the RAG module.
This simplified version provides a cleaner interface to the RAG functionality.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends
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
import asyncio

# Import the unified RAG module
from .neo4j_rag_integration import get_neo4j_rag_manager

# Setup logging
logger = logging.getLogger("rag-api")

# Create router
rag_router = APIRouter(prefix="/rag", tags=["RAG"])

# Request and response models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filter_tags: Optional[List[str]] = None

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    tags: Optional[List[str]] = None

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
        rag = get_neo4j_rag_manager()
        
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
        
        # Function for async document processing
        async def process_document_background():
            try:
                # Process the document
                result = await rag.process_document(
                    file_path=temp_path,
                    filename=file.filename,
                    content_type=content_type,
                    title=title,
                    tags=tag_list
                )
                
                logger.info(f"Document processing completed for {file.filename}")
                
                # Clean up the temporary file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error removing temp file: {e}")
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                # Make sure to clean up temp file in case of error
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temp file: {cleanup_error}")
        
        # Add background task
        background_tasks.add_task(process_document_background)
        
        return {
            "status": "processing",
            "message": "Document uploaded and processing started",
            "filename": file.filename,
            "content_type": content_type,
            "title": title or file.filename,
            "tags": tag_list
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # Make sure to clean up temp file in case of error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    
@rag_router.get("/documents")
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tags: Optional[str] = None
):
    """
    List all documents with pagination and optional tag filtering (Now uses Neo4j)
    """
    try:
        # Process tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Get Neo4j RAG manager
        try:
            rag = get_neo4j_rag_manager()
        except Exception as e:
            logger.error(f"Error getting Neo4j RAG manager: {e}")
            return JSONResponse( status_code=500, content={ "status": "error", "error": f"RAG manager initialization failed: {str(e)}" } )

        # Get documents using Neo4jRAGManager.list_documents
        try:
            # *** REMOVE await HERE ***
            result = rag.list_documents( # Call the Neo4j manager's method (it's synchronous)
                limit=limit,
                offset=offset,
                filter_tags=tag_list
            )

            # Check if the manager returned an error internally
            if "error" in result:
                 logger.error(f"Error listing documents from Neo4j: {result['error']}")
                 raise HTTPException(status_code=500, detail=result['error'])

            return {
                "documents": result.get("documents", []),
                "total": result.get("total", 0),
                "offset": offset,
                "limit": limit,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error listing documents via Neo4j manager: {e}")
            logger.exception(e)
            # Remove or comment out the Redis-specific fallback logic
            # try:
            #     # Direct key fetch as fallback # <<< REMOVE THIS BLOCK START
            #     doc_keys = rag.redis_client.redis_client.keys("doc:*")
            #     # ... rest of redis fallback ...
            #     return { ... } # <<< REMOVE THIS BLOCK END
            # except Exception as fallback_e:
            #     logger.error(f"Fallback method failed: {fallback_e}")

            # Raise the original error or return a generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "documents": [], "total": 0, "status": "error",
                    "error": f"Error listing documents: {str(e)}"
                }
            )

    except Exception as e:
        # Catch any unexpected error
        logger.error(f"Unexpected error in documents API: {e}")
        return JSONResponse( status_code=500, content={ "documents": [], "total": 0, "status": "error", "error": f"Unexpected error: {str(e)}" } )

@rag_router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """
    Get a document by ID
    """
    try:
        rag = get_neo4j_rag_manager()
        
        result = await rag.get_document_by_id(doc_id)
        
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
        rag = get_neo4j_rag_manager()
        
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
        rag = get_neo4j_rag_manager()
        
        # Get document metadata
        doc_data = rag.get_document_by_id(doc_id)
        
        if "error" in doc_data:
            raise HTTPException(status_code=404, detail=doc_data["error"])
        
        # Get file path from metadata
        if "file_path" in doc_data:
            file_path = doc_data["file_path"]
        else:
            # Fallback for older documents
            doc_dir = os.path.join(rag.config.DOCUMENTS_DIRECTORY, doc_id)
            storage_filename = doc_data.get("storage_filename")
            
            if storage_filename:
                file_path = os.path.join(doc_dir, storage_filename)
            else:
                # Very old format fallback
                content_type = doc_data.get("content_type", "")
                extension = content_type.split('/')[-1]
                file_path = os.path.join(rag.config.DOCUMENTS_DIRECTORY, f"{doc_id}.{extension}")
        
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

@rag_router.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search documents using RAG
    """
    try:
        rag = get_neo4j_rag_manager()
        
        results = await rag.search(
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
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@rag_router.post("/chat")
async def rag_chat(request: SearchRequest):
    """
    Search and format results for chat integration
    """
    try:
        rag = get_neo4j_rag_manager()
        
        results = await rag.search(
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

# Fallback endpoint for when the main endpoint isn't working
@rag_router.get("/simple-documents")
async def simple_documents_list():
    """
    Simple endpoint that just returns valid JSON with documents
    (Updated to use Neo4j as a simple fallback)
    """
    try:
        rag = get_neo4j_rag_manager()

        # Use the primary list_documents method, maybe with default limit
        # This avoids duplicating logic. Set a reasonable limit.
        result = rag.list_documents(limit=50, offset=0)

        if "error" in result:
             logger.warning(f"Error getting documents from Neo4j in simple endpoint: {result['error']}")
             raise Exception(result['error'])

        # Force content type to application/json
        return JSONResponse(
            content={
                "documents": result.get("documents", []),
                "total": result.get("total", 0),
                "source": "simple-documents (Neo4j)"
            },
            status_code=200,
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error in simple documents endpoint (Neo4j): {e}")
        # Always return valid JSON even on error
        return JSONResponse(
            content={ "documents": [], "total": 0, "error": str(e) },
            status_code=500,
            media_type="application/json"
        )

@rag_router.get("/health/documents")
async def documents_health_check():
    """
    Health check specifically for the documents API
    """
    try:
        rag = get_neo4j_rag_manager()
        
        # Check Redis connection
        redis_connected = False
        try:
            redis_connected = rag.redis_client.ping()
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
        docs_dir = rag.config.DOCUMENTS_DIRECTORY
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
async def clear_document_cache(confirmation: bool = False):
    """
    Clear all document data from Redis
    """
    # Check confirmation
    if not confirmation:
        return JSONResponse(
            status_code=400,
            content={"message": "Confirmation required"}
        )
    
    try:
        rag = get_neo4j_rag_manager()
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
        
        try:
            redis_client.ft(rag.chunks_index_name).dropindex()
            logger.info(f"Dropped index {rag.chunks_index_name}")
        except:
            logger.info(f"Index {rag.chunks_index_name} not found or could not be dropped")
        
        # Re-create the indices
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
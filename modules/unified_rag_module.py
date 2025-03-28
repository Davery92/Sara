"""
Unified RAG (Retrieval-Augmented Generation) module.
This streamlined module integrates Redis for storage and vector search.
"""

import os
import json
import uuid
import logging
import re
import numpy as np
import tempfile
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
import ollama
import traceback
from pathlib import Path
from redis.commands.search.field import TextField, VectorField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# Configure logging
logger = logging.getLogger("rag-module")

class RagConfig:
    """Configuration for the RAG module"""
    DOCUMENTS_DIRECTORY = os.environ.get("DOCUMENTS_DIRECTORY", "/home/david/Sara/documents")
    CHUNK_SIZE = 1000  # tokens
    CHUNK_OVERLAP = 100  # tokens
    TOP_K_INITIAL = 15  # Initial retrieval count
    TOP_K_FINAL = 3  # Final count after reranking
    EMBEDDING_MODEL = "bge-m3"  # Default embedding model
    EMBEDDING_DIMENSION = 1024  # BGE-M3 embedding dimension
    
    def __init__(self, **kwargs):
        """Initialize with optional overrides for any config values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Override config {key}={value}")
        
        # Ensure documents directory exists
        os.makedirs(self.DOCUMENTS_DIRECTORY, exist_ok=True)

class RedisConnectionManager:
    """Manages Redis connections and provides helper methods"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize the Redis connection"""
        try:
            import redis
            # For general commands with string responses
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # For binary data (like vectors)
            self.redis_binary = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False
            )
            logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError:
            logger.error("Redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            raise
    
    def ping(self):
        """Test the Redis connection"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False


class RAGManager:
    """Main RAG manager class that handles document processing and retrieval"""
    
    def __init__(self, redis_connection, config=None):
        """Initialize the RAG manager"""
        self.redis_client = redis_connection
        self.config = config or RagConfig()
        self.docs_index_name = "docs_idx"
        self.chunks_index_name = "chunks_idx"
        self._ensure_indices()
    
    def _ensure_indices(self):
        """Create the necessary Redis indices if they don't exist"""
        try:
            # Try to get document index info
            self.redis_client.redis_client.ft(self.docs_index_name).info()
            logger.info("Document index already exists")
        except Exception:
            logger.info("Creating document index")
            self._create_document_index()
        
        try:
            # Try to get chunks index info
            self.redis_client.redis_client.ft(self.chunks_index_name).info()
            logger.info("Chunks index already exists")
        except Exception:
            logger.info("Creating chunks index")
            self._create_chunks_index()
    
    def _create_document_index(self):
        """Create the document index"""
        try:
            # Define schema for document metadata
            docs_schema = (
                TextField("$.title", as_name="title"),
                TextField("$.filename", as_name="filename"),
                TextField("$.content_type", as_name="content_type"),
                NumericField("$.size", as_name="size"),
                TagField("$.tags", as_name="tags"),
                TextField("$.date_added", as_name="date_added")
            )
            
            # Create index
            self.redis_client.redis_client.ft(self.docs_index_name).create_index(
                docs_schema,
                definition=IndexDefinition(
                    prefix=["doc:"],
                    index_type=IndexType.JSON
                )
            )
            logger.info("Created document index")
        except Exception as e:
            logger.error(f"Error creating document index: {e}")
            raise
    
    def _create_chunks_index(self):
        """Create the chunks index"""
        try:
            # Define schema for chunks
            chunks_schema = (
                TextField("$.text", as_name="text"),
                TextField("$.doc_id", as_name="doc_id"),
                NumericField("$.chunk_index", as_name="chunk_index"),
                TextField("$.metadata.title", as_name="title"),
                TextField("$.metadata.filename", as_name="filename"),
                TagField("$.metadata.tags", as_name="tags"),
                VectorField("$.embedding", 
                           "HNSW", {
                               "TYPE": "FLOAT32", 
                               "DIM": self.config.EMBEDDING_DIMENSION,
                               "DISTANCE_METRIC": "COSINE"
                           }, as_name="embedding")
            )
            
            # Create index
            self.redis_client.redis_client.ft(self.chunks_index_name).create_index(
                chunks_schema,
                definition=IndexDefinition(
                    prefix=["chunk:"],
                    index_type=IndexType.JSON
                )
            )
            logger.info("Created chunks index")
        except Exception as e:
            logger.error(f"Error creating chunks index: {e}")
            raise
    
    async def process_document(self, file_path: str, filename: str, content_type: str, 
                            title: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a document: extract text, chunk it, embed chunks, and store in Redis
        """
        try:
            # Generate a unique document ID
            doc_id = str(uuid.uuid4())
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
                file_size = len(file_content)
            
            # Extract text based on content type
            text_content = self._extract_text(file_path, content_type)
            if not text_content:
                return {"error": f"Failed to extract text from document: {filename}"}
            
            # Create document metadata
            if not title:
                title = os.path.splitext(filename)[0]  # Use filename without extension as title
                
            # Create a clean filename for storage
            safe_title = "".join(c if c.isalnum() else "_" for c in title).lower()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            storage_filename = f"{safe_title}_{timestamp}{os.path.splitext(filename)[1]}"
            
            # Store document metadata
            doc_metadata = {
                "id": doc_id,
                "title": title,
                "filename": filename,
                "storage_filename": storage_filename,
                "content_type": content_type,
                "size": file_size,
                "tags": tags or [],
                "date_added": datetime.now().isoformat(),
                "chunk_count": 0  # Will update after chunking
            }
            
            # Store document metadata in Redis
            self.redis_client.redis_client.json().set(f"doc:{doc_id}", '$', doc_metadata)
            
            # Chunk the text
            chunks = self._chunk_text(text_content)
            doc_metadata["chunk_count"] = len(chunks)
            
            # Update document with chunk count
            self.redis_client.redis_client.json().set(f"doc:{doc_id}", '$', doc_metadata)
            
            # Process chunks (embed and store)
            for i, chunk in enumerate(chunks):
                await self._process_chunk(chunk, doc_id, i, doc_metadata)
            
            # Calculate document hash for deduplication
            content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
            doc_metadata["content_hash"] = content_hash
            
            # Save document to disk
            doc_storage_dir = os.path.join(self.config.DOCUMENTS_DIRECTORY, doc_id)
            os.makedirs(doc_storage_dir, exist_ok=True)
            
            # Save original document
            original_doc_path = os.path.join(doc_storage_dir, storage_filename)
            with open(original_doc_path, 'wb') as f:
                f.write(file_content)
            
            # Save metadata as JSON
            metadata_path = os.path.join(doc_storage_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(doc_metadata, f, indent=2)
            
            # Add paths to metadata and update in Redis
            doc_metadata["file_path"] = original_doc_path
            doc_metadata["metadata_path"] = metadata_path
            self.redis_client.redis_client.json().set(f"doc:{doc_id}", '$', doc_metadata)
            
            logger.info(f"Processed document: {title} ({doc_id}) with {len(chunks)} chunks")
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            logger.exception(e)
            return {"error": str(e)}
    
    def _extract_text(self, file_path: str, content_type: str) -> str:
        """Extract text from various file types"""
        try:
            # Plain text files
            if content_type in ['text/plain', 'text/markdown']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            
            # PDF files
            elif content_type == 'application/pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n\n"
                        return text
                except ImportError:
                    logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
                    return ""
            
            # Word documents
            elif content_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return "\n\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.error("python-docx not installed. Install with: pip install python-docx")
                    return ""
            
            # HTML files
            elif content_type in ['text/html', 'application/xhtml+xml']:
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.extract()
                        return soup.get_text(separator="\n")
                except ImportError:
                    logger.error("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
                    return ""
            
            # CSV files
            elif content_type == 'text/csv':
                try:
                    import csv
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        reader = csv.reader(f)
                        return "\n".join([",".join(row) for row in reader])
                except ImportError:
                    logger.error("CSV support requires Python's csv module")
                    return ""
            
            # JSON files
            elif content_type == 'application/json':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = json.load(f)
                    # Convert JSON to string representation
                    return json.dumps(data, indent=2)
            
            # Unsupported file type
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into segments with specified token size and overlap
        """
        try:
            import tiktoken
            # Initialize tokenizer
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            # Tokenize the text
            tokens = tokenizer.encode(text)
            
            # Initialize chunks
            chunks = []
            i = 0
            
            # Create chunks with overlap
            while i < len(tokens):
                # Get the chunk
                chunk_end = min(i + self.config.CHUNK_SIZE, len(tokens))
                chunk_tokens = tokens[i:chunk_end]
                
                # Decode the chunk
                chunk_text = tokenizer.decode(chunk_tokens)
                
                # Clean up the chunk (remove excessive whitespace)
                chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
                
                # Add to chunks if not empty
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Move to next chunk with overlap
                i += (self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP)
                
                # Make sure we make progress
                if i <= 0:
                    i = chunk_end
            
            return chunks
        except ImportError:
            logger.error("tiktoken not installed. Install with: pip install tiktoken")
            # Fallback to simple character-based chunking
            chunks = []
            words = text.split()
            current_chunk = []
            current_length = 0
            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for the space
                if current_length >= self.config.CHUNK_SIZE:
                    chunks.append(' '.join(current_chunk))
                    # Apply overlap
                    overlap_words = current_chunk[-(self.config.CHUNK_OVERLAP // 5):]  # Approximate tokens to words
                    current_chunk = overlap_words
                    current_length = sum(len(word) + 1 for word in overlap_words)
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Emergency fallback - split by paragraphs
            return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    async def _process_chunk(self, chunk_text: str, doc_id: str, chunk_index: int, doc_metadata: Dict[str, Any]):
        """Process a single chunk: generate embedding and store in Redis"""
        try:
            # Generate embedding using Ollama
            embedding_result = ollama.embeddings(
                model=self.config.EMBEDDING_MODEL,
                prompt=chunk_text
            )
            
            embedding = embedding_result.get('embedding', [])
            
            if not embedding:
                logger.error(f"Failed to generate embedding for chunk {chunk_index} of doc {doc_id}")
                return
            
            # Use different key names for JSON data and vector data
            chunk_json_key = f"chunk:json:{doc_id}:{chunk_index}"
            chunk_vector_key = f"chunk:vector:{doc_id}:{chunk_index}"
            
            # Create chunk data
            chunk_data = {
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "metadata": {
                    "title": doc_metadata.get("title", ""),
                    "filename": doc_metadata.get("filename", ""),
                    "tags": doc_metadata.get("tags", [])
                },
                # Add a reference to the vector key
                "vector_key": chunk_vector_key
            }
            
            # Delete existing keys if they exist
            keys_to_delete = [
                chunk_json_key, 
                chunk_vector_key,
                f"chunk:{doc_id}:{chunk_index}"  # Legacy format
            ]
            
            for key in keys_to_delete:
                if self.redis_client.redis_client.exists(key):
                    self.redis_client.redis_client.delete(key)
            
            # Store chunk JSON data
            self.redis_client.redis_client.json().set(chunk_json_key, '$', chunk_data)
            
            # Store vector data separately using a string key with binary value
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            self.redis_client.redis_binary.set(chunk_vector_key, embedding_bytes)
            
            # Create an index record for backward compatibility
            index_data = {
                "json_key": chunk_json_key,
                "vector_key": chunk_vector_key
            }
            self.redis_client.redis_client.json().set(f"chunk:{doc_id}:{chunk_index}", '$', index_data)
            
            logger.debug(f"Processed chunk {chunk_index} of document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} of doc {doc_id}: {e}")
    
    async def search(self, query: str, top_k: int = None, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        if top_k is None:
            top_k = self.config.TOP_K_FINAL
            
        try:
            # First, perform semantic search
            semantic_results = await self._semantic_search(query, top_k * 2, filter_tags)
            
            # If we have enough results, rerank and return
            if len(semantic_results) >= top_k:
                reranked_results = self._rerank_results(query, semantic_results)
                return reranked_results[:top_k]
            
            # If not enough results, try BM25 search as fallback
            logger.info(f"Semantic search returned only {len(semantic_results)} results, adding BM25 results")
            bm25_results = await self._bm25_search(query, top_k * 2, filter_tags)
            
            # Combine results (avoiding duplicates)
            combined_results = semantic_results.copy()
            seen_chunk_ids = {result["chunk_id"] for result in semantic_results}
            
            for result in bm25_results:
                if result["chunk_id"] not in seen_chunk_ids:
                    combined_results.append(result)
                    seen_chunk_ids.add(result["chunk_id"])
                    
                    # Stop once we have enough
                    if len(combined_results) >= top_k * 2:
                        break
            
            # Rerank the combined results
            reranked_results = self._rerank_results(query, combined_results)
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    async def _semantic_search(self, query: str, top_k: int = None, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        if top_k is None:
            top_k = self.config.TOP_K_INITIAL
            
        try:
            # Get query embedding
            embedding_result = ollama.embeddings(
                model=self.config.EMBEDDING_MODEL,
                prompt=query
            )
            query_embedding = embedding_result.get('embedding', [])
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return await self._bm25_search(query, top_k, filter_tags)
            
            # Get all chunks (we'll do vector matching manually due to Redis limitations)
            all_chunks = await self._get_all_chunks(filter_tags)
            
            # Calculate vector similarity between query and all chunks
            results = []
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            
            for chunk in all_chunks:
                # Try to get the vector
                vector_key = chunk.get("vector_key")
                chunk_id = chunk.get("chunk_id")
                
                if vector_key and self.redis_client.redis_binary.exists(vector_key):
                    # Get embedding bytes and convert to numpy array
                    embedding_bytes = self.redis_client.redis_binary.get(vector_key)
                    chunk_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding_np, chunk_embedding)
                    
                    results.append({
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "title": chunk.get("title", ""),
                        "score": float(similarity),
                        "search_type": "vector"
                    })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            # Fallback to BM25 search
            return await self._bm25_search(query, top_k, filter_tags)
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)
    
    async def _bm25_search(self, query: str, top_k: int = None, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform BM25 search as a fallback"""
        if top_k is None:
            top_k = self.config.TOP_K_INITIAL
            
        try:
            # Get all chunks
            all_chunks = await self._get_all_chunks(filter_tags)
            
            if not all_chunks:
                logger.warning("No chunks found for BM25 search")
                return []
            
            try:
                from rank_bm25 import BM25Okapi
                
                # Extract texts and tokenize
                texts = [doc.get("text", "") for doc in all_chunks]
                tokenized_corpus = [text.lower().split() for text in texts]
                
                # Create BM25 model
                bm25 = BM25Okapi(tokenized_corpus)
                
                # Tokenize query
                tokenized_query = query.lower().split()
                
                # Get BM25 scores
                doc_scores = bm25.get_scores(tokenized_query)
                
                # Create results with scores
                results = []
                for i, (doc, score) in enumerate(zip(all_chunks, doc_scores)):
                    results.append({
                        "chunk_id": doc.get("chunk_id"),
                        "text": doc.get("text", ""),
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),
                        "score": float(score),
                        "search_type": "bm25"
                    })
                
                # Sort by score (descending) and take top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
                
            except ImportError:
                logger.warning("rank_bm25 not installed. Using simple keyword matching.")
                
                # Fall back to simple keyword matching
                query_terms = set(query.lower().split())
                results = []
                
                for doc in all_chunks:
                    text = doc.get("text", "").lower()
                    
                    # Count matching terms
                    matched_terms = sum(1 for term in query_terms if term in text)
                    if matched_terms > 0:
                        results.append({
                            "chunk_id": doc.get("chunk_id"),
                            "text": doc.get("text", ""),
                            "doc_id": doc.get("doc_id", ""),
                            "title": doc.get("title", ""),
                            "score": float(matched_terms) / len(query_terms),
                            "search_type": "keyword"
                        })
                
                # Sort by score (descending) and take top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    async def _get_all_chunks(self, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get all chunks, optionally filtered by tags"""
        try:
            # Get all JSON chunk keys
            chunk_keys = self.redis_client.redis_client.keys("chunk:json:*")
            
            if not chunk_keys:
                logger.warning("No chunk keys found")
                return []
            
            # Process results
            processed_results = []
            for key in chunk_keys:
                try:
                    # Get the JSON data
                    chunk_data = self.redis_client.redis_client.json().get(key)
                    
                    # Filter by tags if needed
                    if filter_tags and len(filter_tags) > 0:
                        tags = chunk_data.get("metadata", {}).get("tags", [])
                        if not any(tag in tags for tag in filter_tags):
                            continue
                    
                    # Create result object
                    processed_results.append({
                        "chunk_id": key,
                        "text": chunk_data.get("text", ""),
                        "doc_id": chunk_data.get("doc_id", ""),
                        "title": chunk_data.get("metadata", {}).get("title", ""),
                        "vector_key": chunk_data.get("vector_key")
                    })
                except Exception as e:
                    logger.error(f"Error processing chunk {key}: {e}")
            
            logger.info(f"Found {len(processed_results)} chunks")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error getting all chunks: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results using a basic reranking strategy"""
        if not results:
            return []
            
        try:
            # Try to use cross-encoder reranker if available
            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                
                # Prepare pairs for reranking
                pairs = [[query, result["text"]] for result in results]
                
                # Get scores
                scores = reranker.predict(pairs)
                
                # Add scores to results
                for i, score in enumerate(scores):
                    results[i]["rerank_score"] = float(score)
                
                # Sort by reranker score
                results.sort(key=lambda x: x["rerank_score"], reverse=True)
                return results
                
            except ImportError:
                logger.warning("sentence_transformers not installed. Using fallback reranking.")
            
            # Fallback: combine vector similarity and keyword relevance
            query_terms = set(query.lower().split())
            
            for result in results:
                text = result.get("text", "").lower()
                
                # Calculate term overlap
                matched_terms = sum(1 for term in query_terms if term in text)
                term_score = matched_terms / len(query_terms) if query_terms else 0
                
                # Combine with vector score
                combined_score = 0.7 * result.get("score", 0) + 0.3 * term_score
                result["rerank_score"] = combined_score
            
            # Sort by combined score
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # If reranking fails, just use original scores
            for result in results:
                if "rerank_score" not in result:
                    result["rerank_score"] = result.get("score", 0)
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
    
    async def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata by ID"""
        try:
            doc_key = f"doc:{doc_id}"
            if not self.redis_client.redis_client.exists(doc_key):
                return {"error": f"Document not found: {doc_id}"}
            
            doc_data = self.redis_client.redis_client.json().get(doc_key)
            return doc_data
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return {"error": str(e)}
    
    async def list_documents(self, limit: int = 100, offset: int = 0,
                   filter_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """List all documents with pagination and filtering"""
        try:
            # Try to execute search via index
            try:
                # Build query string based on filters
                if filter_tags and len(filter_tags) > 0:
                    tag_filters = " | ".join([f"@tags:{{{tag}}}" for tag in filter_tags])
                    query_string = f"({tag_filters})"
                else:
                    query_string = "*"

                # *** Create a Query object ***
                redis_query = Query(query_string) \
                                .sort_by("date_added", asc=False) \
                                .paging(offset, limit) \
                                .return_fields("id", "json") # Ask Redis to return the JSON

                # *** Execute the search using the Query object ***
                results = self.redis_client.redis_client.ft(self.docs_index_name).search(redis_query)

                # Process results
                docs = []
                for doc in results.docs:
                    try:
                        # The JSON string should be in doc.json if return_fields was successful
                        if hasattr(doc, 'json') and doc.json:
                            doc_data = json.loads(doc.json)
                            # Ensure the ID is present, sometimes it's only in doc.id
                            if 'id' not in doc_data and hasattr(doc, 'id'):
                                # Extract the actual ID part from the Redis key (e.g., "doc:uuid" -> "uuid")
                                doc_data['id'] = doc.id.split(':')[-1]
                            docs.append(doc_data)
                        elif hasattr(doc, 'id'):
                            # Fallback if doc.json is missing, get the full JSON using the key
                            logger.warning(f"doc.json missing for {doc.id}, attempting direct fetch.")
                            try:
                                doc_data = self.redis_client.redis_client.json().get(doc.id)
                                if isinstance(doc_data, dict):
                                    docs.append(doc_data)
                                else:
                                    logger.error(f"Direct fetch for {doc.id} did not return a dictionary.")
                            except Exception as fetch_err:
                                logger.error(f"Error fetching doc {doc.id} directly: {fetch_err}")
                        else:
                            logger.warning(f"Search result document missing 'json' and 'id' attributes: {vars(doc)}")

                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing document JSON for doc {getattr(doc, 'id', 'N/A')}: {e}. Content: {getattr(doc, 'json', None)}")
                        continue # Skip this document
                    except Exception as e:
                        logger.error(f"Error processing search result doc {getattr(doc, 'id', 'N/A')}: {e}")
                        continue # Skip this document

                return {
                    "documents": docs,
                    "total": results.total, # Total matching documents found by the index
                    "offset": offset,
                    "limit": limit
                }

            except Exception as e:
                # Keep the fallback logic
                logger.warning(f"Search index error: {e}, falling back to manual listing")
                # (Fallback code remains unchanged)
                # ... (existing fallback code) ...

        except Exception as e:
            # (Outer exception handling remains unchanged)
            # ... (existing outer exception code) ...
            logger.error(f"Error listing documents: {e}")
            # Always return a valid JSON response, even in case of error
            return {
                "documents": [],
                "total": 0,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document and all its chunks"""
        try:
            # Get document first to verify it exists
            doc_key = f"doc:{doc_id}"
            if not self.redis_client.redis_client.exists(doc_key):
                return {"error": f"Document not found: {doc_id}"}
            
            # Get document data for the response
            doc_data = self.redis_client.redis_client.json().get(doc_key)
            
            # Delete all chunks
            chunk_prefix = f"chunk:*:{doc_id}:"
            chunk_keys = self.redis_client.redis_client.keys(f"{chunk_prefix}*")
            
            # Also get chunk:json and chunk:vector keys
            json_keys = self.redis_client.redis_client.keys(f"chunk:json:{doc_id}:*")
            vector_keys = self.redis_client.redis_client.keys(f"chunk:vector:{doc_id}:*")
            
            # Combine all keys to delete
            all_keys = set(chunk_keys + json_keys + vector_keys)
            
            if all_keys:
                deleted = 0
                # Delete in batches to avoid overloading Redis
                batch_size = 100
                key_batches = [list(all_keys)[i:i + batch_size] for i in range(0, len(all_keys), batch_size)]
                
                for batch in key_batches:
                    batch_deleted = self.redis_client.redis_client.delete(*batch)
                    deleted += batch_deleted
            
            # Delete document
            self.redis_client.redis_client.delete(doc_key)
            
            # Delete document directory from disk if it exists
            doc_dir_path = os.path.join(self.config.DOCUMENTS_DIRECTORY, doc_id)
            if os.path.exists(doc_dir_path):
                import shutil
                shutil.rmtree(doc_dir_path)
            
            logger.info(f"Deleted document {doc_id} with {len(all_keys)} related keys")
            
            return {
                "success": True,
                "document": doc_data,
                "chunks_deleted": len(all_keys)
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return {"error": str(e)}

# Function to get a singleton instance of the RAG manager
_rag_manager_instance = None

def get_rag_manager(**kwargs):
    """Get or create a singleton instance of the RAG manager"""
    global _rag_manager_instance
    
    if _rag_manager_instance is None:
        # Create Redis connection
        redis_connection = RedisConnectionManager()
        
        # Create RAG config with any overrides
        config = RagConfig(**kwargs)
        
        # Create RAG manager
        _rag_manager_instance = RAGManager(redis_connection, config)
        logger.info("Initialized RAG manager singleton")
    
    return _rag_manager_instance
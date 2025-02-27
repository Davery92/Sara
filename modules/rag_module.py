"""
RAG (Retrieval-Augmented Generation) module for the OpenAI-compatible server.
This module handles document processing, chunking, embedding, and retrieval.
"""

import os
import json
import uuid
import logging
import re
import numpy as np
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import ollama
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tiktoken
from redis.commands.search.field import TextField, VectorField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from rank_bm25 import BM25Okapi
import traceback

# Configure logging
logger = logging.getLogger("rag-module")

DOCUMENTS_DIRECTORY = "/home/david/Sara/documents"
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 100  # tokens
TOP_K_INITIAL = 15  # Initial retrieval count
TOP_K_FINAL = 3  # Final count after reranking
EMBEDDING_MODEL = "bge-m3"  # Ollama embedding model
EMBEDDING_DIMENSION = 1024  # BGE-M3 embedding dimension
# Update to use the local model path
RERANKER_MODEL_PATH = "/home/david/Sara/ms-marco-MiniLM-L-6-v2" 

# Ensure documents directory exists
os.makedirs(DOCUMENTS_DIRECTORY, exist_ok=True)

# Initialize tokenizer for chunking
tokenizer_name = "gpt-3.5-turbo"  # Use OpenAI-compatible tokenizer for chunk sizing
tokenizer = tiktoken.encoding_for_model(tokenizer_name)

# Initialize reranker model (lazy loading to save memory)
reranker_model = None
reranker_tokenizer = None

def get_reranker():
    """Lazily load the reranker model from local directory"""
    global reranker_model, reranker_tokenizer
    if reranker_model is None:
        try:
            logger.info(f"Loading reranker model from local path: {RERANKER_MODEL_PATH}")
            # Load from local directory
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
            reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
            logger.info("Local reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading local reranker model: {e}")
            # Create a simple fallback reranker
            logger.warning("Using fallback BM25-style reranking")
            reranker_model = None
            reranker_tokenizer = None
    return reranker_model, reranker_tokenizer

class RAGManager:

    DOCUMENTS_DIRECTORY = "/home/david/Sara/documents"

    def __init__(self, redis_client):
        """Initialize the RAG manager with Redis client"""
        self.redis_client = redis_client
        self.docs_index_name = "docs_idx"
        self.chunks_index_name = "chunks_idx"
        self._ensure_indices()

        os.makedirs(self.DOCUMENTS_DIRECTORY, exist_ok=True)
        
    def _ensure_indices(self):
        """Create the necessary Redis indices if they don't exist"""
        # Try to create document index
        try:
            self.redis_client.redis_client.ft(self.docs_index_name).info()
            logger.info("Document index already exists")
        except:
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
            try:
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
        
        # Try to create chunks index
        try:
            self.redis_client.redis_client.ft(self.chunks_index_name).info()
            logger.info("Chunks index already exists")
        except:
            # Define schema for chunks
            chunks_schema = (
                TextField("$.text", as_name="text"),
                TextField("$.doc_id", as_name="doc_id"),
                NumericField("$.chunk_index", as_name="chunk_index"),
                TextField("$.metadata.title", as_name="title"),  # Document title
                TextField("$.metadata.filename", as_name="filename"),  # Document filename
                TagField("$.metadata.tags", as_name="tags"),  # Document tags
                VectorField("$.embedding", 
                           "HNSW", {
                               "TYPE": "FLOAT32", 
                               "DIM": EMBEDDING_DIMENSION,
                               "DISTANCE_METRIC": "COSINE"
                           }, as_name="embedding")
            )
            
            # Create index
            try:
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
    
    def _create_clean_filename(self, title: str, content_type: str) -> str:
        """
        Create a clean filename from the title and content type
        that is safe for filesystem use
        """
        # Remove any characters that might cause issues in filenames
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        
        # Get the appropriate extension based on content type
        extension = content_type.split('/')[-1]
        if extension == 'octet-stream':
            # Try to guess extension from the title if content_type is generic
            title_ext = os.path.splitext(title)[1]
            if title_ext:
                extension = title_ext.lstrip('.')
            else:
                extension = 'bin'  # Default binary extension
        
        # Create the filename with extension
        return f"{clean_title}.{extension}"

    def process_document(self, file_path: str, filename: str, content_type: str, 
                      title: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a document:
        1. Store document metadata
        2. Extract text content
        3. Chunk the text
        4. Embed chunks
        5. Store chunks with embeddings
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
            clean_title = self._create_clean_filename(title, content_type)
            
            doc_metadata = {
                "id": doc_id,
                "title": title,
                "filename": filename,
                "storage_filename": clean_title,  # Add the clean filename to metadata
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
                self._process_chunk(chunk, doc_id, i, doc_metadata)
            
            # Calculate document hash for deduplication
            content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
            doc_metadata["content_hash"] = content_hash
            
            # Update document with hash
            self.redis_client.redis_client.json().set(f"doc:{doc_id}", '$', doc_metadata)
            
            # Save document to disk as backup - MODIFIED TO USE BETTER NAMING
            # Create a directory structure for better organization
            doc_storage_dir = os.path.join(DOCUMENTS_DIRECTORY, doc_id)
            os.makedirs(doc_storage_dir, exist_ok=True)
            
            # Save with original title for usability
            original_doc_path = os.path.join(doc_storage_dir, clean_title)
            with open(original_doc_path, 'wb') as f:
                f.write(file_content)
            
            # Also save JSON metadata for easy retrieval
            metadata_path = os.path.join(doc_storage_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(doc_metadata, f, indent=2)
            
            # Add paths to metadata
            doc_metadata["file_path"] = original_doc_path
            doc_metadata["metadata_path"] = metadata_path
            self.redis_client.redis_client.json().set(f"doc:{doc_id}", '$', doc_metadata)
            
            logger.info(f"Processed document: {title} ({doc_id}) with {len(chunks)} chunks, saved to {original_doc_path}")
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return {"error": str(e)}
    
    def _extract_text(self, file_path: str, content_type: str) -> str:
        """Extract text content from various file types"""
        try:
            # Plain text files
            if content_type == 'text/plain':
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
                    
            # Markdown files
            elif content_type == 'text/markdown':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            
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
        # Tokenize the text
        tokens = tokenizer.encode(text)
        
        # Initialize chunks
        chunks = []
        i = 0
        
        # Create chunks with overlap
        while i < len(tokens):
            # Get the chunk
            chunk_end = min(i + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            
            # Decode the chunk
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk (remove excessive whitespace)
            chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            
            # Add to chunks if not empty
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            i += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            # Make sure we make progress
            if i <= 0:
                i = chunk_end
        
        return chunks
    
    def _process_chunk(self, chunk_text: str, doc_id: str, chunk_index: int, doc_metadata: Dict[str, Any]):
        """
        Process a single chunk with completely separate keys for JSON and vector data
        """
        try:
            # Generate embedding using Ollama
            embedding_result = ollama.embeddings(
                model=EMBEDDING_MODEL,
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
            
            # 1. Delete existing keys if they exist
            keys_to_delete = [
                chunk_json_key, 
                chunk_vector_key,
                # Also delete the old combined key format
                f"chunk:{doc_id}:{chunk_index}"
            ]
            
            for key in keys_to_delete:
                if self.redis_client.redis_client.exists(key):
                    self.redis_client.redis_client.delete(key)
            
            # 2. Store chunk JSON data
            self.redis_client.redis_client.json().set(chunk_json_key, '$', chunk_data)
            
            # 3. Store vector data separately using a string key with binary value
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            self.redis_client.redis_binary.set(chunk_vector_key, embedding_bytes)
            
            # 4. Create an index record that maps the original format to our new keys
            # This helps with backward compatibility for searching
            index_data = {
                "json_key": chunk_json_key,
                "vector_key": chunk_vector_key
            }
            self.redis_client.redis_client.json().set(f"chunk:{doc_id}:{chunk_index}", '$', index_data)
            
            logger.info(f"Processed chunk {chunk_index} of document {doc_id} with separate keys")
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} of doc {doc_id}: {e}")
    
    def search(self, query: str, top_k: int = TOP_K_FINAL, 
               filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks:
        1. Perform semantic search with embeddings
        2. Optional: Perform BM25 search
        3. Rerank results
        4. Return top matches
        """
        try:
            # First, perform semantic search
            semantic_results = self._semantic_search(query, TOP_K_INITIAL, filter_tags)
            
            # If we have enough results, rerank and return
            if len(semantic_results) >= top_k:
                reranked_results = self._rerank_results(query, semantic_results)
                return reranked_results[:top_k]
            
            # If not enough results, also try BM25 search
            logger.info(f"Semantic search returned only {len(semantic_results)} results, adding BM25 results")
            bm25_results = self._bm25_search(query, TOP_K_INITIAL, filter_tags)
            
            # Combine results (avoiding duplicates)
            combined_results = semantic_results.copy()
            seen_chunk_ids = {result["chunk_id"] for result in semantic_results}
            
            for result in bm25_results:
                if result["chunk_id"] not in seen_chunk_ids:
                    combined_results.append(result)
                    seen_chunk_ids.add(result["chunk_id"])
                    
                    # Stop once we have enough
                    if len(combined_results) >= TOP_K_INITIAL:
                        break
            
            # Rerank the combined results
            reranked_results = self._rerank_results(query, combined_results)
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def _semantic_search(self, query: str, top_k: int = TOP_K_INITIAL,
                    filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using basic string matching since vector search is failing"""
        try:
            # First try to get all chunks to do manual filtering
            all_chunks = self._get_all_chunks(filter_tags)
            
            if not all_chunks:
                logger.warning("No chunks found for searching")
                return []
                
            # Get query embedding for potential similarity calculation
            embedding_result = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=query
            )
            query_embedding = embedding_result.get('embedding', [])
            
            # Simple text matching search
            results = []
            for chunk in all_chunks:
                # Calculate a simple text match score (count occurrences of query terms)
                text = chunk.get("text", "").lower()
                query_terms = query.lower().split()
                
                # Count matches in text
                base_score = 0
                for term in query_terms:
                    if term in text:
                        base_score += text.count(term)
                
                # Only include results with at least some match
                if base_score > 0:
                    results.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "text": chunk.get("text", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "title": chunk.get("title", ""),
                        "score": float(base_score),
                        "search_type": "text_match"
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top results
            top_results = results[:top_k]
            
            logger.info(f"Basic text search returned {len(top_results)} results")
            return top_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    
    def _bm25_search(self, query: str, top_k: int = TOP_K_INITIAL,
                    filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform BM25 search"""
        try:
            # First, get all chunks (potentially limited by tags)
            all_chunks = self._get_all_chunks(filter_tags)
            
            if not all_chunks:
                logger.warning("No chunks found for BM25 search")
                return []
            
            # Extract texts and tokenize
            tokenized_corpus = [doc["text"].lower().split() for doc in all_chunks]
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Create results with scores
            results_with_scores = []
            for i, (doc, score) in enumerate(zip(all_chunks, doc_scores)):
                results_with_scores.append({
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "score": float(score),
                    "search_type": "bm25"
                })
            
            # Sort by score (descending) and take top_k
            results_with_scores.sort(key=lambda x: x["score"], reverse=True)
            top_results = results_with_scores[:top_k]
            
            logger.info(f"BM25 search returned {len(top_results)} results")
            return top_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _get_all_chunks(self, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get all chunks using direct key iteration"""
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
                        "title": chunk_data.get("metadata", {}).get("title", "")
                    })
                except Exception as e:
                    logger.error(f"Error processing chunk {key}: {e}")
            
            logger.info(f"Found {len(processed_results)} chunks")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error getting all chunks: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using locally downloaded cross-encoder model"""
        try:
            if not results:
                return []
            
            # Get texts from results
            texts = [result["text"] for result in results]
            
            # Get the reranker model
            model, tokenizer = get_reranker()
            
            # If model is None, use a simple fallback
            if model is None or tokenizer is None:
                logger.warning("Using fallback reranking (preserving original order)")
                # Just use the original scores and order
                for i, result in enumerate(results):
                    if "rerank_score" not in result:
                        result["rerank_score"] = result.get("score", 1.0 - (i * 0.01))  # Preserve original order
                
                results.sort(key=lambda x: x["rerank_score"], reverse=True)
                return results
                
            # Prepare pairs for reranking
            pairs = [[query, text] for text in texts]
            
            # Tokenize - add error handling for each pair
            processed_pairs = []
            valid_indices = []
            
            for i, pair in enumerate(pairs):
                try:
                    # Tokenize individual pairs to handle errors gracefully
                    encoded_pair = tokenizer(
                        pair[0],
                        pair[1],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    processed_pairs.append(encoded_pair)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Error tokenizing pair {i}: {e}")
            
            # If we have valid pairs, process them
            if processed_pairs:
                scores_list = []
                
                # Process each pair individually to avoid batch errors
                with torch.no_grad():
                    for encoded_pair in processed_pairs:
                        try:
                            score = model(**encoded_pair).logits.squeeze().item()
                            scores_list.append(score)
                        except Exception as e:
                            logger.warning(f"Error scoring pair: {e}")
                            scores_list.append(0.0)  # Default score on error
                
                # Add scores to results for valid pairs
                for valid_idx, score in zip(valid_indices, scores_list):
                    results[valid_idx]["rerank_score"] = float(score)
            
            # Make sure all results have rerank_score
            for result in results:
                if "rerank_score" not in result:
                    result["rerank_score"] = result.get("score", 0.0)
            
            # Sort by reranker score (descending)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Reranked {len(results)} results using local model")
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # If reranking fails, return original results sorted by their original scores
            for result in results:
                if "rerank_score" not in result:
                    result["rerank_score"] = result.get("score", 0.0)
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
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
    
    def list_documents(self, limit: int = 100, offset: int = 0, 
                  filter_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """List all documents with pagination and filtering - with improved error handling"""
        try:
            # Build query based on filters
            if filter_tags and len(filter_tags) > 0:
                tag_filters = " | ".join([f"@tags:{{{tag}}}" for tag in filter_tags])
                query = f"({tag_filters})"
            else:
                query = "*"
            
            # Try to execute search via index
            try:
                results = self.redis_client.redis_client.ft(self.docs_index_name).search(
                    query,
                    limit=offset, limit_num=limit,
                    sort_by="date_added", sort_desc=True
                )
                
                # Process results
                docs = []
                for doc in results.docs:
                    try:
                        doc_data = json.loads(doc.json)
                        docs.append(doc_data)
                    except Exception as e:
                        logger.error(f"Error parsing document JSON: {e}")
                        # Skip this document but continue processing
                        continue
                        
                return {
                    "documents": docs,
                    "total": results.total,
                    "offset": offset,
                    "limit": limit
                }
                
            except Exception as e:
                logger.warning(f"Search index error: {e}, falling back to manual listing")
                
                # Fallback: manual listing from Redis keys
                doc_keys = self.redis_client.redis_client.keys("doc:*")
                total_count = len(doc_keys)
                
                # Sort and paginate keys (basic implementation)
                sorted_keys = sorted(doc_keys, reverse=True)
                paginated_keys = sorted_keys[offset:offset+limit]
                
                # Retrieve document data
                docs = []
                for key in paginated_keys:
                    try:
                        doc_data = self.redis_client.redis_client.json().get(key)
                        
                        # Filter by tags if needed
                        if filter_tags and len(filter_tags) > 0:
                            doc_tags = doc_data.get("tags", [])
                            if not any(tag in doc_tags for tag in filter_tags):
                                continue
                                
                        docs.append(doc_data)
                    except Exception as doc_e:
                        logger.error(f"Error retrieving document {key}: {doc_e}")
                        # Skip this document but continue processing
                        continue
                
                return {
                    "documents": docs,
                    "total": total_count,
                    "offset": offset,
                    "limit": limit,
                    "note": "Used fallback listing method"
                }
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            # Always return a valid JSON response, even in case of error
            return {
                "documents": [],
                "total": 0,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document and all its chunks - Updated to handle new storage structure"""
        try:
            # Get document first to verify it exists
            doc_key = f"doc:{doc_id}"
            if not self.redis_client.redis_client.exists(doc_key):
                return {"error": f"Document not found: {doc_id}"}
            
            # Get document data for the response
            doc_data = self.redis_client.redis_client.json().get(doc_key)
            
            # Delete all chunks
            chunk_prefix = f"chunk:{doc_id}:"
            chunk_keys = self.redis_client.redis_client.keys(f"{chunk_prefix}*")
            
            if chunk_keys:
                self.redis_client.redis_client.delete(*chunk_keys)
            
            # Delete document
            self.redis_client.redis_client.delete(doc_key)
            
            # Delete document directory from disk if it exists
            doc_dir_path = os.path.join(DOCUMENTS_DIRECTORY, doc_id)
            if os.path.exists(doc_dir_path):
                shutil.rmtree(doc_dir_path)
            
            logger.info(f"Deleted document {doc_id} with {len(chunk_keys)} chunks")
            
            return {
                "success": True,
                "document": doc_data,
                "chunks_deleted": len(chunk_keys)
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return {"error": str(e)}
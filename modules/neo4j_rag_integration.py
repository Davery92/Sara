"""
Neo4j-based RAG implementation.
Replaces Redis storage with Neo4j for document chunks and relationships.
Uses LLama model to extract semantic information and build relationships.
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
from rank_bm25 import BM25Okapi
import traceback
from neo4j import GraphDatabase
import asyncio

# Configure logging
logger = logging.getLogger("neo4j-rag")

DOCUMENTS_DIRECTORY = "/home/david/Sara/documents"
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 100  # tokens
TOP_K_INITIAL = 15  # Initial retrieval count
TOP_K_FINAL = 3  # Final count after reranking
EMBEDDING_MODEL = "bge-m3"  # Ollama embedding model
EMBEDDING_DIMENSION = 1024  # BGE-M3 embedding dimension
SEMANTIC_MODEL = "llama3.1"  # Model for semantic analysis

# Ensure documents directory exists
os.makedirs(DOCUMENTS_DIRECTORY, exist_ok=True)

# Initialize tokenizer for chunking
tokenizer_name = "gpt-3.5-turbo"  # Use OpenAI-compatible tokenizer for chunk sizing
tokenizer = tiktoken.encoding_for_model(tokenizer_name)

class Neo4jRAGManager:
    """RAG Manager that uses Neo4j for document storage and retrieval"""

    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_db):
        """Initialize with Neo4j connection details"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_db = neo4j_db
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Try to create constraints, but continue if it fails
        try:
            self._ensure_constraints()
        except Exception as e:
            logger.error(f"Error creating constraints: {e}")
            # Try to continue without constraints
            if not self.initialize_without_constraints():
                raise Exception(f"Failed to initialize Neo4j connection: {e}")
        
        # Directories
        self.DOCUMENTS_DIRECTORY = DOCUMENTS_DIRECTORY
        os.makedirs(self.DOCUMENTS_DIRECTORY, exist_ok=True)
        
        logger.info("Neo4j RAG Manager initialized")
    
    def initialize_without_constraints(self):
        """
        Initialize the database without constraints
        Used as a fallback when constraint creation fails
        """
        try:
            with self.driver.session(database=self.neo4j_db) as session:
                # Just create a test node to verify connection
                session.run("CREATE (t:TestConnection {id: $id}) RETURN t", 
                        id=str(uuid.uuid4()))
                session.run("MATCH (t:TestConnection) DELETE t")
                logger.info("Neo4j connection verified without constraints")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j even without constraints: {e}")
            return False


    def _ensure_constraints(self):
        """Create necessary constraints and indices in Neo4j"""
        try:
            with self.driver.session(database=self.neo4j_db) as session:
                # Neo4j 5.x syntax (newer version)
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.date_added)")
                
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)")
                
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE")
                    
                logger.info("Neo4j constraints and indices created")
        except Exception as e:
            logger.error(f"Error creating constraints: {e}")
            # Continue without constraints - they might already exist or we lack permissions
            logger.warning("Continuing without constraints - indexes may be missing")
    async def process_document(self, file_path: str, filename: str, content_type: str, 
                        title: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a document:
        1. Extract text and metadata
        2. Chunk the text
        3. Extract semantic information from chunks
        4. Create Neo4j graph with documents, chunks, entities, concepts
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
                "storage_filename": clean_title,
                "content_type": content_type,
                "size": file_size,
                "tags": tags or [],
                "date_added": datetime.now().isoformat()
            }
            
            # Chunk the text
            chunks = self._chunk_text(text_content)
            doc_metadata["chunk_count"] = len(chunks)
            
            # Store document in Neo4j
            self._store_document(doc_id, doc_metadata)
            
            # Process chunks sequentially for better memory management
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}:{i}"
                await self._process_chunk(chunk, chunk_id, doc_id, i, doc_metadata)
            
            # Calculate document hash for deduplication
            content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
            doc_metadata["content_hash"] = content_hash
            
            # Update document with hash and chunk count
            self._update_document(doc_id, {
                "content_hash": content_hash,
                "chunk_count": len(chunks)
            })
            
            # Save document to disk as backup
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
            self._update_document(doc_id, {
                "file_path": original_doc_path,
                "metadata_path": metadata_path
            })
            
            logger.info(f"Processed document: {title} ({doc_id}) with {len(chunks)} chunks, saved to {original_doc_path}")
            
            # Create semantic relationships between chunks
            await self._create_chunk_relationships(doc_id)
            
            return doc_metadata
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            logger.exception(e)
            return {"error": str(e)}
    
    def _store_document(self, doc_id: str, metadata: Dict[str, Any]):
        """Store document metadata in Neo4j"""
        query = """
        CREATE (d:Document {
            id: $id,
            title: $title,
            filename: $filename,
            storage_filename: $storage_filename,
            content_type: $content_type,
            size: $size,
            date_added: $date_added,
            chunk_count: $chunk_count
        })
        """
        
        # Add tags as nodes with relationships
        tag_query = """
        MATCH (d:Document {id: $doc_id})
        MERGE (t:Tag {name: $tag_name})
        CREATE (d)-[:HAS_TAG]->(t)
        """
        
        with self.driver.session(database=self.neo4j_db) as session:
            # Store document
            session.run(
                query,
                id=doc_id,
                title=metadata["title"],
                filename=metadata["filename"],
                storage_filename=metadata["storage_filename"],
                content_type=metadata["content_type"],
                size=metadata["size"],
                date_added=metadata["date_added"],
                chunk_count=metadata.get("chunk_count", 0)
            )
            
            # Store tags
            for tag in metadata.get("tags", []):
                session.run(tag_query, doc_id=doc_id, tag_name=tag)
    
    def _update_document(self, doc_id: str, updates: Dict[str, Any]):
        """Update document properties in Neo4j"""
        # Build dynamic SET clause based on updates
        set_clauses = []
        params = {"doc_id": doc_id}
        
        for key, value in updates.items():
            param_name = f"param_{key}"
            set_clauses.append(f"d.{key} = ${param_name}")
            params[param_name] = value
        
        if not set_clauses:
            return
        
        query = f"""
        MATCH (d:Document {{id: $doc_id}})
        SET {", ".join(set_clauses)}
        """
        
        with self.driver.session(database=self.neo4j_db) as session:
            session.run(query, **params)
    
    def _create_clean_filename(self, title: str, content_type: str) -> str:
        """Create a clean filename from the title and content type"""
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
        """Chunk text into segments with specified token size and overlap"""
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
    
    async def _process_chunk(self, chunk_text: str, chunk_id: str, doc_id: str, 
                          chunk_index: int, doc_metadata: Dict[str, Any]):
        """
        Process a single chunk:
        1. Generate embedding
        2. Extract entities, concepts, and other semantic info using LLama
        3. Store in Neo4j with relationships
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
            
            # Extract semantic information using LLama
            semantic_info = await self._extract_semantic_info(chunk_text)
            
            # Store chunk in Neo4j
            self._store_chunk(chunk_id, doc_id, chunk_index, chunk_text, 
                            doc_metadata, embedding, semantic_info)
            
            logger.info(f"Processed chunk {chunk_index} of document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} of doc {doc_id}: {e}")
            logger.exception(e)
    
    async def _extract_semantic_info(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic information from text using LLama model:
        - Entities (people, places, organizations, etc.)
        - Concepts (abstract ideas, themes)
        - Key points (main ideas, facts)
        - Sentiment
        - Relevance
        """
        try:
            # Create a prompt for LLama to extract information
            prompt = f"""
            Analyze the following text and extract semantic information in JSON format.
            Include these categories:
            - entities: List of named entities (people, places, organizations, etc.) as simple strings
            - concepts: List of abstract concepts and themes as simple strings
            - key_points: List of main ideas or facts as simple strings
            - sentiment: Overall sentiment (positive, negative, or neutral) as a simple string
            - relevance: Rate the relevance and importance of this text (1-10) as a number

            Format the output as a valid JSON object with these fields.
            All values must be simple strings, numbers, or arrays of strings - not complex objects.

            Text to analyze:
            {text}
            """
            
            # Using the ollama API
            response = ollama.chat(
                model=SEMANTIC_MODEL,
                messages=[
                    {"role": "system", "content": "You are a semantic analysis assistant that extracts structured information from text and outputs only valid JSON with simple values - no nested objects or complex structures."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.get('message', {}).get('content', '')
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no JSON found, use the whole response
                    json_str = content
            
            # Parse JSON
            try:
                data = json.loads(json_str)
                
                # Process all fields to ensure they are primitive types
                # Entities - ensure they're simple strings
                processed_entities = []
                for entity in data.get('entities', []):
                    if isinstance(entity, dict):
                        if 'name' in entity:
                            processed_entities.append(entity['name'])
                    elif entity:
                        processed_entities.append(str(entity))
                data['entities'] = processed_entities
                
                # Concepts - ensure they're simple strings
                processed_concepts = []
                for concept in data.get('concepts', []):
                    if isinstance(concept, dict):
                        if 'name' in concept:
                            processed_concepts.append(concept['name'])
                    elif concept:
                        processed_concepts.append(str(concept))
                data['concepts'] = processed_concepts
                
                # Key points - ensure they're simple strings
                processed_key_points = []
                for point in data.get('key_points', []):
                    if isinstance(point, dict):
                        if 'text' in point:
                            processed_key_points.append(point['text'])
                    elif point:
                        processed_key_points.append(str(point))
                data['key_points'] = processed_key_points
                
                # Sentiment - ensure it's a simple string
                sentiment = data.get('sentiment', 'neutral')
                if isinstance(sentiment, dict):
                    if 'analysis' in sentiment:
                        data['sentiment'] = sentiment['analysis']
                    else:
                        # Extract first string value from the dict
                        for val in sentiment.values():
                            if isinstance(val, str):
                                data['sentiment'] = val
                                break
                        else:
                            data['sentiment'] = 'neutral'
                else:
                    data['sentiment'] = str(sentiment)
                
                # Relevance - ensure it's a number
                relevance = data.get('relevance', 5)
                if isinstance(relevance, dict) or not isinstance(relevance, (int, float)):
                    data['relevance'] = 5
                
                # Ensure default values
                data.setdefault('entities', [])
                data.setdefault('concepts', [])
                data.setdefault('key_points', [])
                data.setdefault('sentiment', 'neutral')
                data.setdefault('relevance', 5)
                
                return data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLama response: {content}")
                # Return default structure
                return {
                    'entities': [],
                    'concepts': [],
                    'key_points': [],
                    'sentiment': 'neutral',
                    'relevance': 5
                }
                    
        except Exception as e:
            logger.error(f"Error extracting semantic info: {e}")
            logger.exception(e)
            # Return default structure
            return {
                'entities': [],
                'concepts': [],
                'key_points': [],
                'sentiment': 'neutral',
                'relevance': 5
            }
    
    def _store_chunk(self, chunk_id: str, doc_id: str, chunk_index: int, 
              chunk_text: str, doc_metadata: Dict[str, Any], 
              embedding: List[float], semantic_info: Dict[str, Any]):
        """
        Store a chunk and its semantic information in Neo4j
        Creates relationships between the chunk and:
        - Parent document
        - Entities
        - Concepts
        """
        # For efficient storage and query, we'll store just a limited vector sample
        embedding_sample = embedding[:10]  # First 10 dimensions as a sample
        
        # Validate all values to ensure they are primitive types
        # Ensure sentiment is a string
        sentiment = semantic_info.get('sentiment', 'neutral')
        if not isinstance(sentiment, str):
            sentiment = str(sentiment)
        
        # Ensure relevance is a number
        relevance = semantic_info.get('relevance', 5)
        if not isinstance(relevance, (int, float)):
            relevance = 5
        
        # Base query to create the chunk node
        chunk_query = """
        MATCH (d:Document {id: $doc_id})
        CREATE (c:Chunk {
            id: $chunk_id,
            doc_id: $doc_id,
            chunk_index: $chunk_index,
            text: $text,
            sentiment: $sentiment,
            relevance: $relevance,
            embedding_sample: $embedding_sample,
            embedding_dimensions: $embedding_dimensions
        })
        CREATE (d)-[:CONTAINS]->(c)
        RETURN c
        """
        
        # Queries for creating relationships with semantic elements
        entity_query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e:Entity {name: $entity_name})
        CREATE (c)-[:MENTIONS]->(e)
        """
        
        concept_query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (co:Concept {name: $concept_name})
        CREATE (c)-[:RELATES_TO]->(co)
        """
        
        key_point_query = """
        MATCH (c:Chunk {id: $chunk_id})
        CREATE (k:KeyPoint {text: $key_point, chunk_id: $chunk_id})
        CREATE (c)-[:CONTAINS_POINT]->(k)
        """
        
        with self.driver.session(database=self.neo4j_db) as session:
            # Create chunk node
            session.run(
                chunk_query,
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                sentiment=sentiment,
                relevance=relevance,
                embedding_sample=embedding_sample,
                embedding_dimensions=len(embedding)
            )
            
            # Create entity relationships
            for entity in semantic_info.get('entities', []):
                # Skip if entity is None or empty
                if not entity:
                    continue
                    
                # Handle entities that may be dictionaries
                if isinstance(entity, dict):
                    # Extract the name from the entity dictionary
                    entity_name = entity.get('name', '')
                else:
                    # It's already a string or another primitive type
                    entity_name = str(entity)
                    
                # Only create entities with meaningful names
                if entity_name and len(entity_name) > 1:  # Ignore very short entities
                    try:
                        session.run(entity_query, chunk_id=chunk_id, entity_name=entity_name)
                    except Exception as e:
                        logger.warning(f"Error creating entity {entity_name}: {e}")
                        continue
            
            # Create concept relationships
            for concept in semantic_info.get('concepts', []):
                # Skip if concept is None or empty
                if not concept:
                    continue
                    
                # Handle concepts that may be dictionaries
                if isinstance(concept, dict):
                    # Extract the name from the concept dictionary
                    concept_name = concept.get('name', '')
                else:
                    # It's already a string or another primitive type
                    concept_name = str(concept)
                    
                # Only create concepts with meaningful names
                if concept_name and len(concept_name) > 1:  # Ignore very short concepts
                    try:
                        session.run(concept_query, chunk_id=chunk_id, concept_name=concept_name)
                    except Exception as e:
                        logger.warning(f"Error creating concept {concept_name}: {e}")
                        continue
            
            # Create key point nodes
            for key_point in semantic_info.get('key_points', []):
                # Skip if key_point is None or empty
                if not key_point:
                    continue
                    
                # Handle key_points that may be dictionaries
                if isinstance(key_point, dict):
                    # Extract the text from the key_point dictionary
                    key_point_text = key_point.get('text', '')
                else:
                    # It's already a string or another primitive type
                    key_point_text = str(key_point)
                    
                # Only create key points with meaningful text
                if key_point_text and len(key_point_text) > 5:  # Ignore very short points
                    try:
                        session.run(key_point_query, chunk_id=chunk_id, key_point=key_point_text)
                    except Exception as e:
                        logger.warning(f"Error creating key point: {e}")
                        continue
    
    async def _create_chunk_relationships(self, doc_id: str):
        """
        Create semantic relationships between chunks in the same document
        Uses LLama to determine relatedness between chunks
        """
        logger.info(f"Creating semantic relationships between chunks for document {doc_id}")
        
        # Get all chunks for this document
        chunks = self._get_document_chunks(doc_id)
        
        if len(chunks) <= 1:
            logger.info(f"Document {doc_id} has {len(chunks)} chunks, skipping relationship creation")
            return
        
        # Create batches of chunk pairs to analyze
        # We'll create a full relationship graph but limit comparisons for efficiency
        
        # Define maximum number of connections per chunk to limit complexity
        max_connections = min(5, len(chunks) - 1)
        
        # Create a relationship query
        relationship_query = """
        MATCH (c1:Chunk {id: $chunk1_id})
        MATCH (c2:Chunk {id: $chunk2_id})
        CREATE (c1)-[:RELATED_TO {strength: $strength, relationship_type: $rel_type}]->(c2)
        """
        
        # Process chunks in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            for chunk in batch:
                # Get candidate chunks to compare with
                other_chunks = [c for c in chunks if c['id'] != chunk['id']]
                
                # Select chunks to compare (nearest neighbors by index + some random ones)
                chunk_index = chunk['chunk_index']
                
                # Select neighboring chunks first
                neighbors = sorted(other_chunks, 
                                 key=lambda c: abs(c['chunk_index'] - chunk_index))[:max_connections]
                
                # Compare with each neighbor
                for neighbor in neighbors:
                    try:
                        # Determine the relationship type and strength
                        relationship = await self._analyze_chunk_relationship(
                            chunk['text'], 
                            neighbor['text']
                        )
                        
                        # Only create relationships if they're meaningful
                        if relationship['strength'] > 0.5:
                            with self.driver.session(database=self.neo4j_db) as session:
                                session.run(
                                    relationship_query,
                                    chunk1_id=chunk['id'],
                                    chunk2_id=neighbor['id'],
                                    strength=relationship['strength'],
                                    rel_type=relationship['type']
                                )
                    except Exception as e:
                        logger.error(f"Error analyzing relationship between chunks: {e}")
                        continue
    
    async def _analyze_chunk_relationship(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Analyze the relationship between two text chunks
        Returns the type of relationship and strength
        """
        try:
            # Create a shortened version of texts if they're too long
            max_length = 500
            short_text1 = text1[:max_length] + ("..." if len(text1) > max_length else "")
            short_text2 = text2[:max_length] + ("..." if len(text2) > max_length else "")
            
            # Create a prompt for LLama
            prompt = f"""
            Analyze the relationship between these two text passages and provide the following in JSON format:
            1. type: The type of relationship (continuation, elaboration, contrast, example, etc.)
            2. strength: A numeric value from 0 to 1 indicating how strongly related they are (0=unrelated, 1=strongly related)

            Text 1:
            {short_text1}

            Text 2:
            {short_text2}
            """
            
            # Using the ollama API
            response = ollama.chat(
                model=SEMANTIC_MODEL,
                messages=[
                    {"role": "system", "content": "You are a text analysis assistant that determines relationships between texts and outputs only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.get('message', {}).get('content', '')
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no JSON found, use the whole response
                    json_str = content
            
            # Parse JSON
            try:
                data = json.loads(json_str)
                
                # Ensure required fields are present
                if 'type' not in data:
                    data['type'] = 'related'
                
                if 'strength' not in data:
                    data['strength'] = 0.5
                elif not isinstance(data['strength'], (int, float)):
                    # Convert string to float if necessary
                    try:
                        data['strength'] = float(data['strength'])
                    except ValueError:
                        data['strength'] = 0.5
                
                # Ensure strength is between 0 and 1
                data['strength'] = max(0, min(1, data['strength']))
                
                return data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLama response: {content}")
                return {
                    'type': 'related',
                    'strength': 0.5
                }
                
        except Exception as e:
            logger.error(f"Error analyzing chunk relationship: {e}")
            return {
                'type': 'related',
                'strength': 0.5
            }
    
    def _get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document from Neo4j"""
        query = """
        MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)
        RETURN c.id as id, c.chunk_index as chunk_index, c.text as text
        ORDER BY c.chunk_index
        """
        
        with self.driver.session(database=self.neo4j_db) as session:
            result = session.run(query, doc_id=doc_id)
            return [dict(record) for record in result]
    
    def search(self, query: str, top_k: int = TOP_K_FINAL, 
            filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using Neo4j
        Combines semantic search using embeddings, entity/concept matching, 
        and graph relationships
        """
        try:
            # Get embedding for the query
            embedding_result = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=query
            )
            embedding = embedding_result.get('embedding', [])
            
            # Extract entities and concepts from the query
            semantic_info = asyncio.run(self._extract_semantic_info(query))
            entities = semantic_info.get('entities', [])
            concepts = semantic_info.get('concepts', [])
            
            results = []
            
            # If we have entities or concepts, use semantic search
            if entities or concepts:
                # Build a comprehensive search query
                search_query = """
                // Match chunks based on a combination of factors
                
                // 1. First, try exact entity matches
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $entities
                
                WITH c, count(DISTINCT e) as entity_matches
                
                // 2. Add concept matches
                OPTIONAL MATCH (c)-[:RELATES_TO]->(co:Concept)
                WHERE co.name IN $concepts
                
                WITH c, entity_matches, count(DISTINCT co) as concept_matches
                
                // 3. Get document info and check tags if specified
                MATCH (d:Document)-[:CONTAINS]->(c)
                """
                
                # Add tag filtering if requested
                if filter_tags and len(filter_tags) > 0:
                    search_query += """
                    MATCH (d)-[:HAS_TAG]->(t:Tag)
                    WHERE t.name IN $tags
                    """
                
                # Complete the query with scoring and results
                search_query += """
                // Calculate a composite score with weights
                WITH c, d, 
                    entity_matches * 3.0 + concept_matches * 2.0 AS semantic_score,
                    c.relevance AS relevance
                    
                // Calculate a final score combining all factors
                WITH c, d,
                    (semantic_score * 0.6) + (relevance * 0.4) AS final_score
                    
                WHERE final_score > 0
                
                // Return results with document info
                RETURN c.id AS chunk_id,
                    c.text AS text,
                    d.id AS doc_id,
                    d.title AS title,
                    final_score AS score
                ORDER BY final_score DESC
                LIMIT $limit
                """
                
                # Run the semantic search query
                params = {
                    "entities": entities,
                    "concepts": concepts,
                    "limit": top_k
                }
                
                if filter_tags and len(filter_tags) > 0:
                    params["tags"] = filter_tags
                
                with self.driver.session(database=self.neo4j_db) as session:
                    result = session.run(search_query, **params)
                    semantic_results = [dict(record) for record in result]
                
                results.extend(semantic_results)
            
            # If we have fewer than top_k results or no entities/concepts, 
            # try text search as well
            if not results or len(results) < top_k:
                # Fall back to a simpler search using text matching
                text_search_query = """
                // Text-based search
                MATCH (c:Chunk)
                WHERE c.text CONTAINS $query_text
                
                // Get document info and check tags if specified
                MATCH (d:Document)-[:CONTAINS]->(c)
                """
                
                # Add tag filtering if requested
                if filter_tags and len(filter_tags) > 0:
                    text_search_query += """
                    MATCH (d)-[:HAS_TAG]->(t:Tag)
                    WHERE t.name IN $tags
                    """
                
                # Complete the query with scoring and results
                text_search_query += """
                // Calculate text match score based on relevance
                WITH c, d, c.relevance AS relevance
                
                // Return results with document info
                RETURN c.id AS chunk_id,
                    c.text AS text,
                    d.id AS doc_id,
                    d.title AS title,
                    relevance AS score
                ORDER BY relevance DESC
                LIMIT $limit
                """
                
                # Run the text search query
                params = {
                    "query_text": query,
                    "limit": top_k
                }
                
                if filter_tags and len(filter_tags) > 0:
                    params["tags"] = filter_tags
                
                with self.driver.session(database=self.neo4j_db) as session:
                    result = session.run(text_search_query, **params)
                    text_results = [dict(record) for record in result]
                
                # Merge both result sets, removing duplicates
                existing_chunk_ids = {r["chunk_id"] for r in results}
                for result in text_results:
                    if result["chunk_id"] not in existing_chunk_ids:
                        results.append(result)
                        existing_chunk_ids.add(result["chunk_id"])
            
            # Try graph traversal search if we still have fewer than top_k results
            if len(results) < top_k:
                # Use graph traversal to find related chunks through entity and concept connections
                traversal_query = """
                // Find chunks related to our current results
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Chunk)
                WHERE c.id IN $chunk_ids AND related.id NOT IN $chunk_ids
                
                WITH related, count(DISTINCT e) as shared_entities
                
                // Get document info
                MATCH (d:Document)-[:CONTAINS]->(related)
                """
                
                # Add tag filtering if requested
                if filter_tags and len(filter_tags) > 0:
                    traversal_query += """
                    MATCH (d)-[:HAS_TAG]->(t:Tag)
                    WHERE t.name IN $tags
                    """
                
                # Complete the query with scoring and results
                traversal_query += """
                // Calculate a score based on shared entities
                WITH related, d, shared_entities, related.relevance as relevance
                
                // Return results with document info
                RETURN related.id AS chunk_id,
                    related.text AS text,
                    d.id AS doc_id,
                    d.title AS title,
                    shared_entities + (relevance * 0.2) AS score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                # Only run traversal if we have some initial results to expand from
                if results:
                    chunk_ids = [r["chunk_id"] for r in results]
                    
                    # Run the traversal query
                    params = {
                        "chunk_ids": chunk_ids,
                        "limit": top_k - len(results)
                    }
                    
                    if filter_tags and len(filter_tags) > 0:
                        params["tags"] = filter_tags
                    
                    with self.driver.session(database=self.neo4j_db) as session:
                        result = session.run(traversal_query, **params)
                        traversal_results = [dict(record) for record in result]
                    
                    # Add traversal results to our result set
                    existing_chunk_ids = {r["chunk_id"] for r in results}
                    for result in traversal_results:
                        if result["chunk_id"] not in existing_chunk_ids:
                            results.append(result)
                            existing_chunk_ids.add(result["chunk_id"])
            
            # Sort final results by score and limit to top_k
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"Error in search: {e}")
            logger.exception(e)
            return []
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata by ID"""
        try:
            query = """
            MATCH (d:Document {id: $doc_id})
            RETURN d
            """
            
            with self.driver.session(database=self.neo4j_db) as session:
                result = session.run(query, doc_id=doc_id)
                record = result.single()
                
                if not record:
                    return {"error": f"Document not found: {doc_id}"}
                
                # Extract document properties
                doc = dict(record["d"])
                
                # Get tags
                tags_query = """
                MATCH (d:Document {id: $doc_id})-[:HAS_TAG]->(t:Tag)
                RETURN t.name AS tag
                """
                
                tags_result = session.run(tags_query, doc_id=doc_id)
                doc["tags"] = [record["tag"] for record in tags_result]
                
                return doc
                
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return {"error": str(e)}
    
    def list_documents(self, limit: int = 100, offset: int = 0, 
                filter_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """List all documents with pagination and filtering"""
        try:
            # Base query to get documents
            query = """
            MATCH (d:Document)
            """
            
            # Add tag filtering if requested
            if filter_tags and len(filter_tags) > 0:
                query += """
                MATCH (d)-[:HAS_TAG]->(t:Tag)
                WHERE t.name IN $tags
                """
            
            # Complete the query with sorting, pagination, and results
            query += """
            RETURN d
            ORDER BY d.date_added DESC
            SKIP $offset
            LIMIT $limit
            """
            
            # Count query to get total number of documents
            count_query = """
            MATCH (d:Document)
            """
            
            # Add tag filtering to count query if requested
            if filter_tags and len(filter_tags) > 0:
                count_query += """
                MATCH (d)-[:HAS_TAG]->(t:Tag)
                WHERE t.name IN $tags
                """
            
            # Complete the count query
            count_query += """
            RETURN count(d) AS total
            """
            
            with self.driver.session(database=self.neo4j_db) as session:
                # Get documents
                params = {"offset": offset, "limit": limit}
                if filter_tags and len(filter_tags) > 0:
                    params["tags"] = filter_tags
                
                result = session.run(query, **params)
                documents = []
                
                for record in result:
                    doc = dict(record["d"])
                    
                    # Get tags for each document
                    tags_query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_TAG]->(t:Tag)
                    RETURN t.name AS tag
                    """
                    
                    tags_result = session.run(tags_query, doc_id=doc["id"])
                    doc["tags"] = [record["tag"] for record in tags_result]
                    
                    documents.append(doc)
                
                # Get total count
                count_params = {}
                if filter_tags and len(filter_tags) > 0:
                    count_params["tags"] = filter_tags
                
                count_result = session.run(count_query, **count_params)
                total = count_result.single()["total"]
                
                return {
                    "documents": documents,
                    "total": total,
                    "offset": offset,
                    "limit": limit
                }
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return {
                "documents": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "error": str(e)
            }
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document and all its chunks"""
        try:
            # Get document metadata first to return in the response
            doc_data = self.get_document_by_id(doc_id)
            if "error" in doc_data:
                return doc_data
            
            # Delete all nodes and relationships related to this document
            query = """
            // Match the document
            MATCH (d:Document {id: $doc_id})
            
            // Match all chunks
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            
            // Match all key points
            OPTIONAL MATCH (c)-[:CONTAINS_POINT]->(k:KeyPoint)
            
            // Count chunks for the result
            WITH d, c, k, count(c) AS chunk_count
            
            // Delete key points
            DETACH DELETE k
            
            // Delete chunks
            WITH d, chunk_count
            MATCH (d)-[:CONTAINS]->(c:Chunk)
            DETACH DELETE c
            
            // Delete document
            WITH d, chunk_count
            DETACH DELETE d
            
            // Return the chunk count
            RETURN chunk_count
            """
            
            with self.driver.session(database=self.neo4j_db) as session:
                result = session.run(query, doc_id=doc_id)
                record = result.single()
                chunk_count = record["chunk_count"] if record else 0
                
                # Delete the document directory from disk
                doc_dir_path = os.path.join(DOCUMENTS_DIRECTORY, doc_id)
                if os.path.exists(doc_dir_path):
                    import shutil
                    shutil.rmtree(doc_dir_path)
                
                return {
                    "success": True,
                    "document": doc_data,
                    "chunks_deleted": chunk_count
                }
                
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return {"error": str(e)}

# Integration functions to use with the RAG API
def create_neo4j_rag_manager():
    """Create and return a Neo4j RAG Manager instance"""
    # Neo4j connection details
    neo4j_uri = "bolt://10.185.1.8:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Nutman17!"
    neo4j_db = "neo4j"
    
    # Create the manager
    return Neo4jRAGManager(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_db=neo4j_db
    )

# Function to replace RAGManager in the RAG API
def get_neo4j_rag_manager():
    """Get or create a Neo4j RAG Manager instance"""
    global neo4j_rag_manager
    if neo4j_rag_manager is None:
        neo4j_rag_manager = create_neo4j_rag_manager()
    return neo4j_rag_manager

# Initialize the global manager
neo4j_rag_manager = None

# Example usage
if __name__ == "__main__":
    # Set up basic logging for the test
    logging.basicConfig(level=logging.INFO)
    
    # Create a manager
    manager = create_neo4j_rag_manager()
    
    try:
        # Test search functionality
        results = manager.search("example query about machine learning")
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']}: {result['text'][:100]}...")
    finally:
        # Close the manager
        manager.close()

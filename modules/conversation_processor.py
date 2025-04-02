"""
Conversation Processing Module for Neo4j Integration.
Processes, chunks, and stores conversation history in Neo4j
while keeping it separate from document storage.
"""

import os
import json
import uuid
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import ollama
import tiktoken
import asyncio
from neo4j import GraphDatabase

# Configure logging
logger = logging.getLogger("conversation-processor")

# Configuration variables
CHUNK_SIZE = 1000  # tokens (same as document processing)
CHUNK_OVERLAP = 100  # tokens
EMBEDDING_MODEL = "bge-m3"  # Default embedding model
EMBEDDING_DIMENSION = 1024  # BGE-M3 embedding dimension
SEMANTIC_MODEL = "command-r7b"  # Model for semantic analysis
MAX_MESSAGES_BEFORE_PROCESSING = 40  # Process conversation after this many messages

# Initialize tokenizer for chunking
tokenizer_name = "gpt-3.5-turbo"  # Use OpenAI-compatible tokenizer for chunk sizing
try:
    tokenizer = tiktoken.encoding_for_model(tokenizer_name)
except:
    logger.warning(f"Could not load tokenizer {tokenizer_name}. Using fallback text chunking.")
    tokenizer = None

class ConversationProcessor:
    """Processes and stores conversations in Neo4j with semantic analysis"""
    
    def __init__(self, neo4j_manager=None):
        """
        Initialize with an existing Neo4j manager or create a new connection
        
        Args:
            neo4j_manager: Optional Neo4jRAGManager instance to reuse
        """
        self.neo4j_manager = neo4j_manager
        
        # If no manager provided, we need direct Neo4j access
        if not neo4j_manager:
            # Neo4j connection details (same as in neo4j_rag_integration.py)
            self.neo4j_uri = "bolt://10.185.1.8:7687"
            self.neo4j_user = "neo4j"
            self.neo4j_password = "Nutman17!"
            self.neo4j_db = "neo4j"
            
            # Create direct Neo4j driver
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Ensure constraints and indices
            self._ensure_constraints()
        else:
            # Use the manager's driver
            self.driver = neo4j_manager.driver
            self.neo4j_db = neo4j_manager.neo4j_db
        
        logger.info("Conversation Processor initialized")
    
    def _ensure_constraints(self):
        """Create necessary constraints and indices for conversation storage"""
        try:
            with self.driver.session(database=self.neo4j_db) as session:
                # Create constraints for Conversation nodes
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE"
                )
                session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (c:Conversation) ON (c.date)"
                )
                
                # Create constraints for ConversationChunk nodes
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ConversationChunk) REQUIRE c.id IS UNIQUE"
                )
                session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (c:ConversationChunk) ON (c.conversation_id)"
                )
                
                logger.info("Conversation constraints and indices created")
        except Exception as e:
            logger.error(f"Error creating constraints for conversations: {e}")
            logger.warning("Continuing without constraints - some features may not work correctly")
    
    def close(self):
        """Close the Neo4j connection if we created our own"""
        if not self.neo4j_manager and hasattr(self, 'driver'):
            self.driver.close()
    
    async def process_conversation(self, messages: List[Dict[str, str]], 
                               conversation_id: str = None, title: str = None) -> Dict[str, Any]:
        """
        Process a conversation and store it in Neo4j
        
        Args:
            messages: List of message objects with 'role' and 'content' fields
            conversation_id: Optional ID for the conversation
            title: Optional title for the conversation
            
        Returns:
            Dict with processing results and conversation_id
        """
        try:
            # Skip processing if there are too few messages
            if len(messages) < 3:  # Need at least a couple exchanges to be meaningful
                return {
                    "status": "skipped",
                    "reason": "Too few messages to process",
                    "count": len(messages)
                }
            
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Generate a title if not provided
            if not title:
                title = await self._generate_conversation_title(messages)
            
            # Format the messages into a single text for processing
            formatted_text = self._format_conversation(messages)
            
            # Split the conversation into chunks
            chunks = self._chunk_text(formatted_text)
            
            if not chunks:
                logger.warning(f"No chunks generated for conversation {conversation_id}")
                return {
                    "status": "error",
                    "reason": "Failed to generate chunks",
                    "conversation_id": conversation_id
                }
            
            # Create conversation node in Neo4j
            conversation_data = {
                "id": conversation_id,
                "title": title,
                "message_count": len(messages),
                "date": datetime.now().isoformat(),
                "chunk_count": len(chunks)
            }
            
            self._store_conversation(conversation_id, conversation_data)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"conv:{conversation_id}:{i}"
                await self._process_chunk(chunk, chunk_id, conversation_id, i)
            
            # Create relationships between chunks
            await self._create_conversation_chunk_relationships(conversation_id)
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "title": title,
                "chunk_count": len(chunks),
                "message_count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "reason": str(e),
                "conversation_id": conversation_id if conversation_id else "unknown"
            }
    
    async def _generate_conversation_title(self, messages: List[Dict[str, str]]) -> str:
        """Generate a title for the conversation using LLM"""
        try:
            # Get a sample of the conversation to generate a title
            sample_size = min(5, len(messages))
            conversation_sample = messages[:sample_size]
            
            # Format the sample
            sample_text = self._format_conversation(conversation_sample)
            
            # Create a prompt for title generation
            prompt = f"""
            Please generate a short, descriptive title (maximum 6 words) for this conversation:
            
            {sample_text[:500]}...
            
            Title:
            """
            
            # Using the ollama API to generate a title
            response = ollama.chat(
                model=SEMANTIC_MODEL,
                messages=[
                    {"role": "system", "content": "You generate concise, descriptive titles for conversations."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            title = response.get('message', {}).get('content', '').strip()
            
            # Clean up the title
            title = title.replace("Title:", "").replace("\"", "").strip()
            
            # If title is still too long, truncate it
            if len(title) > 50:
                title = title[:47] + "..."
            
            # If we couldn't generate a title, use a timestamp
            if not title:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            return title
            
        except Exception as e:
            logger.error(f"Error generating conversation title: {e}")
            # Fallback title
            return f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation into a single text string"""
        formatted_lines = []
        
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Skip system messages as they're not part of the conversation flow
            if role.lower() == 'system':
                continue
                
            # Format based on role
            if role.lower() == 'user':
                formatted_lines.append(f"User: {content}")
            elif role.lower() == 'assistant':
                formatted_lines.append(f"Assistant: {content}")
            else:
                formatted_lines.append(f"{role.capitalize()}: {content}")
        
        # Join with double newlines for clear separation
        return "\n\n".join(formatted_lines)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into segments with specified token size and overlap"""
        if not text:
            return []
            
        # Use tiktoken if available
        if tokenizer:
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
        else:
            # Fallback to paragraph-based chunking
            paragraphs = text.split('\n\n')
            
            # For very short conversations, just return the whole text as one chunk
            if len(paragraphs) <= 3:
                return [text]
                
            # For longer conversations, create chunks of related paragraphs
            chunks = []
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para_length = len(para)
                
                # If adding this paragraph would exceed chunk size, save current chunk
                if current_length + para_length > CHUNK_SIZE and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some overlap by retaining the last paragraph
                    current_chunk = current_chunk[-1:] if current_chunk else []
                    current_length = len(current_chunk[0]) if current_chunk else 0
                
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_length += para_length
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            return chunks
    
    def _store_conversation(self, conversation_id: str, data: Dict[str, Any]):
        """Store conversation metadata in Neo4j"""
        query = """
        CREATE (c:Conversation {
            id: $id,
            title: $title,
            message_count: $message_count,
            date: $date,
            chunk_count: $chunk_count
        })
        """
        
        with self.driver.session(database=self.neo4j_db) as session:
            session.run(
                query,
                id=data["id"],
                title=data["title"],
                message_count=data["message_count"],
                date=data["date"],
                chunk_count=data["chunk_count"]
            )
    
    async def _process_chunk(self, chunk_text: str, chunk_id: str, 
                         conversation_id: str, chunk_index: int):
        """Process a conversation chunk and store in Neo4j"""
        try:
            # Generate embedding
            embedding_result = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=chunk_text
            )
            
            embedding = embedding_result.get('embedding', [])
            
            if not embedding:
                logger.error(f"Failed to generate embedding for chunk {chunk_index} of conversation {conversation_id}")
                return
            
            # Extract semantic information
            semantic_info = await self._extract_semantic_info(chunk_text)
            
            # Store chunk in Neo4j
            self._store_conversation_chunk(
                chunk_id, 
                conversation_id, 
                chunk_index, 
                chunk_text, 
                embedding, 
                semantic_info
            )
            
            logger.info(f"Processed chunk {chunk_index} of conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error processing conversation chunk {chunk_index}: {e}")
    
    async def _extract_semantic_info(self, text: str) -> Dict[str, Any]:
        """Extract semantic information from text"""
        try:
            # Create a prompt for the model
            prompt = f"""
            Analyze the following conversation excerpt and extract semantic information in JSON format.
            Include these categories:
            - topics: List of main topics discussed
            - entities: List of named entities (people, places, organizations, etc.)
            - concepts: List of abstract concepts and themes
            - key_points: List of main ideas or facts
            - sentiment: Overall sentiment (positive, negative, or neutral)
            - relevance: Rate the relevance and importance of this excerpt (1-10)

            Format the output as a valid JSON object with these fields.

            Text to analyze:
            {text[:1000]}  # Limit text length to avoid token limits
            """
            
            # Use ollama for semantic analysis
            response = ollama.chat(
                model=SEMANTIC_MODEL,
                messages=[
                    {"role": "system", "content": "You are a semantic analysis assistant that extracts structured information from conversation texts."},
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
                
                # Ensure all required fields exist
                data.setdefault('topics', [])
                data.setdefault('entities', [])
                data.setdefault('concepts', [])
                data.setdefault('key_points', [])
                data.setdefault('sentiment', 'neutral')
                data.setdefault('relevance', 5)
                
                return data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {content}")
                return {
                    'topics': [],
                    'entities': [],
                    'concepts': [],
                    'key_points': [],
                    'sentiment': 'neutral',
                    'relevance': 5
                }
                
        except Exception as e:
            logger.error(f"Error extracting semantic info: {e}")
            return {
                'topics': [],
                'entities': [],
                'concepts': [],
                'key_points': [],
                'sentiment': 'neutral',
                'relevance': 5
            }
    
    def _store_conversation_chunk(self, chunk_id: str, conversation_id: str, 
                              chunk_index: int, chunk_text: str, 
                              embedding: List[float], semantic_info: Dict[str, Any]):
        """Store a conversation chunk in Neo4j"""
        try:
            # Use a sample of the embedding for efficient storage
            embedding_sample = embedding[:10]
            
            # Base query to create chunk
            query = """
            MATCH (c:Conversation {id: $conversation_id})
            CREATE (chunk:ConversationChunk {
                id: $chunk_id,
                conversation_id: $conversation_id,
                chunk_index: $chunk_index,
                text: $text,
                sentiment: $sentiment,
                relevance: $relevance,
                embedding_sample: $embedding_sample,
                embedding_dimensions: $embedding_dimensions
            })
            CREATE (c)-[:CONTAINS]->(chunk)
            RETURN chunk
            """
            
            # Parameters for the query
            params = {
                "chunk_id": chunk_id,
                "conversation_id": conversation_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "sentiment": semantic_info.get('sentiment', 'neutral'),
                "relevance": semantic_info.get('relevance', 5),
                "embedding_sample": embedding_sample,
                "embedding_dimensions": len(embedding)
            }
            
            # Execute the query
            with self.driver.session(database=self.neo4j_db) as session:
                # Create chunk
                result = session.run(query, **params)
                chunk_node = result.single()
                
                # Store entities
                for entity in semantic_info.get('entities', []):
                    if entity and isinstance(entity, str):
                        try:
                            session.run("""
                            MATCH (chunk:ConversationChunk {id: $chunk_id})
                            MERGE (e:Entity {name: $entity_name})
                            CREATE (chunk)-[:MENTIONS]->(e)
                            """, chunk_id=chunk_id, entity_name=entity)
                        except Exception as e:
                            logger.warning(f"Error creating entity {entity}: {e}")
                
                # Store concepts
                for concept in semantic_info.get('concepts', []):
                    if concept and isinstance(concept, str):
                        try:
                            session.run("""
                            MATCH (chunk:ConversationChunk {id: $chunk_id})
                            MERGE (c:Concept {name: $concept_name})
                            CREATE (chunk)-[:RELATES_TO]->(c)
                            """, chunk_id=chunk_id, concept_name=concept)
                        except Exception as e:
                            logger.warning(f"Error creating concept {concept}: {e}")
                
                # Store topics (specific to conversations)
                for topic in semantic_info.get('topics', []):
                    if topic and isinstance(topic, str):
                        try:
                            session.run("""
                            MATCH (chunk:ConversationChunk {id: $chunk_id})
                            MERGE (t:Topic {name: $topic_name})
                            CREATE (chunk)-[:DISCUSSES]->(t)
                            """, chunk_id=chunk_id, topic_name=topic)
                        except Exception as e:
                            logger.warning(f"Error creating topic {topic}: {e}")
                
                # Store key points
                for key_point in semantic_info.get('key_points', []):
                    if key_point and isinstance(key_point, str):
                        try:
                            session.run("""
                            MATCH (chunk:ConversationChunk {id: $chunk_id})
                            CREATE (k:KeyPoint {
                                text: $key_point_text,
                                source_id: $chunk_id,
                                type: 'conversation'
                            })
                            CREATE (chunk)-[:CONTAINS_POINT]->(k)
                            """, chunk_id=chunk_id, key_point_text=key_point)
                        except Exception as e:
                            logger.warning(f"Error creating key point: {e}")
        
        except Exception as e:
            logger.error(f"Error storing conversation chunk: {e}")
            raise
    
    async def _create_conversation_chunk_relationships(self, conversation_id: str):
        """Create relationships between chunks in the conversation"""
        try:
            logger.info(f"Creating relationships between chunks for conversation {conversation_id}")
            
            # Get all chunks for this conversation
            query = """
            MATCH (c:Conversation {id: $conversation_id})-[:CONTAINS]->(chunk:ConversationChunk)
            RETURN chunk.id as id, chunk.chunk_index as chunk_index
            ORDER BY chunk.chunk_index
            """
            
            chunks = []
            with self.driver.session(database=self.neo4j_db) as session:
                result = session.run(query, conversation_id=conversation_id)
                chunks = [dict(record) for record in result]
            
            # Connect sequential chunks automatically
            if len(chunks) >= 2:
                sequence_query = """
                MATCH (c1:ConversationChunk {id: $chunk1_id})
                MATCH (c2:ConversationChunk {id: $chunk2_id})
                CREATE (c1)-[:NEXT]->(c2)
                """
                
                with self.driver.session(database=self.neo4j_db) as session:
                    for i in range(len(chunks) - 1):
                        chunk1_id = chunks[i]['id']
                        chunk2_id = chunks[i + 1]['id']
                        session.run(sequence_query, chunk1_id=chunk1_id, chunk2_id=chunk2_id)
                
                logger.info(f"Created sequential relationships for {len(chunks)-1} chunk pairs")
            
            # If there are enough chunks, also create relationships based on shared entities/concepts
            if len(chunks) >= 3:
                relationship_query = """
                MATCH (c1:ConversationChunk {id: $chunk1_id})
                MATCH (c2:ConversationChunk {id: $chunk2_id})
                WHERE c1 <> c2 AND NOT (c1)-[:NEXT]->(c2) AND NOT (c2)-[:NEXT]->(c1)
                
                // Count shared entities
                OPTIONAL MATCH (c1)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2)
                WITH c1, c2, COUNT(e) as shared_entities
                
                // Count shared concepts
                OPTIONAL MATCH (c1)-[:RELATES_TO]->(co:Concept)<-[:RELATES_TO]-(c2)
                WITH c1, c2, shared_entities, COUNT(co) as shared_concepts
                
                // Only create relationship if they share something
                WHERE shared_entities > 0 OR shared_concepts > 0
                
                CREATE (c1)-[:RELATED_TO {
                    strength: (shared_entities + shared_concepts) / 5.0,
                    shared_entities: shared_entities,
                    shared_concepts: shared_concepts
                }]->(c2)
                """
                
                with self.driver.session(database=self.neo4j_db) as session:
                    for i in range(len(chunks)):
                        for j in range(i + 2, len(chunks)):  # Skip adjacent chunks
                            chunk1_id = chunks[i]['id']
                            chunk2_id = chunks[j]['id']
                            session.run(relationship_query, chunk1_id=chunk1_id, chunk2_id=chunk2_id)
                
                logger.info(f"Created semantic relationships for conversation chunks")
        
        except Exception as e:
            logger.error(f"Error creating chunk relationships: {e}")
    
    async def search_conversations(self, query: str, limit: int = 5, 
                             filter_date_start: str = None, 
                             filter_date_end: str = None) -> List[Dict[str, Any]]:
        """
        Search for relevant conversation chunks based on a query
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            filter_date_start: Optional start date filter (ISO format)
            filter_date_end: Optional end date filter (ISO format)
            
        Returns:
            List of matching conversation chunks with metadata
        """
        try:
            # Generate embedding for the query
            embedding_result = ollama.embeddings(
                model=EMBEDDING_MODEL,
                prompt=query
            )
            embedding = embedding_result.get('embedding', [])
            
            # Extract entities and concepts from the query for semantic search
            semantic_info = await self._extract_semantic_info(query)
            entities = semantic_info.get('entities', [])
            concepts = semantic_info.get('concepts', [])
            topics = semantic_info.get('topics', [])
            
            # Get relevant conversation chunks
            chunks = []
            
            # First try semantic search if we have entities or concepts
            if embedding and (entities or concepts or topics):
                with self.driver.session(database=self.neo4j_db) as session:
                    # Build a query that combines multiple relevance factors
                    search_query = """
                    // Start with all conversation chunks
                    MATCH (c:ConversationChunk)
                    
                    // Get parent conversation
                    MATCH (conv:Conversation)-[:CONTAINS]->(c)
                    """
                    
                    # Add date filtering if specified
                    if filter_date_start or filter_date_end:
                        search_query += """
                        WHERE """
                        
                        if filter_date_start:
                            search_query += "conv.date >= $date_start"
                            
                        if filter_date_start and filter_date_end:
                            search_query += " AND "
                            
                        if filter_date_end:
                            search_query += "conv.date <= $date_end"
                    
                    # Continue with semantic matches
                    search_query += """
                    // Match entities
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $entities
                    
                    WITH c, conv, COUNT(DISTINCT e) as entity_matches
                    
                    // Match concepts
                    OPTIONAL MATCH (c)-[:RELATES_TO]->(co:Concept)
                    WHERE co.name IN $concepts
                    
                    WITH c, conv, entity_matches, COUNT(DISTINCT co) as concept_matches
                    
                    // Match topics
                    OPTIONAL MATCH (c)-[:DISCUSSES]->(t:Topic)
                    WHERE t.name IN $topics
                    
                    WITH c, conv, entity_matches, concept_matches, COUNT(DISTINCT t) as topic_matches
                    
                    // Text matching
                    WHERE c.text CONTAINS $query_text OR 
                          entity_matches > 0 OR 
                          concept_matches > 0 OR
                          topic_matches > 0
                    
                    // Calculate a relevance score
                    WITH c, conv,
                         (entity_matches * 3.0) + 
                         (concept_matches * 2.0) + 
                         (topic_matches * 4.0) +
                         (CASE WHEN c.text CONTAINS $query_text THEN 5.0 ELSE 0.0 END) +
                         c.relevance AS relevance_score
                    
                    // Return results
                    RETURN c.id AS chunk_id,
                           c.text AS text,
                           c.chunk_index AS chunk_index,
                           conv.id AS conversation_id,
                           conv.title AS title,
                           conv.date AS date,
                           relevance_score AS score
                    ORDER BY relevance_score DESC
                    LIMIT $limit
                    """
                    
                    # Parameters for the query
                    params = {
                        "query_text": query,
                        "entities": entities,
                        "concepts": concepts,
                        "topics": topics,
                        "limit": limit
                    }
                    
                    # Add date parameters if specified
                    if filter_date_start:
                        params["date_start"] = filter_date_start
                    if filter_date_end:
                        params["date_end"] = filter_date_end
                    
                    # Execute the query
                    result = session.run(search_query, **params)
                    chunks = [dict(record) for record in result]
            
            # If no results from semantic search or not enough entities/concepts,
            # fall back to text search
            if not chunks:
                with self.driver.session(database=self.neo4j_db) as session:
                    text_query = """
                    MATCH (c:ConversationChunk)
                    WHERE c.text CONTAINS $query_text
                    
                    // Get parent conversation
                    MATCH (conv:Conversation)-[:CONTAINS]->(c)
                    """
                    
                    # Add date filtering if specified
                    if filter_date_start or filter_date_end:
                        text_query += """
                        AND """
                        
                        if filter_date_start:
                            text_query += "conv.date >= $date_start"
                            
                        if filter_date_start and filter_date_end:
                            text_query += " AND "
                            
                        if filter_date_end:
                            text_query += "conv.date <= $date_end"
                    
                    # Complete the query
                    text_query += """
                    RETURN c.id AS chunk_id,
                           c.text AS text,
                           c.chunk_index AS chunk_index,
                           conv.id AS conversation_id,
                           conv.title AS title,
                           conv.date AS date,
                           c.relevance AS score
                    ORDER BY c.relevance DESC
                    LIMIT $limit
                    """
                    
                    # Parameters for the query
                    params = {
                        "query_text": query,
                        "limit": limit
                    }
                    
                    # Add date parameters if specified
                    if filter_date_start:
                        params["date_start"] = filter_date_start
                    if filter_date_end:
                        params["date_end"] = filter_date_end
                    
                    # Execute the query
                    result = session.run(text_query, **params)
                    chunks = [dict(record) for record in result]
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    def get_conversation_by_id(self, conversation_id: str) -> Dict[str, Any]:
        """Get a conversation by ID with all its chunks"""
        try:
            with self.driver.session(database=self.neo4j_db) as session:
                # Get conversation metadata
                query = """
                MATCH (c:Conversation {id: $conversation_id})
                RETURN c
                """
                
                result = session.run(query, conversation_id=conversation_id)
                record = result.single()
                
                if not record:
                    return {"error": f"Conversation not found: {conversation_id}"}
                
                conversation_data = dict(record["c"])
                
                # Get all chunks for this conversation
                chunks_query = """
                MATCH (c:Conversation {id: $conversation_id})-[:CONTAINS]->(chunk:ConversationChunk)
                RETURN chunk.id AS id, 
                       chunk.text AS text,
                       chunk.chunk_index AS chunk_index,
                       chunk.sentiment AS sentiment,
                       chunk.relevance AS relevance
                ORDER BY chunk.chunk_index
                """
                
                chunks_result = session.run(chunks_query, conversation_id=conversation_id)
                chunks = [dict(record) for record in chunks_result]
                
                # Get semantic elements for this conversation
                topics_query = """
                MATCH (c:Conversation {id: $conversation_id})-[:CONTAINS]->(chunk:ConversationChunk)
                MATCH (chunk)-[:DISCUSSES]->(t:Topic)
                RETURN DISTINCT t.name AS topic
                """
                
                topics_result = session.run(topics_query, conversation_id=conversation_id)
                topics = [record["topic"] for record in topics_result]
                
                entities_query = """
                MATCH (c:Conversation {id: $conversation_id})-[:CONTAINS]->(chunk:ConversationChunk)
                MATCH (chunk)-[:MENTIONS]->(e:Entity)
                RETURN DISTINCT e.name AS entity
                """
                
                entities_result = session.run(entities_query, conversation_id=conversation_id)
                entities = [record["entity"] for record in entities_result]
                
                # Combine all data
                conversation_data["chunks"] = chunks
                conversation_data["topics"] = topics
                conversation_data["entities"] = entities
                
                return conversation_data
                
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return {"error": str(e)}
    
    def list_conversations(self, limit: int = 10, offset: int = 0,
                       filter_date_start: str = None,
                       filter_date_end: str = None) -> Dict[str, Any]:
        """List conversations with pagination and filtering"""
        try:
            with self.driver.session(database=self.neo4j_db) as session:
                # Base query
                query = """
                MATCH (c:Conversation)
                """
                
                # Add date filtering if specified
                if filter_date_start or filter_date_end:
                    query += "WHERE "
                    
                    if filter_date_start:
                        query += "c.date >= $date_start"
                        
                    if filter_date_start and filter_date_end:
                        query += " AND "
                        
                    if filter_date_end:
                        query += "c.date <= $date_end"
                
                # Add ordering, pagination and return
                query += """
                RETURN c
                ORDER BY c.date DESC
                SKIP $offset
                LIMIT $limit
                """
                
                # Parameters
                params = {
                    "offset": offset,
                    "limit": limit
                }
                
                # Add date parameters if specified
                if filter_date_start:
                    params["date_start"] = filter_date_start
                if filter_date_end:
                    params["date_end"] = filter_date_end
                
                # Execute the query
                result = session.run(query, **params)
                conversations = [dict(record["c"]) for record in result]
                
                # Count query for total
                count_query = """
                MATCH (c:Conversation)
                """
                
                # Add the same date filtering to count query
                if filter_date_start or filter_date_end:
                    count_query += "WHERE "
                    
                    if filter_date_start:
                        count_query += "c.date >= $date_start"
                        
                    if filter_date_start and filter_date_end:
                        count_query += " AND "
                        
                    if filter_date_end:
                        count_query += "c.date <= $date_end"
                
                count_query += """
                RETURN COUNT(c) AS total
                """
                
                # Execute count query
                count_result = session.run(count_query, **params)
                total = count_result.single()["total"]
                
                return {
                    "conversations": conversations,
                    "total": total,
                    "offset": offset,
                    "limit": limit
                }
                
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return {
                "conversations": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "error": str(e)
            }
    
    def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Delete a conversation and all its chunks"""
        try:
            # Get conversation metadata first for the response
            conversation_data = self.get_conversation_by_id(conversation_id)
            if "error" in conversation_data:
                return conversation_data
            
            with self.driver.session(database=self.neo4j_db) as session:
                # Delete all nodes and relationships related to this conversation
                query = """
                // Match conversation
                MATCH (c:Conversation {id: $conversation_id})
                
                // Match all chunks
                OPTIONAL MATCH (c)-[:CONTAINS]->(chunk:ConversationChunk)
                
                // Match all key points
                OPTIONAL MATCH (chunk)-[:CONTAINS_POINT]->(k:KeyPoint)
                
                // Count chunks for the response
                WITH c, chunk, k, count(chunk) AS chunk_count
                
                // Delete key points
                DETACH DELETE k
                
                // Delete chunks
                WITH c, chunk_count
                MATCH (c)-[:CONTAINS]->(chunk:ConversationChunk)
                DETACH DELETE chunk
                
                // Delete conversation
                WITH c, chunk_count
                DETACH DELETE c
                
                // Return the chunk count
                RETURN chunk_count
                """
                
                result = session.run(query, conversation_id=conversation_id)
                record = result.single()
                chunk_count = record["chunk_count"] if record else 0
                
                return {
                    "success": True,
                    "conversation": conversation_data,
                    "chunks_deleted": chunk_count
                }
                
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return {"error": str(e)}


class ConversationMonitor:
    """
    Monitors conversation and processes it when it reaches a certain length
    """
    
    def __init__(self, processor=None, max_messages=MAX_MESSAGES_BEFORE_PROCESSING):
        """
        Initialize the conversation monitor
        
        Args:
            processor: ConversationProcessor instance (created if not provided)
            max_messages: Maximum messages before processing
        """
        self.processor = processor or ConversationProcessor()
        self.max_messages = max_messages
        self.message_counts = {}  # conversation_id -> count
        self.last_processed = {}  # conversation_id -> timestamp
        
        logger.info(f"Conversation Monitor initialized (threshold: {max_messages} messages)")
    
    def close(self):
        """Close the processor if we created it"""
        if hasattr(self, 'processor') and self.processor:
            self.processor.close()
    
    def add_message(self, message, conversation_id="primary_conversation"):
        """
        Add a message to the conversation and process if needed
        
        Returns True if processing was triggered
        """
        # Update message count
        if conversation_id not in self.message_counts:
            self.message_counts[conversation_id] = 0
        
        self.message_counts[conversation_id] += 1
        
        # Check if we should process
        should_process = (
            self.message_counts[conversation_id] >= self.max_messages and
            (conversation_id not in self.last_processed or 
             (datetime.now() - self.last_processed[conversation_id]).total_seconds() > 3600)
        )
        
        return should_process
    
    async def process_if_needed(self, messages, conversation_id="primary_conversation", force=False):
        """
        Check if processing is needed and process the conversation
        
        Args:
            messages: List of message objects with 'role' and 'content'
            conversation_id: ID for the conversation
            force: Force processing regardless of message count
            
        Returns:
            Dict with processing result or None if not processed
        """
        # Check if we should process
        should_process = force or self.add_message(messages[-1], conversation_id)
        
        if should_process:
            logger.info(f"Processing conversation {conversation_id} with {len(messages)} messages")
            
            # Process the conversation
            result = await self.processor.process_conversation(messages, conversation_id)
            
            # Reset count and update last processed time
            self.message_counts[conversation_id] = 0
            self.last_processed[conversation_id] = datetime.now()
            
            return result
        
        return None
    
    def reset_count(self, conversation_id="primary_conversation"):
        """Reset the message count for a conversation"""
        self.message_counts[conversation_id] = 0


# Function to get a singleton instance of the conversation monitor
_conversation_monitor_instance = None

def get_conversation_monitor(max_messages=MAX_MESSAGES_BEFORE_PROCESSING):
    """Get or create a singleton instance of the conversation monitor"""
    global _conversation_monitor_instance
    
    if _conversation_monitor_instance is None:
        # Try to get the Neo4j RAG manager if it exists
        neo4j_manager = None
        try:
            from modules.neo4j_rag_integration import get_neo4j_rag_manager
            neo4j_manager = get_neo4j_rag_manager()
            logger.info("Using existing Neo4j RAG manager")
        except Exception as e:
            logger.warning(f"Could not get Neo4j RAG manager: {e}")
        
        # Create the processor with the Neo4j manager
        processor = ConversationProcessor(neo4j_manager)
        
        # Create and return the monitor
        _conversation_monitor_instance = ConversationMonitor(
            processor=processor,
            max_messages=max_messages
        )
        
        logger.info(f"Initialized conversation monitor singleton (threshold: {max_messages})")
    
    return _conversation_monitor_instance


# Function to integrate conversation processing with the server
def integrate_conversation_processor_with_server(app):
    """
    Integrate the conversation processor with the server
    
    This hooks into the main message handling to monitor and process conversations
    """
    from fastapi import APIRouter
    
    # Create a router for conversation endpoints
    router = APIRouter()
    
    # Get the conversation monitor
    monitor = get_conversation_monitor()
    
    # Add endpoint to process a conversation
    @router.post("/process")
    async def process_conversation(data: dict):
        """Force processing of a conversation"""
        messages = data.get("messages", [])
        conversation_id = data.get("conversation_id", "primary_conversation")
        
        if not messages:
            return {"error": "No messages provided"}
        
        result = await monitor.processor.process_conversation(messages, conversation_id)
        return result or {"status": "no_processing_needed"}
    
    # Add endpoint to list processed conversations
    @router.get("/list")
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
    @router.get("/{conversation_id}")
    async def get_conversation(conversation_id: str):
        """Get a specific conversation by ID"""
        return monitor.processor.get_conversation_by_id(conversation_id)
    
    # Add endpoint to search conversations
    @router.post("/search")
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
    @router.delete("/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        """Delete a conversation"""
        return monitor.processor.delete_conversation(conversation_id)
    
    # Include the router
    app.include_router(router, prefix="/v1/conversations", tags=["conversations"])
    
    logger.info("Conversation processor integrated with server")
    
    return monitor
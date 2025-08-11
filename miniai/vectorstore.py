"""
Pluggable vector store implementations
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import faiss
import pickle
import os
from pathlib import Path
import aiosqlite
from sentence_transformers import SentenceTransformer
from .models import MemoryEntry, RetrievalResult
from .observability import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_entries(self, entries: List[MemoryEntry]) -> None:
        """Add entries to the vector store"""
        pass
    
    @abstractmethod
    async def query(self, query_text: str, k: int = 5, 
                   filter_metadata: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Query the vector store"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the vector store"""
        pass


class FaissVectorStore(VectorStore):
    """FAISS-based vector store with SQLite metadata"""
    
    def __init__(self, config, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.config = config
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index = None
        self.metadata_db = None
        self.dimension = config.vectorstore.dimension
        self.index_path = config.vectorstore.index_path
        self.db_path = config.memory.long_term_db_path
        self._initialized = False
    
    async def _initialize(self):
        """Initialize the vector store"""
        if self._initialized:
            return
            
        # Create directories
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Initialize metadata database
        self.metadata_db = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        
        self._initialized = True
        logger.info(f"Initialized FaissVectorStore with dimension {self.dimension}")
    
    async def _create_tables(self):
        """Create SQLite tables for metadata and FTS"""
        await self.metadata_db.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP,
                vector_index INTEGER
            )
        """)
        
        # Full-text search table
        await self.metadata_db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                id, text, content='entries', content_rowid='rowid'
            )
        """)
        
        await self.metadata_db.commit()
    
    async def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if not self.embedding_model:
            await self._initialize()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            self.embedding_model.encode, 
            texts,
            {"show_progress_bar": False}
        )
        return embeddings
    
    async def add_entries(self, entries: List[MemoryEntry]) -> None:
        """Add entries to FAISS and metadata DB"""
        if not self._initialized:
            await self._initialize()
        
        if not entries:
            return
        
        texts = [entry.text for entry in entries]
        embeddings = await self._embed_texts(texts)
        
        # Add to FAISS index
        start_index = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))
        
        # Add to metadata database
        for i, entry in enumerate(entries):
            await self.metadata_db.execute(
                "INSERT OR REPLACE INTO entries (id, text, metadata, created_at, vector_index) VALUES (?, ?, ?, ?, ?)",
                (entry.id, entry.text, pickle.dumps(entry.metadata), 
                 entry.created_at, start_index + i)
            )
            
            # Add to FTS
            await self.metadata_db.execute(
                "INSERT OR REPLACE INTO entries_fts (id, text) VALUES (?, ?)",
                (entry.id, entry.text)
            )
        
        await self.metadata_db.commit()
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        logger.debug(f"Added {len(entries)} entries to vector store")
    
    async def query(self, query_text: str, k: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Hybrid retrieval: semantic + keyword search"""
        if not self._initialized:
            await self._initialize()
        
        import time
        start_time = time.time()
        
        # Semantic search
        query_embedding = await self._embed_texts([query_text])
        semantic_scores, semantic_indices = self.index.search(
            query_embedding.astype(np.float32), min(k * 2, self.index.ntotal)
        )
        
        # Keyword search using FTS
        fts_results = await self.metadata_db.execute(
            "SELECT id, rank FROM entries_fts WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
            (query_text, k * 2)
        )
        fts_ids = {row[0]: row[1] for row in await fts_results.fetchall()}
        
        # Combine results
        combined_results = {}
        alpha = self.config.memory.hybrid_alpha
        
        # Add semantic results
        for i, (score, idx) in enumerate(zip(semantic_scores[0], semantic_indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            result = await self.metadata_db.execute(
                "SELECT id, text, metadata, created_at FROM entries WHERE vector_index = ?",
                (int(idx),)
            )
            row = await result.fetchone()
            if row:
                entry_id, text, metadata_blob, created_at = row
                semantic_score = float(score)
                
                if entry_id in combined_results:
                    combined_results[entry_id]['score'] += alpha * semantic_score
                else:
                    combined_results[entry_id] = {
                        'entry': MemoryEntry(
                            id=entry_id,
                            text=text,
                            metadata=pickle.loads(metadata_blob) if metadata_blob else {},
                            created_at=created_at
                        ),
                        'score': alpha * semantic_score
                    }
        
        # Add keyword results
        for entry_id, fts_rank in fts_ids.items():
            keyword_score = 1.0 / (1.0 + fts_rank)  # Convert rank to score
            
            if entry_id in combined_results:
                combined_results[entry_id]['score'] += (1 - alpha) * keyword_score
            else:
                result = await self.metadata_db.execute(
                    "SELECT text, metadata, created_at FROM entries WHERE id = ?",
                    (entry_id,)
                )
                row = await result.fetchone()
                if row:
                    text, metadata_blob, created_at = row
                    combined_results[entry_id] = {
                        'entry': MemoryEntry(
                            id=entry_id,
                            text=text,
                            metadata=pickle.loads(metadata_blob) if metadata_blob else {},
                            created_at=created_at
                        ),
                        'score': (1 - alpha) * keyword_score
                    }
        
        # Sort and limit results
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        entries = [r['entry'] for r in sorted_results]
        scores = [r['score'] for r in sorted_results]
        
        query_time = time.time() - start_time
        
        return RetrievalResult(
            entries=entries,
            scores=scores,
            total_found=len(entries),
            query_time=query_time
        )
    
    async def close(self):
        """Close the vector store"""
        if self.metadata_db:
            await self.metadata_db.close()


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing"""
    
    def __init__(self, config):
        self.config = config
        self.entries: Dict[str, MemoryEntry] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.embedding_model = None
    
    async def _initialize(self):
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    async def add_entries(self, entries: List[MemoryEntry]) -> None:
        await self._initialize()
        
        for entry in entries:
            self.entries[entry.id] = entry
            # Generate embedding
            embedding = self.embedding_model.encode([entry.text])[0]
            self.embeddings[entry.id] = embedding.tolist()
    
    async def query(self, query_text: str, k: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        await self._initialize()
        
        if not self.entries:
            return RetrievalResult()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Calculate similarities
        similarities = []
        for entry_id, entry in self.entries.items():
            embedding = np.array(self.embeddings[entry_id])
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((similarity, entry_id))
        
        # Sort and take top k
        similarities.sort(reverse=True)
        top_k = similarities[:k]
        
        entries = [self.entries[entry_id] for _, entry_id in top_k]
        scores = [score for score, _ in top_k]
        
        return RetrievalResult(
            entries=entries,
            scores=scores,
            total_found=len(entries),
            query_time=0.001
        )
    
    async def close(self):
        pass
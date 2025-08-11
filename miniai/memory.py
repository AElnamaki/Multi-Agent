"""
Memory management for MiniAI - short-term and long-term memory
"""

import asyncio
from collections import deque
from typing import List, Dict, Any, Optional
from .models import MemoryEntry, ExecutionResult, RetrievalResult, RetrievalQuery
from .vectorstore import VectorStore, FaissVectorStore, InMemoryVectorStore
from .config import Config
from .observability import get_logger, trace_operation

logger = get_logger(__name__)


class ShortTermMemory:
    """Rolling buffer for recent execution results"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def add_result(self, result: ExecutionResult) -> None:
        """Add execution result to short-term memory"""
        entry = MemoryEntry(
            text=f"Task: {result.task_name}\nResult: {result.result_text}",
            metadata={
                "task_id": result.task_id,
                "task_name": result.task_name,
                "status": result.status,
                "execution_time": result.execution_time,
                "memory_type": "short_term"
            }
        )
        self.buffer.append(entry)
        logger.debug(f"Added result to short-term memory: {result.task_name}")
    
    def get_recent(self, k: int = 5) -> List[MemoryEntry]:
        """Get k most recent entries"""
        recent = list(self.buffer)[-k:]
        return recent
    
    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """Simple keyword search in short-term memory"""
        query_lower = query.lower()
        matches = []
        
        for entry in reversed(self.buffer):  # Most recent first
            if query_lower in entry.text.lower():
                matches.append(entry)
                if len(matches) >= k:
                    break
        
        return matches
    
    def clear(self) -> None:
        """Clear short-term memory"""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current size of buffer"""
        return len(self.buffer)


class LongTermMemory:
    """Persistent memory using vector store"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store: Optional[VectorStore] = None
    
    async def initialize(self) -> None:
        """Initialize the vector store"""
        if self.config.vectorstore.provider == "faiss":
            self.vector_store = FaissVectorStore(self.config)
        else:
            self.vector_store = InMemoryVectorStore(self.config)
        
        logger.info(f"Initialized long-term memory with {self.config.vectorstore.provider}")
    
    async def add_result(self, result: ExecutionResult) -> None:
        """Add execution result to long-term memory"""
        if not self.vector_store:
            await self.initialize()
        
        # Create memory entry
        entry = MemoryEntry(
            text=f"Task: {result.task_name}\nResult: {result.result_text}",
            metadata={
                "task_id": result.task_id,
                "task_name": result.task_name,
                "status": result.status,
                "execution_time": result.execution_time,
                "tokens_used": result.tokens_used,
                "memory_type": "long_term",
                "structured_facts": result.structured_facts,
                "actionable_steps": result.actionable_steps
            }
        )
        
        with trace_operation("long_term_memory.add"):
            await self.vector_store.add_entries([entry])
        
        logger.info(f"Added result to long-term memory: {result.task_name}")
    
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """Search long-term memory"""
        if not self.vector_store:
            await self.initialize()
        
        with trace_operation("long_term_memory.search"):
            result = await self.vector_store.query(
                query_text=query.query_text,
                k=query.k,
                filter_metadata=query.filter_metadata
            )
        
        logger.debug(f"Retrieved {len(result.entries)} entries for query: {query.query_text}")
        return result
    
    async def close(self) -> None:
        """Close long-term memory"""
        if self.vector_store:
            await self.vector_store.close()


class HybridMemoryManager:
    """Manages both short-term and long-term memory"""
    
    def __init__(self, config: Config):
        self.config = config
        self.short_term = ShortTermMemory(config.memory.short_term_size)
        self.long_term = LongTermMemory(config)
    
    async def initialize(self) -> None:
        """Initialize memory systems"""
        await self.long_term.initialize()
        logger.info("Hybrid memory manager initialized")
    
    async def store_result(self, result: ExecutionResult) -> None:
        """Store result in both short-term and long-term memory"""
        # Always store in short-term memory
        self.short_term.add_result(result)
        
        # Store in long-term memory if successful
        if result.status in ["success", "partial"]:
            await self.long_term.add_result(result)
    
    async def retrieve_context(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """Retrieve context from both memory systems"""
        context_entries = []
        
        # Get from short-term memory (more recent, higher priority)
        short_term_k = min(k // 2, self.short_term.size())
        if short_term_k > 0:
            recent_entries = self.short_term.search(query, short_term_k)
            context_entries.extend(recent_entries)
        
        # Get remaining from long-term memory
        remaining_k = k - len(context_entries)
        if remaining_k > 0:
            query_obj = RetrievalQuery(
                query_text=query,
                k=remaining_k,
                similarity_threshold=0.1  # Lower threshold for broader context
            )
            long_term_result = await self.long_term.search(query_obj)
            context_entries.extend(long_term_result.entries)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_entries = []
        for entry in context_entries:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                unique_entries.append(entry)
        
        return unique_entries[:k]
    
    async def close(self) -> None:
        """Close memory systems"""
        await self.long_term.close()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "short_term_size": self.short_term.size(),
            "short_term_max": self.short_term.max_size,
            "long_term_initialized": self.long_term.vector_store is not None
        }

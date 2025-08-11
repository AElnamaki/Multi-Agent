import asyncio
import json
import redis.asyncio as redis
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from typing import List, Optional, Dict, Any

from ..types import KnowledgeEntry, Task, AgentOutput
from ..config import settings
from ..utils.logging import logger

class MemoryStore:
    """Hybrid memory system with vector database and key-value store."""
    
    def __init__(self):
        self.redis_client = None
        self.chroma_client = None
        self.chroma_collection = None
        
    async def initialize(self):
        """Initialize both Redis and ChromaDB connections."""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="miniai_knowledge",
                metadata={"description": "MiniAI knowledge base"}
            )
            logger.info("ChromaDB connection established")
            
        except Exception as e:
            logger.error("Failed to initialize memory store", error=str(e))
            raise
    
    async def store_task(self, task: Task) -> bool:
        """Store task in Redis."""
        try:
            await self.redis_client.set(
                f"task:{task.id}",
                task.model_dump_json(),
                ex=86400  # 24 hours expiry
            )
            return True
        except Exception as e:
            logger.error("Failed to store task", task_id=task.id, error=str(e))
            return False
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task from Redis."""
        try:
            data = await self.redis_client.get(f"task:{task_id}")
            if data:
                return Task.model_validate_json(data)
            return None
        except Exception as e:
            logger.error("Failed to retrieve task", task_id=task_id, error=str(e))
            return None
    
    async def store_agent_output(self, output: AgentOutput) -> bool:
        """Store agent output in Redis."""
        try:
            await self.redis_client.set(
                f"output:{output.task_id}:{output.agent_name}",
                output.model_dump_json(),
                ex=86400
            )
            return True
        except Exception as e:
            logger.error("Failed to store agent output", output=output, error=str(e))
            return False
    
    async def store_knowledge(self, entry: KnowledgeEntry) -> bool:
        """Store knowledge entry in vector database."""
        try:
            # Store in ChromaDB
            self.chroma_collection.add(
                documents=[entry.content],
                metadatas=[{
                    "source": entry.source,
                    "timestamp": entry.timestamp.isoformat(),
                    "tags": ",".join(entry.tags)
                }],
                ids=[entry.id]
            )
            
            # Also store in Redis for quick access
            await self.redis_client.set(
                f"knowledge:{entry.id}",
                entry.model_dump_json(),
                ex=604800  # 7 days expiry
            )
            
            return True
        except Exception as e:
            logger.error("Failed to store knowledge", entry_id=entry.id, error=str(e))
            return False
    
    async def search_knowledge(self, query: str, n_results: int = 5) -> List[KnowledgeEntry]:
        """Search knowledge base using vector similarity."""
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            entries = []
            for i, doc_id in enumerate(results['ids'][0]):
                data = await self.redis_client.get(f"knowledge:{doc_id}")
                if data:
                    entries.append(KnowledgeEntry.model_validate_json(data))
            
            return entries
        except Exception as e:
            logger.error("Failed to search knowledge", query=query, error=str(e))
            return []
    
    async def cleanup(self):
        """Close connections and cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
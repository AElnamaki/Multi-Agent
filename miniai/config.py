"""
Configuration management for MiniAI framework
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
import os


class RetryConfig(BaseModel):
    """Configuration for retry behavior"""
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay: float = Field(default=0.5, ge=0.1, le=5.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.1, le=10.0)


class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: str = Field(default="openai", pattern="^(openai|anthropic|local)$")
    model: str = Field(default="gpt-4o-mini")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32000)
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)


class VectorStoreConfig(BaseModel):
    """Configuration for vector storage"""
    provider: str = Field(default="faiss", pattern="^(faiss|memory)$")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = Field(default=384, ge=1, le=4096)
    index_path: Optional[str] = Field(default="./data/faiss.index")
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)


class MemoryConfig(BaseModel):
    """Configuration for memory systems"""
    short_term_size: int = Field(default=100, ge=1, le=1000)
    long_term_db_path: str = Field(default="./data/memory.db")
    retrieval_k: int = Field(default=5, ge=1, le=50)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)


class Config(BaseModel):
    """Main configuration for MiniAI"""
    retry: RetryConfig = Field(default_factory=RetryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    
    # Agent behavior
    max_iterations: int = Field(default=10, ge=1, le=100)
    task_dedup_window: int = Field(default=200, ge=10, le=1000)
    loop_detection_threshold: int = Field(default=5, ge=2, le=20)
    batch_size: int = Field(default=5, ge=1, le=20)
    
    # Observability
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    enable_metrics: bool = Field(default=True)
    trace_requests: bool = Field(default=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        config_dict = {}
        
        # LLM config from env
        if openai_key := os.getenv("OPENAI_API_KEY"):
            config_dict["llm"] = {"api_key": openai_key}
            
        if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
            config_dict["llm"] = {"provider": "anthropic", "api_key": anthropic_key}
            
        return cls(**config_dict)

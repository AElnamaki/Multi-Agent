import os
from typing import Optional
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # LLM Configuration
    primary_llm_provider: str = os.getenv("PRIMARY_LLM_PROVIDER", "openai")
    fallback_llm_provider: str = os.getenv("FALLBACK_LLM_PROVIDER", "anthropic")
    primary_model: str = os.getenv("PRIMARY_MODEL", "gpt-4")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "claude-3-sonnet-20240229")
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))
    
    # Database Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # System Configuration
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
    retry_attempts: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    backoff_factor: float = float(os.getenv("BACKOFF_FACTOR", "2.0"))
    
    class Config:
        case_sensitive = False

settings = Settings()
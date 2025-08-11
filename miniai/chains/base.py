"""
Base chain implementation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..config import Config
from ..llm_client import AsyncLLMClient
from ..observability import get_logger, trace_operation

logger = get_logger(__name__)


class BaseChain(ABC):
    """Abstract base class for all chains"""
    
    def __init__(self, config: Config, llm_client: AsyncLLMClient):
        self.config = config
        self.llm_client = llm_client
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the chain"""
        pass
    
    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Build message list for LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages

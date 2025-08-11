from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Any, Optional

from ..types import Task, AgentOutput, LLMRequest, AgentType
from ..llm_client import LLMClient
from ..memory.knowledge_store import MemoryStore
from ..utils.logging import logger

class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""
    
    def __init__(self, agent_type: AgentType, llm_client: LLMClient, memory_store: MemoryStore):
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.name = f"{agent_type.value}_agent"
    
    @abstractmethod
    async def execute_task(self, task: Task) -> AgentOutput:
        """Execute the assigned task and return output."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent type."""
        pass
    
    async def _generate_llm_response(self, user_prompt: str, **kwargs) -> str:
        """Generate response from LLM with agent-specific system prompt."""
        request = LLMRequest(
            model=kwargs.get('model', 'gpt-4'),
            temperature=kwargs.get('temperature', 0.2),
            max_tokens=kwargs.get('max_tokens', 2000),
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        try:
            response = await self.llm_client.generate_response(request)
            return response.content
        except Exception as e:
            logger.error("LLM response generation failed", agent=self.name, error=str(e))
            raise
    
    async def _search_knowledge(self, query: str, n_results: int = 3) -> str:
        """Search knowledge base and return relevant context."""
        entries = await self.memory_store.search_knowledge(query, n_results)
        if not entries:
            return "No relevant knowledge found."
        
        context = "Relevant knowledge:\n"
        for entry in entries:
            context += f"- {entry.content}\n"
        
        return context
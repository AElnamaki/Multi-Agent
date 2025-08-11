import logging
from ..types import Task, AgentOutput, AgentType
from base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    """Agent specialized in gathering and processing information."""
    
    def __init__(self, llm_client, memory_store):
        super().__init__(AgentType.RESEARCH, llm_client, memory_store)
    
    def get_system_prompt(self) -> str:
        return """You are a Research Agent specialized in gathering, analyzing, and synthesizing information.
        
Your capabilities:
- Search and retrieve relevant information from knowledge bases
- Analyze and fact-check information sources
- Synthesize multiple sources into coherent summaries
- Identify gaps in information and suggest additional research needs

Always provide:
- Well-sourced and factual information
- Clear distinction between facts and interpretations
- Confidence levels for your findings
- Suggestions for further research if needed

Be thorough, accurate, and cite your reasoning."""
    
    async def execute_task(self, task: Task) -> AgentOutput:
        """Execute research task by gathering and analyzing information."""
        try:
            # Search existing knowledge
            context = await self._search_knowledge(task.description)
            
            # Generate research-focused prompt
            prompt = f"""Task: {task.description}
            
Existing knowledge context:
{context}

Please conduct thorough research on this topic. Provide:
1. Key findings and facts
2. Analysis of the information
3. Confidence level in your findings
4. Areas needing additional research
5. Sources and reasoning for your conclusions"""
            
            response = await self._generate_llm_response(prompt)
            
            # Calculate confidence based on available context
            confidence = 0.8 if "No relevant knowledge found" not in context else 0.6
            
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=response,
                confidence_score=confidence,
                metadata={"context_available": "No relevant knowledge found" not in context}
            )
            
        except Exception as e:
            logger.error("Research task execution failed", task_id=task.id, error=str(e))
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=f"Research failed: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
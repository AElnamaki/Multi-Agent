import aiohttp
import json
from typing import Dict, Any

class ExecutionAgent(BaseAgent):
    """Agent specialized in executing tasks and interacting with external systems."""
    
    def __init__(self, llm_client, memory_store):
        super().__init__(AgentType.EXECUTION, llm_client, memory_store)
    
    def get_system_prompt(self) -> str:
        return """You are an Execution Agent specialized in implementing plans, executing tasks, and interacting with external systems.

Your capabilities:
- Execute specific tasks and commands
- Interface with APIs and external services
- Implement solutions and run processes
- Monitor execution and handle errors
- Provide status updates and progress reports

Always provide:
- Clear execution status and results
- Error handling and recovery steps
- Progress indicators and completion metrics
- Resource usage and performance data
- Next steps or follow-up actions needed

Be efficient, reliable, and handle errors gracefully."""
    
    async def execute_task(self, task: Task) -> AgentOutput:
        """Execute implementation tasks."""
        try:
            # Get execution context
            context = await self._search_knowledge(f"execution {task.description}")
            
            prompt = f"""Task: {task.description}

Context:
{context}

Please execute this task by:
1. Analyzing the requirements
2. Determining the execution approach
3. Implementing the solution
4. Monitoring progress and handling errors
5. Providing status updates
6. Confirming successful completion

If this involves external systems or APIs, describe the integration approach."""
            
            response = await self._generate_llm_response(prompt, temperature=0.2)
            
            # Execution confidence depends on task complexity
            confidence = 0.75
            
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=response,
                confidence_score=confidence,
                metadata={"execution_type": "task_implementation"}
            )
            
        except Exception as e:
            logger.error("Execution task failed", task_id=task.id, error=str(e))
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=f"Execution failed: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )

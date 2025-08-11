"""
Task execution chain implementation
"""

from typing import Dict, Any, List
from ..models import ExecutionRequest, ExecutionResult, MemoryEntry, TaskStatus
from .base import BaseChain
from ..observability import trace_operation
import time
import json


class ExecutionChain(BaseChain):
    """Chain for executing individual tasks"""
    
    SYSTEM_PROMPT = """You are a task execution AI that performs tasks to help achieve objectives.

Your job is to execute the given task using the provided context and return a structured result.

CRITICAL RULES:
1. Return ONLY a valid JSON object with this exact structure:
{
  "result_text": "detailed description of what was accomplished",
  "structured_facts": [{"fact": "specific fact learned", "confidence": 0.95}],
  "actionable_steps": ["step 1", "step 2"],
  "artifact_urls": ["url1", "url2"],
  "status": "success|partial|failed"
}

2. Be thorough and specific in your execution
3. Extract key facts and insights
4. Suggest concrete next steps
5. Use "partial" status if task is started but needs more work
6. Use "failed" status only if task cannot be completed at all
7. Include relevant URLs or references in artifact_urls"""

    USER_PROMPT_TEMPLATE = """OBJECTIVE: {objective}

TASK TO EXECUTE: {task_name}

RELEVANT CONTEXT:
{context}

Execute this task thoroughly. Consider the context provided and work toward the overall objective.

Examples of good responses:

Example 1:
{{
  "result_text": "Completed market research on competitor pricing. Found 3 main competitors with prices ranging from $10-25/month for similar services. Identified key differentiators in features and customer segments.",
  "structured_facts": [
    {{"fact": "Competitor A charges $15/month for basic plan", "confidence": 0.95}},
    {{"fact": "Market average is $18/month", "confidence": 0.85}}
  ],
  "actionable_steps": [
    "Analyze our feature set vs competitors",
    "Survey potential customers on price sensitivity",
    "Draft pricing strategy document"
  ],
  "artifact_urls": [],
  "status": "success"
}}

Example 2:
{{
  "result_text": "Started drafting the project timeline but need additional information about resource availability and dependencies to complete it accurately.",
  "structured_facts": [
    {{"fact": "Project has 4 main phases identified", "confidence": 0.9}}
  ],
  "actionable_steps": [
    "Meet with team leads to discuss resource allocation",
    "Identify critical path dependencies",
    "Finalize timeline with buffer periods"
  ],
  "artifact_urls": [],
  "status": "partial"
}}

Execute the task now:"""

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute the task"""
        start_time = time.time()
        
        with trace_operation("execution_chain.execute", {"task_name": request.task.task_name}):
            # Format context
            context_lines = []
            for entry in request.context:
                context_lines.append(f"- {entry.text[:200]}...")  # Truncate long entries
            context_str = "\n".join(context_lines) if context_lines else "No relevant context available"
            
            # Build user prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                objective=request.objective,
                task_name=request.task.task_name,
                context=context_str
            )
            
            messages = self._build_messages(self.SYSTEM_PROMPT, user_prompt)
            
            try:
                # Call LLM
                response = await self.llm_client.generate_structured(
                    messages=messages,
                    schema_description='{"result_text": str, "structured_facts": [{"fact": str, "confidence": float}], "actionable_steps": [str], "artifact_urls": [str], "status": "success|partial|failed"}'
                )
                
                execution_time = time.time() - start_time
                
                # Create execution result
                result = ExecutionResult(
                    task_id=request.task.task_id,
                    task_name=request.task.task_name,
                    result_text=response.get("result_text", ""),
                    structured_facts=response.get("structured_facts", []),
                    actionable_steps=response.get("actionable_steps", []),
                    artifact_urls=response.get("artifact_urls", []),
                    status=TaskStatus(response.get("status", "success")),
                    execution_time=execution_time,
                    tokens_used=0  # TODO: Extract from LLM response
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Return failed result
                return ExecutionResult(
                    task_id=request.task.task_id,
                    task_name=request.task.task_name,
                    result_text=f"Task execution failed: {str(e)}",
                    status=TaskStatus.FAILED,
                    error=str(e),
                    execution_time=execution_time
                )
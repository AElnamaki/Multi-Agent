"""
Task creation chain implementation
"""

from typing import Dict, Any, List
from ..models import TaskCreationRequest, TaskCreationResponse, Task, ExecutionResult
from .base import BaseChain
from ..observability import trace_operation
import json


class TaskCreationChain(BaseChain):
    """Chain for creating new tasks based on execution results"""
    
    SYSTEM_PROMPT = """You are a task creation AI that generates new tasks based on execution results and objectives.

Your job is to analyze the result of a completed task and generate new tasks that will help achieve the overall objective.

CRITICAL RULES:
1. Return ONLY a valid JSON object with the exact structure: {"tasks": [{"task_name": "...", "meta": {"priority_reasons": "...", "expected_time_mins": 10}}]}
2. Each task must have a clear, actionable task_name (string)
3. Do not create duplicate tasks that already exist in incomplete_tasks
4. Generate 0-5 new tasks maximum
5. Tasks should be specific, measurable, and directly related to the objective
6. If no new tasks are needed, return {"tasks": []}"""

    USER_PROMPT_TEMPLATE = """OBJECTIVE: {objective}

COMPLETED TASK: {task_description}
RESULT: {result}

INCOMPLETE TASKS:
{incomplete_tasks}

Based on the completed task result and the objective, generate new tasks that need to be done.

Examples of good responses:

Example 1:
{{"tasks": [{{"task_name": "Research competitor pricing strategies", "meta": {{"priority_reasons": "Need market data before setting prices", "expected_time_mins": 15}}}}, {{"task_name": "Create price comparison spreadsheet", "meta": {{"priority_reasons": "Organize pricing data for analysis", "expected_time_mins": 10}}}}]}}

Example 2:
{{"tasks": [{{"task_name": "Draft initial project timeline", "meta": {{"priority_reasons": "Timeline needed to coordinate team", "expected_time_mins": 20}}}}]}}

Example 3:
{{"tasks": []}}

Generate new tasks now:"""

    async def execute(self, request: TaskCreationRequest) -> TaskCreationResponse:
        """Execute task creation"""
        with trace_operation("task_creation_chain.execute"):
            # Format incomplete tasks
            incomplete_task_names = [task.task_name for task in request.incomplete_tasks]
            incomplete_str = "\n".join(f"- {name}" for name in incomplete_task_names) or "None"
            
            # Build user prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                objective=request.objective,
                task_description=request.task_description,
                result=request.result.result_text,
                incomplete_tasks=incomplete_str
            )
            
            messages = self._build_messages(self.SYSTEM_PROMPT, user_prompt)
            
            # Call LLM
            response = await self.llm_client.generate_structured(
                messages=messages,
                schema_description='{"tasks": [{"task_name": str, "meta": {"priority_reasons": str, "expected_time_mins": int}}]}'
            )
            
            # Validate response
            task_response = TaskCreationResponse(**response)
            
            return task_response
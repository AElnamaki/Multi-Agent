"""
Task prioritization chain implementation
"""

from typing import Dict, Any, List
from ..models import TaskPrioritizationRequest, TaskPrioritizationResponse, Task
from .base import BaseChain
from ..observability import trace_operation
import json


class TaskPrioritizationChain(BaseChain):
    """Chain for prioritizing tasks in the queue"""
    
    SYSTEM_PROMPT = """You are a task prioritization AI that scores and ranks tasks based on their importance and urgency relative to the objective.

Your job is to analyze a list of tasks and assign priority scores that reflect their importance for achieving the objective.

CRITICAL RULES:
1. Return ONLY a valid JSON object: {"tasks": [{"task_id": int, "task_name": str, "score": float, "estimated_effort": float}]}
2. Score should be between 0.0 and 1.0 (1.0 = highest priority)
3. estimated_effort should be in arbitrary units (1.0 = baseline, higher = more effort)
4. Include ALL tasks from the input list
5. Order tasks by priority score (highest first)
6. Consider dependencies, urgency, and impact on the objective"""

    USER_PROMPT_TEMPLATE = """OBJECTIVE: {objective}

TASKS TO PRIORITIZE:
{task_list}

Consider these factors when scoring:
- How critical is this task for the objective?
- Are there dependencies (should this be done before others)?
- What's the impact vs effort ratio?
- How urgent is this task?

Examples of good responses:

Example 1:
{{"tasks": [{{"task_id": 1, "task_name": "Define project requirements", "score": 0.95, "estimated_effort": 2.0}}, {{"task_id": 2, "task_name": "Set up development environment", "score": 0.8, "estimated_effort": 1.5}}, {{"task_id": 3, "task_name": "Write documentation", "score": 0.3, "estimated_effort": 3.0}}]}}

Example 2:
{{"tasks": [{{"task_id": 5, "task_name": "Gather user feedback", "score": 0.9, "estimated_effort": 2.5}}, {{"task_id": 6, "task_name": "Update website copy", "score": 0.4, "estimated_effort": 1.0}}]}}

Prioritize the tasks now:"""

    async def execute(self, request: TaskPrioritizationRequest) -> TaskPrioritizationResponse:
        """Execute task prioritization"""
        with trace_operation("task_prioritization_chain.execute"):
            # Format task list
            task_lines = []
            for task in request.task_list:
                task_lines.append(f"ID: {task.task_id}, Task: {task.task_name}")
            task_list_str = "\n".join(task_lines)
            
            # Build user prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                objective=request.objective,
                task_list=task_list_str
            )
            
            messages = self._build_messages(self.SYSTEM_PROMPT, user_prompt)
            
            # Call LLM
            response = await self.llm_client.generate_structured(
                messages=messages,
                schema_description='{"tasks": [{"task_id": int, "task_name": str, "score": float, "estimated_effort": float}]}'
            )
            
            # Validate response
            priority_response = TaskPrioritizationResponse(**response)
            
            return priority_response
import asyncio
from typing import List, Dict, Any, Optional
import re

from .types import Task, AgentOutput, TaskStatus, AgentType
from .agents.base_agent import BaseAgent
from .agents.research_agent import ResearchAgent
from .agents.reasoning_agent import ReasoningAgent
from .agents.planning_agent import PlanningAgent
from .agents.execution_agent import ExecutionAgent
from .agents.evaluation_agent import EvaluationAgent
from .llm_client import LLMClient
from .memory.knowledge_store import MemoryStore
from .config import settings
from .utils.logging import logger

class CoordinatorAgent:
    """Main coordinator that orchestrates the multi-agent workflow."""
    
    def __init__(self, llm_client: LLMClient, memory_store: MemoryStore):
        self.llm_client = llm_client
        self.memory_store = memory_store
        
        # Initialize specialized agents
        self.agents: Dict[AgentType, BaseAgent] = {
            AgentType.RESEARCH: ResearchAgent(llm_client, memory_store),
            AgentType.REASONING: ReasoningAgent(llm_client, memory_store),
            AgentType.PLANNING: PlanningAgent(llm_client, memory_store),
            AgentType.EXECUTION: ExecutionAgent(llm_client, memory_store),
            AgentType.EVALUATION: EvaluationAgent(llm_client, memory_store)
        }
        
        self.active_tasks: Dict[str, Task] = {}
        self.task_semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
    
    async def process_goal(self, goal: str, priority: int = 5) -> Dict[str, Any]:
        """Process a high-level goal by breaking it down and coordinating execution."""
        logger.info("Processing new goal", goal=goal, priority=priority)
        
        try:
            # Step 1: Parse and classify the goal
            goal_analysis = await self._analyze_goal(goal)
            
            # Step 2: Generate sub-tasks
            sub_tasks = await self._generate_sub_tasks(goal, goal_analysis, priority)
            
            # Step 3: Execute tasks with coordination
            results = await self._execute_tasks_coordinated(sub_tasks)
            
            # Step 4: Integrate and validate results
            final_result = await self._integrate_results(goal, results)
            
            # Step 5: Store learnings in memory
            await self._store_learnings(goal, final_result)
            
            return {
                "goal": goal,
                "status": "completed",
                "result": final_result,
                "sub_tasks_completed": len([r for r in results if r.confidence_score > 0.5]),
                "total_sub_tasks": len(results)
            }
            
        except Exception as e:
            logger.error("Goal processing failed", goal=goal, error=str(e))
            return {
                "goal": goal,
                "status": "failed",
                "error": str(e),
                "result": None
            }
    
    async def _analyze_goal(self, goal: str) -> Dict[str, Any]:
        """Analyze goal to determine complexity, required agents, and approach."""
        analysis_prompt = f"""Analyze this goal and provide a structured analysis:

Goal: {goal}

Please provide:
1. Goal classification (research, reasoning, planning, execution, mixed)
2. Complexity level (1-10)
3. Required agent types (research, reasoning, planning, execution, evaluation)
4. Estimated sub-tasks needed
5. Dependencies and execution order
6. Success criteria

Format your response as structured analysis."""
        
        from .types import LLMRequest
        
        request = LLMRequest(
            model=settings.primary_model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing goals and determining optimal execution strategies."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.2
        )
        
        response = await self.llm_client.generate_response(request)
        
        # Parse response into structured format (simplified)
        return {
            "raw_analysis": response.content,
            "complexity": self._extract_complexity(response.content),
            "required_agents": self._extract_required_agents(response.content)
        }
    
    def _extract_complexity(self, analysis: str) -> int:
        """Extract complexity level from analysis text."""
        # Simple regex to find complexity number
        match = re.search(r'complexity.*?(\d+)', analysis.lower())
        return int(match.group(1)) if match else 5
    
    def _extract_required_agents(self, analysis: str) -> List[AgentType]:
        """Extract required agent types from analysis text."""
        required_agents = []
        analysis_lower = analysis.lower()
        
        if "research" in analysis_lower:
            required_agents.append(AgentType.RESEARCH)
        if "reasoning" in analysis_lower:
            required_agents.append(AgentType.REASONING)
        if "planning" in analysis_lower:
            required_agents.append(AgentType.PLANNING)
        if "execution" in analysis_lower:
            required_agents.append(AgentType.EXECUTION)
        
        # Always include evaluation for quality assurance
        required_agents.append(AgentType.EVALUATION)
        
        return required_agents if required_agents else [AgentType.RESEARCH, AgentType.EVALUATION]
    
    async def _generate_sub_tasks(self, goal: str, analysis: Dict[str, Any], priority: int) -> List[Task]:
        """Generate specific sub-tasks based on goal analysis."""
        sub_task_prompt = f"""Break down this goal into specific, actionable sub-tasks:

Goal: {goal}
Analysis: {analysis['raw_analysis']}

Generate 3-7 specific sub-tasks that:
1. Are concrete and actionable
2. Can be assigned to specific agent types
3. Build towards the overall goal
4. Have clear success criteria

Format each sub-task as:
TASK: [description]
AGENT: [agent_type]
PRIORITY: [1-10]"""
        
        from .types import LLMRequest
        
        request = LLMRequest(
            model=settings.primary_model,
            messages=[
                {"role": "system", "content": "You are expert at breaking down complex goals into manageable sub-tasks."},
                {"role": "user", "content": sub_task_prompt}
            ],
            temperature=0.3
        )
        
        response = await self.llm_client.generate_response(request)
        
        # Parse sub-tasks from response
        sub_tasks = self._parse_sub_tasks(response.content, priority)
        
        # Store tasks in memory
        for task in sub_tasks:
            await self.memory_store.store_task(task)
            self.active_tasks[task.id] = task
        
        return sub_tasks
    
    def _parse_sub_tasks(self, response: str, base_priority: int) -> List[Task]:
        """Parse sub-tasks from LLM response."""
        tasks = []
        lines = response.split('\n')
        
        current_task = None
        current_agent = None
        current_priority = base_priority
        
        for line in lines:
            line = line.strip()
            if line.startswith('TASK:'):
                current_task = line.replace('TASK:', '').strip()
            elif line.startswith('AGENT:'):
                agent_str = line.replace('AGENT:', '').strip().lower()
                # Map agent string to AgentType
                if 'research' in agent_str:
                    current_agent = AgentType.RESEARCH
                elif 'reasoning' in agent_str:
                    current_agent = AgentType.REASONING
                elif 'planning' in agent_str:
                    current_agent = AgentType.PLANNING
                elif 'execution' in agent_str:
                    current_agent = AgentType.EXECUTION
                elif 'evaluation' in agent_str:
                    current_agent = AgentType.EVALUATION
            elif line.startswith('PRIORITY:'):
                try:
                    current_priority = int(re.search(r'\d+', line).group())
                except:
                    current_priority = base_priority
            
            # If we have complete task info, create Task object
            if current_task and current_agent:
                task = Task(
                    description=current_task,
                    priority=current_priority,
                    assigned_agent=current_agent.value
                )
                tasks.append(task)
                current_task = None
                current_agent = None
                current_priority = base_priority
        
        return tasks
    
    async def _execute_tasks_coordinated(self, tasks: List[Task]) -> List[AgentOutput]:
        """Execute tasks with proper coordination and dependency management."""
        results = []
        
        # Group tasks by agent type for efficient execution
        tasks_by_agent = {}
        for task in tasks:
            agent_type = AgentType(task.assigned_agent)
            if agent_type not in tasks_by_agent:
                tasks_by_agent[agent_type] = []
            tasks_by_agent[agent_type].append(task)
        
        # Execute tasks with semaphore for concurrency control
        async def execute_single_task(task: Task) -> AgentOutput:
            async with self.task_semaphore:
                agent_type = AgentType(task.assigned_agent)
                agent = self.agents[agent_type]
                
                # Update task status
                task.status = TaskStatus.IN_PROGRESS
                await self.memory_store.store_task(task)
                
                try:
                    result = await agent.execute_task(task)
                    task.status = TaskStatus.DONE
                    
                    # Store output in memory
                    await self.memory_store.store_agent_output(result)
                    
                    return result
                    
                except Exception as e:
                    task.status = TaskStatus.ERROR
                    logger.error("Task execution failed", task_id=task.id, error=str(e))
                    return AgentOutput(
                        task_id=task.id,
                        agent_name=agent.name,
                        output=f"Task failed: {str(e)}",
                        confidence_score=0.0,
                        metadata={"error": str(e)}
                    )
                finally:
                    await self.memory_store.store_task(task)
        
        # Execute all tasks concurrently
        execution_tasks = [execute_single_task(task) for task in tasks]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Task execution exception", task_id=tasks[i].id, error=str(result))
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _integrate_results(self, goal: str, results: List[AgentOutput]) -> str:
        """Integrate all agent outputs into a coherent final result."""
        if not results:
            return "No results to integrate."
        
        # Compile all outputs
        compiled_outputs = "\n\n".join([
            f"**{result.agent_name}** (confidence: {result.confidence_score:.2f}):\n{result.output}"
            for result in results
        ])
        
        integration_prompt = f"""Integrate these agent outputs into a coherent, comprehensive response to the original goal:

Original Goal: {goal}

Agent Outputs:
{compiled_outputs}

Please provide:
1. A comprehensive answer to the original goal
2. Key insights from the various agents
3. Any conflicts or inconsistencies resolved
4. Overall confidence in the integrated result
5. Recommendations for follow-up actions if needed

Focus on creating a cohesive, actionable response."""
        
        from .types import LLMRequest
        
        request = LLMRequest(
            model=settings.primary_model,
            messages=[
                {"role": "system", "content": "You are an expert at synthesizing multiple perspectives into coherent, actionable results."},
                {"role": "user", "content": integration_prompt}
            ],
            temperature=0.2
        )
        
        response = await self.llm_client.generate_response(request)
        return response.content
    
    async def _store_learnings(self, goal: str, result: str):
        """Store successful patterns and learnings for future use."""
        from .types import KnowledgeEntry
        
        knowledge_entry = KnowledgeEntry(
            content=f"Goal: {goal}\n\nResult: {result}",
            source="coordinator_learnings",
            tags=["goal_completion", "learning", "pattern"]
        )
        
        await self.memory_store.store_knowledge(knowledge_entry)
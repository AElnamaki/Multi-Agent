"""
Main agent controller that orchestrates the task execution loop
"""

import asyncio
from typing import List, Optional, Dict, Any
from queue import PriorityQueue
import time
from ..config import Config
from ..models import Task, ExecutionResult, TaskStatus, ExecutionRequest, TaskCreationRequest, TaskPrioritizationRequest
from ..memory import HybridMemoryManager
from ..llm_client import AsyncLLMClient
from ..chains import TaskCreationChain, TaskPrioritizationChain, ExecutionChain
from ..utils import TaskDeduplicator, LoopDetector
from ..observability import get_logger, trace_operation, get_metrics

logger = get_logger(__name__)


class AgentController:
    """Main controller for the MiniAI agent"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory = HybridMemoryManager(config)
        self.llm_client = AsyncLLMClient(config)
        
        # Initialize chains
        self.task_creation = TaskCreationChain(config, self.llm_client)
        self.task_prioritization = TaskPrioritizationChain(config, self.llm_client)
        self.execution = ExecutionChain(config, self.llm_client)
        
        # Task management
        self.task_queue: List[Task] = []
        self.task_counter = 1
        self.completed_tasks: List[ExecutionResult] = []
        
        # Loop detection and deduplication
        self.deduplicator = TaskDeduplicator(
            similarity_threshold=config.vectorstore.similarity_threshold,
            window_size=config.task_dedup_window
        )
        self.loop_detector = LoopDetector(
            window_size=config.task_dedup_window,
            threshold=config.loop_detection_threshold
        )
        
        self.metrics = get_metrics()
    
    async def initialize(self) -> None:
        """Initialize the agent controller"""
        await self.memory.initialize()
        logger.info("Agent controller initialized")
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.memory.close()
        await self.llm_client.close()
    
    def add_initial_task(self, task_name: str, meta: Optional[Dict[str, Any]] = None) -> Task:
        """Add initial task to the queue"""
        task = Task(
            task_id=self.task_counter,
            task_name=task_name,
            meta=meta or {},
            priority_score=1.0  # Initial task gets highest priority
        )
        self.task_counter += 1
        self.task_queue.append(task)
        
        logger.info(f"Added initial task: {task_name}")
        return task
    
    def _get_highest_priority_task(self) -> Optional[Task]:
        """Get the highest priority task from queue"""
        if not self.task_queue:
            return None
        
        # Sort by priority score (highest first)
        self.task_queue.sort(key=lambda t: t.priority_score, reverse=True)
        return self.task_queue.pop(0)
    
    async def _execute_task(self, task: Task, objective: str) -> ExecutionResult:
        """Execute a single task"""
        with trace_operation("controller.execute_task", {"task_name": task.task_name}):
            # Retrieve context from memory
            context_entries = await self.memory.retrieve_context(
                query=f"{objective} {task.task_name}",
                k=self.config.memory.retrieval_k
            )
            
            # Create execution request
            request = ExecutionRequest(
                objective=objective,
                context=context_entries,
                task=task
            )
            
            # Execute task
            result = await self.execution.execute(request)
            
            self.metrics.increment("tasks.executed", {"status": result.status})
            self.metrics.histogram("tasks.execution_time", result.execution_time)
            
            return result
    
    async def _create_new_tasks(self, objective: str, result: ExecutionResult) -> List[Task]:
        """Create new tasks based on execution result"""
        with trace_operation("controller.create_tasks"):
            # Create task creation request
            request = TaskCreationRequest(
                objective=objective,
                result=result,
                task_description=result.task_name,
                incomplete_tasks=self.task_queue
            )
            
            # Generate new tasks
            response = await self.task_creation.execute(request)
            
            # Convert to Task objects
            new_tasks = []
            for task_data in response.tasks:
                # Check for duplicates
                if not self.deduplicator.is_duplicate(task_data["task_name"]):
                    task = Task(
                        task_id=self.task_counter,
                        task_name=task_data["task_name"],
                        meta=task_data.get("meta", {}),
                        priority_score=0.5  # Default priority, will be updated
                    )
                    self.task_counter += 1
                    new_tasks.append(task)
                    
                    # Add to deduplicator
                    self.deduplicator.add_task(task_data["task_name"])
                else:
                    self.metrics.increment("tasks.duplicates_filtered")
            
            logger.info(f"Created {len(new_tasks)} new tasks")
            return new_tasks
    
    async def _prioritize_tasks(self, tasks: List[Task], objective: str) -> List[Task]:
        """Prioritize task queue"""
        if not tasks:
            return []
        
        with trace_operation("controller.prioritize_tasks"):
            request = TaskPrioritizationRequest(
                task_list=tasks,
                objective=objective,
                next_task_id=self.task_counter
            )
            
            response = await self.task_prioritization.execute(request)
            
            # Update task priorities
            priority_map = {t["task_id"]: t["score"] for t in response.tasks}
            effort_map = {t["task_id"]: t["estimated_effort"] for t in response.tasks}
            
            for task in tasks:
                if task.task_id in priority_map:
                    task.priority_score = priority_map[task.task_id]
                    task.estimated_effort = effort_map[task.task_id]
            
            # Sort by priority
            tasks.sort(key=lambda t: t.priority_score, reverse=True)
            
            logger.info(f"Prioritized {len(tasks)} tasks")
            return tasks
    
    async def run(self, objective: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Run the main agent loop"""
        if not self.task_queue:
            raise ValueError("No initial tasks provided. Call add_initial_task() first.")
        
        max_iter = max_iterations or self.config.max_iterations
        start_time = time.time()
        iteration = 0
        
        logger.info(f"Starting agent run with objective: {objective}")
        
        try:
            while iteration < max_iter and self.task_queue:
                iteration += 1
                
                with trace_operation("controller.iteration", {"iteration": iteration}):
                    # Get next task
                    current_task = self._get_highest_priority_task()
                    if not current_task:
                        break
                    
                    logger.info(f"Iteration {iteration}: Executing task '{current_task.task_name}'")
                    
                    # Check for loops
                    if self.loop_detector.add_task(current_task.task_name):
                        logger.warning(f"Loop detected for task: {current_task.task_name}")
                        self.metrics.increment("tasks.loops_detected")
                        continue
                    
                    # Execute task
                    result = await self._execute_task(current_task, objective)
                    self.completed_tasks.append(result)
                    
                    # Store result in memory
                    await self.memory.store_result(result)
                    
                    # Only create new tasks if execution was successful or partial
                    if result.status in [TaskStatus.SUCCESS, TaskStatus.PARTIAL]:
                        # Create new tasks
                        new_tasks = await self._create_new_tasks(objective, result)
                        
                        # Add new tasks to queue
                        self.task_queue.extend(new_tasks)
                        
                        # Prioritize entire queue
                        self.task_queue = await self._prioritize_tasks(self.task_queue, objective)
                    
                    # Log progress
                    logger.info(f"Iteration {iteration} complete. Queue size: {len(self.task_queue)}")
            
            # Final results
            total_time = time.time() - start_time
            
            results = {
                "objective": objective,
                "iterations": iteration,
                "total_time": total_time,
                "tasks_completed": len(self.completed_tasks),
                "tasks_remaining": len(self.task_queue),
                "successful_tasks": len([r for r in self.completed_tasks if r.status == TaskStatus.SUCCESS]),
                "failed_tasks": len([r for r in self.completed_tasks if r.status == TaskStatus.FAILED]),
                "memory_stats": self.memory.get_memory_stats(),
                "completed_tasks": self.completed_tasks,
                "remaining_tasks": self.task_queue
            }
            
            # Log final metrics
            self.metrics.histogram("agent.total_time", total_time)
            self.metrics.gauge("agent.tasks_completed", len(self.completed_tasks))
            self.metrics.gauge("agent.tasks_remaining", len(self.task_queue))
            
            logger.info(f"Agent run completed. {len(self.completed_tasks)} tasks completed, {len(self.task_queue)} remaining")
            
            return results
            
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            raise

"""
Pydantic models and JSON schemas for MiniAI
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class Task(BaseModel):
    """Task model with validation"""
    task_id: int = Field(..., ge=1, description="Unique task identifier")
    task_name: str = Field(..., min_length=1, max_length=500, description="Task description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    meta: Dict[str, Any] = Field(default_factory=dict)
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_effort: float = Field(default=1.0, ge=0.1, le=100.0)
    
    @validator('task_name')
    def validate_task_name(cls, v):
        if not v.strip():
            raise ValueError("Task name cannot be empty")
        return v.strip()


class ExecutionResult(BaseModel):
    """Result of task execution"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: int = Field(..., ge=1)
    task_name: str = Field(..., min_length=1)
    result_text: str = Field(default="")
    structured_facts: List[Dict[str, Any]] = Field(default_factory=list)
    actionable_steps: List[str] = Field(default_factory=list)
    artifact_urls: List[str] = Field(default_factory=list)
    status: TaskStatus = Field(default=TaskStatus.SUCCESS)
    error: Optional[str] = Field(default=None)
    execution_time: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryEntry(BaseModel):
    """Memory entry for vectorstore"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        # Allow numpy arrays for embeddings
        arbitrary_types_allowed = True


class TaskCreationRequest(BaseModel):
    """Request for task creation chain"""
    objective: str = Field(..., min_length=1)
    result: ExecutionResult
    task_description: str = Field(..., min_length=1)
    incomplete_tasks: List[Task] = Field(default_factory=list)


class TaskCreationResponse(BaseModel):
    """Response from task creation chain"""
    tasks: List[Dict[str, Any]] = Field(..., min_items=0, max_items=20)
    
    @validator('tasks')
    def validate_tasks(cls, v):
        for task in v:
            if 'task_name' not in task:
                raise ValueError("Each task must have a task_name")
            if not isinstance(task['task_name'], str) or not task['task_name'].strip():
                raise ValueError("task_name must be a non-empty string")
        return v


class TaskPrioritizationRequest(BaseModel):
    """Request for task prioritization chain"""
    task_list: List[Task] = Field(..., min_items=1)
    objective: str = Field(..., min_length=1)
    next_task_id: int = Field(..., ge=1)


class TaskPrioritizationResponse(BaseModel):
    """Response from task prioritization chain"""
    tasks: List[Dict[str, Any]] = Field(..., min_items=1)
    
    @validator('tasks')
    def validate_tasks(cls, v):
        required_fields = ['task_id', 'task_name', 'score', 'estimated_effort']
        for task in v:
            for field in required_fields:
                if field not in task:
                    raise ValueError(f"Each task must have {field}")
            
            if not isinstance(task['score'], (int, float)) or task['score'] < 0:
                raise ValueError("score must be a non-negative number")
                
            if not isinstance(task['estimated_effort'], (int, float)) or task['estimated_effort'] <= 0:
                raise ValueError("estimated_effort must be a positive number")
        return v


class ExecutionRequest(BaseModel):
    """Request for execution chain"""
    objective: str = Field(..., min_length=1)
    context: List[MemoryEntry] = Field(default_factory=list)
    task: Task = Field(...)


class RetrievalQuery(BaseModel):
    """Query for memory retrieval"""
    query_text: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    filter_metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    """Result from memory retrieval"""
    entries: List[MemoryEntry] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    total_found: int = Field(default=0, ge=0)
    query_time: float = Field(default=0.0, ge=0.0)

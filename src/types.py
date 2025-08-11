
from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    RESEARCH = "research"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    EVALUATION = "evaluation"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    priority: int = Field(default=1, ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    parent_task_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentOutput(BaseModel):
    task_id: str
    agent_name: str
    output: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class KnowledgeEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: Optional[List[float]] = None
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

class LLMRequest(BaseModel):
    model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    messages: List[Dict[str, str]]
    provider: Optional[str] = None

class LLMResponse(BaseModel):
    content: str
    usage_tokens: int
    model_used: str
    provider_used: str
    timestamp: datetime = Field(default_factory=datetime.now)
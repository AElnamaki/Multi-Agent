"""
MiniAI Framework - A lightweight agent orchestration framework
"""

__version__ = "0.1.0"

from .config import Config
from chains.controller import AgentController
from .models import Task, ExecutionResult, MemoryEntry

__all__ = ["Config", "AgentController", "Task", "ExecutionResult", "MemoryEntry"]

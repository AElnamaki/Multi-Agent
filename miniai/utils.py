"""
Utility functions for MiniAI
"""

import hashlib
import re
from typing import List, Set, Tuple
from .models import Task
import numpy as np


def normalize_task_name(task_name: str) -> str:
    """Normalize task name for deduplication"""
    # Remove extra whitespace and convert to lowercase
    normalized = re.sub(r'\s+', ' ', task_name.strip().lower())
    
    # Remove common variations
    normalized = re.sub(r'^(task:?|step:?|action:?)\s*', '', normalized)
    normalized = re.sub(r'\s*(task|step|action)\s*, '', normalized)
    
    return normalized


def compute_task_hash(task_name: str) -> str:
    """Compute hash for task deduplication"""
    normalized = normalize_task_name(task_name)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    if len(a) != len(b):
        return 0.0
    
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
    
    return np.dot(a, b) / (a_norm * b_norm)


class TaskDeduplicator:
    """Deduplicates tasks using hash and embedding similarity"""
    
    def __init__(self, similarity_threshold: float = 0.92, window_size: int = 200):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.task_hashes: Set[str] = set()
        self.task_embeddings: List[Tuple[str, List[float]]] = []
    
    def is_duplicate(self, task_name: str, embedding: List[float] = None) -> bool:
        """Check if task is duplicate using hash and embedding similarity"""
        task_hash = compute_task_hash(task_name)
        
        # Check exact hash match
        if task_hash in self.task_hashes:
            return True
        
        # Check embedding similarity if provided
        if embedding:
            for stored_name, stored_embedding in self.task_embeddings:
                similarity = cosine_similarity(embedding, stored_embedding)
                if similarity > self.similarity_threshold:
                    return True
        
        return False
    
    def add_task(self, task_name: str, embedding: List[float] = None) -> None:
        """Add task to deduplication tracking"""
        task_hash = compute_task_hash(task_name)
        self.task_hashes.add(task_hash)
        
        if embedding:
            self.task_embeddings.append((task_name, embedding))
            
            # Maintain window size
            if len(self.task_embeddings) > self.window_size:
                self.task_embeddings = self.task_embeddings[-self.window_size:]


class LoopDetector:
    """Detects infinite loops in task generation"""
    
    def __init__(self, window_size: int = 50, threshold: int = 5):
        self.window_size = window_size
        self.threshold = threshold
        self.task_history: List[str] = []
    
    def add_task(self, task_name: str) -> bool:
        """Add task and return True if loop detected"""
        normalized = normalize_task_name(task_name)
        self.task_history.append(normalized)
        
        # Maintain window size
        if len(self.task_history) > self.window_size:
            self.task_history = self.task_history[-self.window_size:]
        
        # Count occurrences of this task in recent history
        count = self.task_history.count(normalized)
        
        return count >= self.threshold


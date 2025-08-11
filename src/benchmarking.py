import asyncio
import time
from typing import List, Dict, Any, Tuple
import json
import statistics

from .coordinator import CoordinatorAgent
from .types import Task, AgentOutput
from .utils.logging import logger

class BenchmarkSuite:
    """Comprehensive benchmarking and evaluation system."""
    
    def __init__(self, coordinator: CoordinatorAgent):
        self.coordinator = coordinator
        self.benchmark_results = []
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting full benchmark suite")
        
        results = {
            "reasoning_benchmarks": await self._run_reasoning_benchmarks(),
            "efficiency_benchmarks": await self._run_efficiency_benchmarks(),
            "error_handling_benchmarks": await self._run_error_handling_benchmarks(),
            "scalability_benchmarks": await self._run_scalability_benchmarks(),
            "summary": {}
        }
        
        # Calculate summary statistics
        results["summary"] = self._calculate_summary_stats(results)
        
        logger.info("Benchmark suite completed", results=results["summary"])
        return results
    
    async def _run_reasoning_benchmarks(self) -> Dict[str, Any]:
        """Test reasoning capabilities with standard benchmarks."""
        reasoning_tasks = [
            {
                "name": "Mathematical Reasoning",
                "goal": "Solve this step by step: If a train travels 120 miles in 2 hours, then speeds up by 25% for the next 3 hours, how far does it travel in total?",
                "expected_pattern": ["60", "mph", "75", "mph", "345", "miles"]
            },
            {
                "name": "Logical Deduction",
                "goal": "All birds can fly. Penguins are birds. Penguins cannot fly. Identify the logical inconsistency and explain the resolution.",
                "expected_pattern": ["inconsistency", "contradiction", "premise"]
            },
            {
                "name": "Causal Reasoning",
                "goal": "Analyze the causal relationship: Coffee shop sales increase 30% when it rains. What are three possible explanations?",
                "expected_pattern": ["indoor", "comfort", "shelter"]
            }
        ]
        
        results = []
        for task in reasoning_tasks:
            start_time = time.time()
            result = await self.coordinator.process_goal(task["goal"])
            execution_time = time.time() - start_time
            
            # Check if expected patterns are present in result
            accuracy_score = self._calculate_accuracy_score(
                result.get("result", ""), 
                task["expected_pattern"]
            )
            
            results.append({
                "name": task["name"],
                "execution_time": execution_time,
                "accuracy_score": accuracy_score,
                "success": result.get("status") == "completed",
                "confidence": self._extract_confidence_from_result(result)
            })
        
        return {
            "individual_results": results,
            "average_accuracy": statistics.mean([r["accuracy_score"] for r in results]),
            "average_time": statistics.mean([r["execution_time"] for r in results]),
            "success_rate": sum([1 for r in results if r["success"]]) / len(results)
        }
    
    async def _run_efficiency_benchmarks(self) -> Dict[str, Any]:
        """Test system efficiency and resource usage."""
        efficiency_tasks = [
            "Create a simple project plan for building a mobile app",
            "Research the benefits and drawbacks of renewable energy",
            "Explain quantum computing to a 10-year-old",
            "Design a marketing strategy for a new restaurant"
        ]
        
        results = []
        total_tokens = 0
        
        for task in efficiency_tasks:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            result = await self.coordinator.process_goal(task)
            
            execution_time = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory
            
            # Estimate token usage (simplified)
            estimated_tokens = len(result.get("result", "").split()) * 1.3
            total_tokens += estimated_tokens
            
            results.append({
                "task": task[:50] + "..." if len(task) > 50 else task,
                "execution_time": execution_time,
                "memory_used": memory_used,
                "estimated_tokens": estimated_tokens,
                "success": result.get("status") == "completed"
            })
        
        return {
            "individual_results": results,
            "total_execution_time": sum([r["execution_time"] for r in results]),
            "average_execution_time": statistics.mean([r["execution_time"] for r in results]),
            "total_tokens_estimated": total_tokens,
            "tasks_per_minute": len(results) / (sum([r["execution_time"] for r in results]) / 60),
            "success_rate": sum([1 for r in results if r["success"]]) / len(results)
        }
    
    async def _run_error_handling_benchmarks(self) -> Dict[str, Any]:
        """Test error handling and recovery capabilities."""
        error_scenarios = [
            {
                "name": "Invalid API Response",
                "goal": "This is a test goal with intentionally invalid parameters: %%INVALID%%",
                "should_recover": True
            },
            {
                "name": "Complex Ambiguous Query",
                "goal": "What is the meaning of life, universe, and everything considering quantum mechanics and philosophy?",
                "should_recover": True
            },
            {
                "name": "Empty Query",
                "goal": "",
                "should_recover": False
            }
        ]
        
        results = []
        for scenario in error_scenarios:
            try:
                start_time = time.time()
                result = await self.coordinator.process_goal(scenario["goal"])
                execution_time = time.time() - start_time
                
                recovered = result.get("status") == "completed"
                expected_recovery = scenario["should_recover"]
                
                results.append({
                    "name": scenario["name"],
                    "recovered": recovered,
                    "expected_recovery": expected_recovery,
                    "correct_handling": recovered == expected_recovery,
                    "execution_time": execution_time,
                    "error_details": result.get("error", "No error")
                })
                
            except Exception as e:
                results.append({
                    "name": scenario["name"],
                    "recovered": False,
                    "expected_recovery": scenario["should_recover"],
                    "correct_handling": not scenario["should_recover"],
                    "execution_time": 0,
                    "error_details": str(e)
                })
        
        return {
            "individual_results": results,
            "correct_handling_rate": sum([1 for r in results if r["correct_handling"]]) / len(results),
            "recovery_rate": sum([1 for r in results if r["recovered"]]) / len(results)
        }
    
    async def _run_scalability_benchmarks(self) -> Dict[str, Any]:
        """Test system scalability with concurrent requests."""
        concurrent_levels = [1, 3, 5, 8]
        results = []
        
        test_goal = "Explain the process of photosynthesis in plants"
        
        for concurrency in concurrent_levels:
            tasks = [self.coordinator.process_goal(f"{test_goal} (request {i})") 
                    for i in range(concurrency)]
            
            start_time = time.time()
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful = sum([1 for r in completed_results 
                            if not isinstance(r, Exception) and r.get("status") == "completed"])
            
            results.append({
                "concurrency_level": concurrency,
                "total_time": total_time,
                "successful_requests": successful,
                "success_rate": successful / concurrency,
                "requests_per_second": concurrency / total_time,
                "average_time_per_request": total_time / concurrency
            })
        
        return {
            "scalability_results": results,
            "max_successful_concurrency": max([r["concurrency_level"] for r in results if r["success_rate"] > 0.8]),
            "best_throughput": max([r["requests_per_second"] for r in results])
        }
    
    def _calculate_accuracy_score(self, result: str, expected_patterns: List[str]) -> float:
        """Calculate accuracy score based on expected patterns in result."""
        if not result:
            return 0.0
        
        result_lower = result.lower()
        matches = sum([1 for pattern in expected_patterns if pattern.lower() in result_lower])
        return matches / len(expected_patterns)
    
    def _extract_confidence_from_result(self, result: Dict[str, Any]) -> float:
        """Extract confidence score from result (simplified)."""
        # This would typically parse confidence from the result
        return 0.8 if result.get("status") == "completed" else 0.2
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage (simplified)."""
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except:
            return 0
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        return {
            "overall_success_rate": statistics.mean([
                results["reasoning_benchmarks"]["success_rate"],
                results["efficiency_benchmarks"]["success_rate"],
                results["error_handling_benchmarks"]["correct_handling_rate"]
            ]),
            "average_response_time": statistics.mean([
                results["reasoning_benchmarks"]["average_time"],
                results["efficiency_benchmarks"]["average_execution_time"]
            ]),
            "reasoning_accuracy": results["reasoning_benchmarks"]["average_accuracy"],
            "error_handling_score": results["error_handling_benchmarks"]["correct_handling_rate"],
            "max_throughput": results["scalability_benchmarks"]["best_throughput"]
        }

import asyncio
import sys
import json
from typing import Optional

from .config import settings
from .utils.logging import setup_logging, logger
from .llm_client import LLMClient
from .memory.knowledge_store import MemoryStore
from .coordinator import CoordinatorAgent
from .benchmarking import BenchmarkSuite

class MiniAI:
    """Main MiniAI application class."""
    
    def __init__(self):
        self.llm_client = None
        self.memory_store = None
        self.coordinator = None
        self.benchmark_suite = None
    
    async def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing MiniAI system")
        
        try:
            # Initialize LLM client
            self.llm_client = LLMClient()
            logger.info("LLM client initialized")
            
            # Initialize memory store
            self.memory_store = MemoryStore()
            await self.memory_store.initialize()
            logger.info("Memory store initialized")
            
            # Initialize coordinator
            self.coordinator = CoordinatorAgent(self.llm_client, self.memory_store)
            logger.info("Coordinator agent initialized")
            
            # Initialize benchmark suite
            self.benchmark_suite = BenchmarkSuite(self.coordinator)
            logger.info("Benchmark suite initialized")
            
            logger.info("MiniAI system fully initialized")
            
        except Exception as e:
            logger.error("Failed to initialize MiniAI system", error=str(e))
            raise
    
    async def process_goal(self, goal: str, priority: int = 5) -> dict:
        """Process a user goal through the multi-agent system."""
        if not self.coordinator:
            raise RuntimeError("MiniAI system not initialized")
        
        return await self.coordinator.process_goal(goal, priority)
    
    async def run_benchmarks(self) -> dict:
        """Run comprehensive system benchmarks."""
        if not self.benchmark_suite:
            raise RuntimeError("MiniAI system not initialized")
        
        return await self.benchmark_suite.run_full_benchmark()
    
    async def shutdown(self):
        """Gracefully shutdown all components."""
        logger.info("Shutting down MiniAI system")
        
        if self.memory_store:
            await self.memory_store.cleanup()
        
        logger.info("MiniAI system shutdown complete")

async def main():
    """Main application entry point."""
    setup_logging()
    
    # Initialize MiniAI system
    miniai = MiniAI()
    
    try:
        await miniai.initialize()
        
        # Example usage
        if len(sys.argv) > 1:
            goal = " ".join(sys.argv[1:])
            logger.info("Processing user goal", goal=goal)
            
            result = await miniai.process_goal(goal)
            
            print("\n" + "="*60)
            print("MINIAI RESULT")
            print("="*60)
            print(json.dumps(result, indent=2, default=str))
            print("="*60)
        
        else:
            # Interactive mode
            print("MiniAI System Ready!")
            print("Enter goals to process, or 'benchmark' to run tests, or 'quit' to exit.")
            
            while True:
                try:
                    user_input = input("\nGoal: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif user_input.lower() == 'benchmark':
                        print("Running benchmarks...")
                        benchmark_results = await miniai.run_benchmarks()
                        print(json.dumps(benchmark_results, indent=2, default=str))
                    elif user_input:
                        result = await miniai.process_goal(user_input)
                        print(f"\nResult: {result.get('result', 'No result')}")
                        print(f"Status: {result.get('status', 'Unknown')}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error("Error processing goal", error=str(e))
                    print(f"Error: {str(e)}")
    
    finally:
        await miniai.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
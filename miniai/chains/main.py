"""
Main entry point and CLI for MiniAI
"""

import asyncio
import typer
import json
from pathlib import Path
from typing import Optional
from .. import config
from chains import controller
from ..observability import get_metrics, InMemoryMetrics, init_metrics

app = typer.Typer()


@app.command()
def run(
    objective: str = typer.Argument(..., help="The main objective for the agent"),
    initial_task: str = typer.Argument(..., help="The first task to execute"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iter", "-n", help="Maximum iterations"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output results to file")
):
    """Run the MiniAI agent with the given objective and initial task"""
    asyncio.run(run_agent(objective, initial_task, config_file, max_iterations, output_file))


@app.command()
def benchmark(
    scenario_file: Path = typer.Argument(..., help="Path to benchmark scenario JSON file"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output benchmark results")
):
    """Run benchmark scenarios"""
    asyncio.run(run_benchmark(scenario_file, output_file))


async def run_agent(
    objective: str,
    initial_task: str,
    config_file: Optional[Path] = None,
    max_iterations: Optional[int] = None,
    output_file: Optional[Path] = None
):
    """Run the agent asynchronously"""
    
    # Initialize metrics
    init_metrics(InMemoryMetrics())
    
    # Load config
    if config_file and config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
        config = Config(**config_data)
    else:
        config = Config.from_env()
    
    if max_iterations:
        config.max_iterations = max_iterations
    
    # Create and initialize controller
    controller = AgentController(config)
    await controller.initialize()
    
    try:
        # Add initial task
        controller.add_initial_task(initial_task)
        
        # Run agent
        results = await controller.run(objective)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"AGENT RUN COMPLETED")
        print(f"{'='*50}")
        print(f"Objective: {objective}")
        print(f"Iterations: {results['iterations']}")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Tasks completed: {results['tasks_completed']}")
        print(f"Tasks remaining: {results['tasks_remaining']}")
        print(f"Success rate: {results['successful_tasks']}/{results['tasks_completed']}")
        
        # Print metrics summary
        metrics_summary = get_metrics().get_summary()
        print(f"\nMETRICS SUMMARY:")
        print(json.dumps(metrics_summary, indent=2))
        
        # Save to file if requested
        if output_file:
            output_data = {
                "results": results,
                "metrics": metrics_summary
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        
    finally:
        await controller.close()


async def run_benchmark(scenario_file: Path, output_file: Optional[Path] = None):
    """Run benchmark scenarios"""
    
    if not scenario_file.exists():
        print(f"Scenario file not found: {scenario_file}")
        return
    
    with open(scenario_file) as f:
        scenarios = json.load(f)
    
    benchmark_results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nRunning scenario {i+1}/{len(scenarios)}: {scenario['name']}")
        
        # Initialize fresh metrics for each scenario
        init_metrics(InMemoryMetrics())
        
        config = Config(**scenario.get('config', {}))
        controller = AgentController(config)
        await controller.initialize()
        
        try:
            # Add initial task
            controller.add_initial_task(scenario['initial_task'])
            
            # Run scenario
            results = await controller.run(
                scenario['objective'], 
                scenario.get('max_iterations', 10)
            )
            
            # Collect metrics
            metrics = get_metrics().get_summary()
            
            benchmark_results.append({
                "scenario_name": scenario['name'],
                "results": results,
                "metrics": metrics
            })
            
            print(f"Scenario completed: {results['tasks_completed']} tasks in {results['total_time']:.2f}s")
            
        finally:
            await controller.close()
    
    # Print benchmark summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in benchmark_results:
        print(f"\nScenario: {result['scenario_name']}")
        print(f"  Tasks completed: {result['results']['tasks_completed']}")
        print(f"  Total time: {result['results']['total_time']:.2f}s")
        print(f"  Success rate: {result['results']['successful_tasks']}/{result['results']['tasks_completed']}")
    
    # Save benchmark results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        print(f"\nBenchmark results saved to: {output_file}")


if __name__ == "__main__":
    app()
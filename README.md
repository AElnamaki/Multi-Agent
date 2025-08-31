## MiniAI Multi-Agent System README

Table of Contents

Overview
Features
Requirements
Installation
Configuration
Usage
Testing
Benchmarking
Contributing
License

## Overview
The MiniAI Multi-Agent System is a sophisticated AI framework designed to process complex goals through coordinated multi-agent workflows. This system leverages specialized agents for research, reasoning, planning, execution, and evaluation, all working together to achieve comprehensive task completion.
Features

Multi-Agent Architecture: Specialized agents for different tasks (research, reasoning, planning, execution, evaluation)
LLM Integration: Support for multiple LLM providers (OpenAI, Anthropic) with retry logic and failover
Memory System: Hybrid memory architecture with vector database (ChromaDB) and key-value store (Redis)
Benchmarking: Comprehensive benchmarking suite for performance evaluation
Structured Logging: JSON-based logging for better log analysis

## Requirements

Python 3.9+
Redis
ChromaDB
OpenAI API key (optional)
Anthropic API key (optional)

## Installation

Clone the repository: git clone https://github.com/AElnamaki/miniai.git
Create a virtual environment: python -m venv venv
Activate the virtual environment: source venv/bin/activate
Install dependencies: pip install -r requirements.txt

## Configuration

Create a .env file based on .env.example
Configure API keys, database settings, and system parameters

## Usage
Command Line Interface

Process a goal: python src/main.py "Your goal description"
Run benchmarks: python src/main.py benchmark

## Interactive Mode

Start interactive mode: python src/main.py
Enter goals to process or type benchmark to run tests

## Testing

Run unit tests: pytest tests/

## Benchmarking

Run comprehensive benchmarks: python src/main.py benchmark
View benchmark results in JSON format

## Contributing

Fork the repository
Create a feature branch
Submit a pull request with detailed changes

## License
This project is licensed under the MIT License. See LICENSE for details

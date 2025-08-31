# MiniAI Multi-Agent System

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)

  * [Command Line Interface](#command-line-interface)
  * [Interactive Mode](#interactive-mode)
* [Testing](#testing)
* [Benchmarking](#benchmarking)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

The **MiniAI Multi-Agent System** is a modular AI framework designed to process complex goals through coordinated multi-agent workflows. It leverages specialized agents for research, reasoning, planning, execution, and evaluation—working together to achieve robust task completion.

---

## Features

* **Multi-Agent Architecture**: Specialized agents for research, reasoning, planning, execution, and evaluation.
* **LLM Integration**: Support for multiple LLM providers (OpenAI, Anthropic) with retry logic and failover mechanisms.
* **Memory System**: Hybrid memory architecture combining a vector database (ChromaDB) with a key-value store (Redis).
* **Benchmarking Suite**: Tools for comprehensive performance evaluation.
* **Structured Logging**: JSON-based logs for improved monitoring and analysis.

---

## Requirements

* Python 3.9+
* Redis
* ChromaDB
* OpenAI API key *(optional)*
* Anthropic API key *(optional)*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AElnamaki/miniai.git

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

1. Create a `.env` file based on `.env.example`.
2. Configure API keys, database settings, and system parameters.

---

## Usage

### Command Line Interface

```bash
# Process a goal
python src/main.py "Your goal description"

# Run benchmarks
python src/main.py benchmark
```

### Interactive Mode

```bash
# Start interactive mode
python src/main.py
```

Enter goals to process, or type `benchmark` to run tests.

---

## Testing

```bash
pytest tests/
```

---

## Benchmarking

```bash
python src/main.py benchmark
```

Benchmark results are output in **JSON format**.

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed changes.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

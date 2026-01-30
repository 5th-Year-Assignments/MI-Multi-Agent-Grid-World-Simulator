# Multi-Agent Grid World Simulator

An integrated simulation framework designed for exploring multi-agent pathfinding strategies, coordination mechanisms, and decision-making behaviors within dynamic grid-based environments.

## Project Overview

This repository contains the implementation of **Project #2** for the Machine Intelligence course—a Multi-Agent Grid World Simulator. The framework showcases:

- Various pathfinding approaches (BFS, DFS, A*, and Reinforcement Learning)
- Multi-agent coordination under both competitive and collaborative settings
- Stochastic event modeling for realistic uncertainty
- Evaluation of agent performance with built-in visualization tools

## Features

### Environment

- Adjustable grid layout featuring obstacles, goal cells, and reward locations
- Probabilistic movement events that introduce environmental uncertainty
- Built-in collision detection and resolution logic

### Agent Types

1. **BFS Agent** — Uses breadth-first search to find paths with minimal length
2. **DFS Agent** — Employs depth-first search for memory-efficient exploration
3. **A* Agent** — Combines heuristics with search for optimal path discovery
4. **RL Agent** — Learns via Q-learning with an epsilon-greedy exploration policy

### Coordination Modes

- **Competitive** — Agents pursue individual goals and contend for limited resources
- **Collaborative** — Agents work together toward common objectives

### Performance Metrics

- Goal attainment rate
- Efficiency (rewards earned per step)
- Coordination score (measuring conflict reduction)
- Cumulative steps and total rewards

## Installation

### Requirements

- Python 3.7 or higher
- NumPy
- Matplotlib

### Setup

```bash
# Install required packages
pip install -r requirements.txt
```

## Usage

### Running the Demo

**Standard mode** (generates static visualizations):

```bash
python main.py
```

The script executes three phases:
1. A competitive multi-agent scenario
2. A collaborative multi-agent scenario
3. An algorithm comparison experiment

**Live visualization mode:**

```bash
python live_demo.py
```

Observe agent behavior as it unfolds. The display refreshes continuously during the run, showing:
- Step-by-step agent trajectories
- Reward collection events
- Goal completion
- Live grid state updates

You can also enable live visualization from the main script by responding `y` when asked.

### Custom Simulations

```python
from grid_world import GridWorld, Position
from agents import AStarAgent
from multi_agent_simulator import MultiAgentSimulator

# Set up the environment
grid_world = GridWorld(width=20, height=20, obstacle_density=0.15)

# Initialize the simulator
simulator = MultiAgentSimulator(grid_world, coordination_mode="competitive")

# Register an agent
agent = AStarAgent(0, grid_world)
simulator.add_agent(agent, Position(0, 0), Position(19, 19))

# Execute the simulation
metrics = simulator.run(max_steps=500)

# Inspect the outcome
print(f"Success: {metrics['overall']['success_rate']}")
```

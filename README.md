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

### Visualization

```python
from visualization import GridWorldVisualizer

# Instantiate the visualizer
visualizer = GridWorldVisualizer(grid_world, simulator)

# Render the current grid state
visualizer.visualize_state(save_path="grid_state.png")

# Generate metric plots
visualizer.plot_metrics(metrics, save_path="metrics.png")

# Produce an animation (requires step_history)
visualizer.animate_simulation(save_path="animation.gif")
```

## Project Structure

```
MI-Multi-Agent-Grid-World-Simulator/
├── grid_world.py              # Grid environment and state management
├── agents.py                  # Agent implementations (BFS, DFS, A*, RL)
├── multi_agent_simulator.py   # Simulation logic and coordination
├── visualization.py           # Plotting and visualization utilities
├── main.py                    # Primary demo entry point
├── live_demo.py               # Demo with live visualization
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation
└── REPORT.md                  # Project report (3–4 pages)
```

## Key Components

### GridWorld (`grid_world.py`)

- Maintains grid state, obstacles, goals, and reward cells
- Processes agent movements and resolves collisions
- Integrates stochastic event handling

### Agents (`agents.py`)

- Abstract `Agent` base class with shared methods
- `BFSAgent`, `DFSAgent`, and `AStarAgent` for pathfinding
- `RLAgent` for reinforcement learning

### MultiAgentSimulator (`multi_agent_simulator.py`)

- Orchestrates multiple agents in a shared environment
- Supports competitive and collaborative coordination
- Aggregates and computes performance metrics

### Visualization (`visualization.py`)

- Static grid rendering
- Metric visualization and charts
- Animation generation

## Configuration

You can customize the grid world using these parameters:

```python
GridWorld(
    width=20,              # Horizontal grid size
    height=20,             # Vertical grid size
    obstacle_density=0.15, # Obstacle probability per cell
    num_goals=3,           # Number of goal positions
    num_rewards=5,         # Number of reward cells
    stochastic_prob=0.1    # Chance of stochastic movement
)
```

## Results

Running a simulation produces:

- Terminal output with detailed performance statistics
- `grid_world_final.png` — snapshot of the final grid state
- `performance_metrics.png` — comparative performance plots

## Deliverables

✅ Grid-world simulation implementation  
✅ Pathfinding agents (BFS, DFS, A*, RL)  
✅ Multi-agent coordination logic  
✅ Performance evaluation framework  
✅ Agent movement visualization  
✅ Written report (3–4 pages)

## License

Developed for educational use as part of the Machine Intelligence course.

## Authors

| Group Member    | ID          |
|-----------------|-------------|
| Heran Eshetu    | UGR/5016/14 |
| Iman Ibrahim    | UGR/1004/14 |
| Ruhama Yohannes | UGR/7382/14 |
| Samrawit Kahsay | UGR/2271/14 |
| Yordanos Melaku | UGR/8211/14 |

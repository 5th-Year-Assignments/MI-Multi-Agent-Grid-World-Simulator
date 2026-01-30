# Multi-Agent Grid World Simulator

A comprehensive simulation system for studying multi-agent pathfinding, coordination, and decision-making in dynamic grid environments.

## Project Overview

This project implements **Project #2** from the Machine Intelligence course: Multi-Agent Grid World Simulator. The system demonstrates:

- Multiple pathfinding algorithms (BFS, DFS, A*, Reinforcement Learning)
- Competitive and collaborative multi-agent coordination
- Stochastic events for uncertainty simulation
- Performance evaluation and visualization

## Features

### Environment
- Configurable grid world with obstacles, goals, and rewards
- Stochastic movement events (simulates uncertainty)
- Collision detection and handling

### Agent Types
1. **BFS Agent**: Breadth-first search for optimal path length
2. **DFS Agent**: Depth-first search for memory efficiency
3. **A* Agent**: Heuristic-based optimal pathfinding
4. **RL Agent**: Q-learning with epsilon-greedy exploration

### Coordination Modes
- **Competitive**: Agents compete for goals and resources
- **Collaborative**: Agents coordinate for shared objectives

### Performance Metrics
- Success rate (goal achievement)
- Efficiency (reward per step)
- Coordination score (conflict avoidance)
- Total steps and rewards

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Demo

**Standard Mode (with static visualizations):**
```bash
python main.py
```

This will run:
1. Competitive multi-agent simulation
2. Collaborative multi-agent simulation
3. Algorithm comparison study

**Live Visualization Mode:**
```bash
python live_demo.py
```

Watch agents move in real-time! The visualization window updates as the simulation runs. You can see:
- Agents moving step-by-step
- Rewards being collected
- Goals being reached
- Real-time grid updates

The main script also supports live visualization - just answer 'y' when prompted.

### Custom Simulations

```python
from grid_world import GridWorld, Position
from agents import AStarAgent
from multi_agent_simulator import MultiAgentSimulator

# Create environment
grid_world = GridWorld(width=20, height=20, obstacle_density=0.15)

# Create simulator
simulator = MultiAgentSimulator(grid_world, coordination_mode="competitive")

# Add agents
agent = AStarAgent(0, grid_world)
simulator.add_agent(agent, Position(0, 0), Position(19, 19))

# Run simulation
metrics = simulator.run(max_steps=500)

# View results
print(f"Success: {metrics['overall']['success_rate']}")
```

### Visualization

```python
from visualization import GridWorldVisualizer

# Create visualizer
visualizer = GridWorldVisualizer(grid_world, simulator)

# Visualize current state
visualizer.visualize_state(save_path="grid_state.png")

# Plot performance metrics
visualizer.plot_metrics(metrics, save_path="metrics.png")

# Animate simulation (requires step_history)
visualizer.animate_simulation(save_path="animation.gif")
```

## Project Structure

```
mi/
├── grid_world.py              # Grid environment implementation
├── agents.py                  # Agent algorithms (BFS, DFS, A*, RL)
├── multi_agent_simulator.py   # Simulation engine and coordination
├── visualization.py           # Visualization and plotting
├── main.py                    # Main demo script
├── live_demo.py               # Live visualization demo
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── REPORT.md                  # Project report (3-4 pages)
```

## Key Components

### GridWorld (`grid_world.py`)
- Manages grid state, obstacles, goals, rewards
- Handles agent movement and collisions
- Implements stochastic events

### Agents (`agents.py`)
- Base `Agent` class with common functionality
- `BFSAgent`, `DFSAgent`, `AStarAgent` for pathfinding
- `RLAgent` for reinforcement learning

### MultiAgentSimulator (`multi_agent_simulator.py`)
- Coordinates multiple agents
- Handles competitive/collaborative modes
- Calculates performance metrics

### Visualization (`visualization.py`)
- Static grid visualization
- Performance metric plots
- Animation support

## Configuration

Key parameters can be adjusted:

```python
GridWorld(
    width=20,              # Grid width
    height=20,             # Grid height
    obstacle_density=0.15, # Probability of obstacle per cell
    num_goals=3,           # Number of goal cells
    num_rewards=5,         # Number of reward cells
    stochastic_prob=0.1    # Probability of stochastic movement
)
```

## Results

The simulation generates:
- Console output with performance metrics
- `grid_world_final.png`: Final state visualization
- `performance_metrics.png`: Performance comparison charts

## Deliverables

✅ Grid-world simulation code  
✅ Agent algorithms (BFS, DFS, A*, RL)  
✅ Multi-agent coordination  
✅ Performance evaluation  
✅ Visualization of agent movements  
✅ Written report (3-4 pages)

## License

This project is created for educational purposes as part of the Machine Intelligence course.

## Authors

Heran Eshetu, Iman Ibrahim, Ruhama Yohannes, Samrawit Kahsay, Yordanos Melaku

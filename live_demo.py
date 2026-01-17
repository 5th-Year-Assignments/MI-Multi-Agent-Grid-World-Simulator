"""
Simple demo script for live visualization
Run this to see agents move in real-time!
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from grid_world import GridWorld, Position
from agents import BFSAgent, DFSAgent, AStarAgent, RLAgent
from multi_agent_simulator import MultiAgentSimulator
from visualization import GridWorldVisualizer


def create_random_position(grid_world: GridWorld) -> Position:
    """Create a random valid position"""
    while True:
        x = random.randint(0, grid_world.width - 1)
        y = random.randint(0, grid_world.height - 1)
        pos = Position(x, y)
        if grid_world.is_valid_position(pos):
            return pos


def main():
    """Run live visualization demo"""
    print("=" * 60)
    print("LIVE MULTI-AGENT GRID WORLD SIMULATOR")
    print("=" * 60)
    print("\nWatch the agents move in real-time!")
    print("Close the visualization window when done.\n")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Create grid world
    grid_world = GridWorld(
        width=20, 
        height=20, 
        obstacle_density=0.15,
        num_goals=3,
        num_rewards=8,
        stochastic_prob=0.1
    )
    
    # Create visualizer with live mode enabled
    visualizer = GridWorldVisualizer(grid_world, None, live_mode=True)
    
    # Create simulator with visualizer
    simulator = MultiAgentSimulator(grid_world, coordination_mode="competitive", 
                                   visualizer=visualizer)
    
    # Update visualizer reference
    visualizer.simulator = simulator
    
    # Add different types of agents
    agents = [
        BFSAgent(0, grid_world),
        DFSAgent(1, grid_world),
        AStarAgent(2, grid_world),
        RLAgent(3, grid_world, learning_rate=0.1, discount=0.95, epsilon=0.2)
    ]
    
    # Place agents at random positions
    print("Initializing agents...")
    for agent in agents:
        start_pos = create_random_position(grid_world)
        goal = random.choice(grid_world.goals) if grid_world.goals else None
        simulator.add_agent(agent, start_pos, goal)
        print(f"Agent {agent.agent_id} ({type(agent).__name__}) "
              f"started at ({start_pos.x}, {start_pos.y}), "
              f"goal at ({goal.x}, {goal.y})")
    
    # Show initial state
    print("\nStarting simulation with live visualization...")
    print("A window will open showing the grid world. Watch the agents move!")
    visualizer.update_live()
    plt.show(block=False)  # Show window without blocking
    
    # Run simulation with live updates
    metrics = simulator.run(max_steps=500, step_delay=0.1)
    
    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Success Rate: {metrics['overall']['success_rate']:.2%}")
    print(f"Total Steps: {metrics['overall']['total_steps']}")
    print(f"Average Reward: {metrics['overall']['average_reward']:.2f}")
    print(f"Coordination Score: {metrics['overall']['coordination_score']:.2f}")
    
    print("\nAgent Performance:")
    for agent_id, agent_metrics in metrics['agent_metrics'].items():
        agent = simulator.agents[agent_id]
        print(f"\nAgent {agent_id} ({type(agent).__name__}):")
        print(f"  Reached Goal: {agent_metrics['reached_goal']}")
        print(f"  Steps Taken: {agent_metrics['steps_taken']}")
        print(f"  Total Reward: {agent_metrics['total_reward']:.2f}")
        print(f"  Efficiency: {agent_metrics['efficiency']:.2f}")
    
    # Save final visualization
    print("\nSaving final state...")
    visualizer.visualize_state(save_path="grid_world_final.png")
    visualizer.plot_metrics(metrics, save_path="performance_metrics.png")
    
    print("\nSimulation complete! Close the visualization window to exit.")
    
    # Keep window open
    try:
        input("\nPress Enter to close...")
    except:
        pass
    
    visualizer.close()


if __name__ == "__main__":
    main()


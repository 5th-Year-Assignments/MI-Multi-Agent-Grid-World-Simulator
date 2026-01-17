"""
Main script for Multi-Agent Grid World Simulator
Demonstrates different agent types and coordination modes
"""

import random
import numpy as np
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


def run_competitive_simulation(live_visualization: bool = False):
    """Run simulation with competitive agents"""
    print("=" * 60)
    print("COMPETITIVE MULTI-AGENT SIMULATION")
    print("=" * 60)
    
    # Create grid world
    grid_world = GridWorld(
        width=20, 
        height=20, 
        obstacle_density=0.15,
        num_goals=3,
        num_rewards=8,
        stochastic_prob=0.1
    )
    
    # Create visualizer if live mode
    visualizer = None
    if live_visualization:
        visualizer = GridWorldVisualizer(grid_world, None, live_mode=True)
    
    # Create simulator
    simulator = MultiAgentSimulator(grid_world, coordination_mode="competitive", 
                                   visualizer=visualizer)
    
    # Update visualizer reference
    if visualizer:
        visualizer.simulator = simulator
    
    # Add different types of agents
    agents = [
        BFSAgent(0, grid_world),
        DFSAgent(1, grid_world),
        AStarAgent(2, grid_world),
        RLAgent(3, grid_world, learning_rate=0.1, discount=0.95, epsilon=0.2)
    ]
    
    # Place agents at random positions
    for agent in agents:
        start_pos = create_random_position(grid_world)
        # Assign random goal
        goal = random.choice(grid_world.goals) if grid_world.goals else None
        simulator.add_agent(agent, start_pos, goal)
        print(f"Agent {agent.agent_id} ({type(agent).__name__}) "
              f"started at ({start_pos.x}, {start_pos.y}), "
              f"goal at ({goal.x}, {goal.y})")
    
    # Run simulation
    print("\nRunning simulation...")
    metrics = simulator.run(max_steps=500)
    
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
    
    # Visualize (if not already in live mode)
    if not live_visualization:
        visualizer = GridWorldVisualizer(grid_world, simulator)
        print("\nGenerating visualization...")
        visualizer.visualize_state(save_path="grid_world_final.png")
        visualizer.plot_metrics(metrics, save_path="performance_metrics.png")
    else:
        # Save final state and metrics
        print("\nSaving final visualization...")
        visualizer.visualize_state(save_path="grid_world_final.png")
        visualizer.plot_metrics(metrics, save_path="performance_metrics.png")
    
    return simulator, metrics


def run_collaborative_simulation():
    """Run simulation with collaborative agents sharing goals"""
    print("\n" + "=" * 60)
    print("COLLABORATIVE MULTI-AGENT SIMULATION")
    print("=" * 60)
    
    # Create grid world
    grid_world = GridWorld(
        width=20, 
        height=20, 
        obstacle_density=0.15,
        num_goals=2,  # Fewer goals to encourage collaboration
        num_rewards=10,
        stochastic_prob=0.1
    )
    
    # Create simulator
    simulator = MultiAgentSimulator(grid_world, coordination_mode="collaborative")
    
    # Add agents with shared goals
    agents = [
        AStarAgent(0, grid_world),
        AStarAgent(1, grid_world),
        BFSAgent(2, grid_world),
    ]
    
    # Assign shared goals
    shared_goal = grid_world.goals[0]
    
    for agent in agents:
        start_pos = create_random_position(grid_world)
        simulator.add_agent(agent, start_pos, shared_goal)
        print(f"Agent {agent.agent_id} ({type(agent).__name__}) "
              f"started at ({start_pos.x}, {start_pos.y}), "
              f"shared goal at ({shared_goal.x}, {shared_goal.y})")
    
    # Run simulation
    print("\nRunning simulation...")
    metrics = simulator.run(max_steps=500)
    
    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Success Rate: {metrics['overall']['success_rate']:.2%}")
    print(f"Total Steps: {metrics['overall']['total_steps']}")
    print(f"Average Reward: {metrics['overall']['average_reward']:.2f}")
    print(f"Coordination Score: {metrics['overall']['coordination_score']:.2f}")
    
    return simulator, metrics


def compare_algorithms():
    """Compare different pathfinding algorithms"""
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    
    results = {}
    
    for alg_name, AgentClass in [
        ("BFS", BFSAgent),
        ("DFS", DFSAgent),
        ("A*", AStarAgent),
        ("RL", RLAgent)
    ]:
        print(f"\nTesting {alg_name}...")
        
        # Create fresh environment
        grid_world = GridWorld(width=15, height=15, obstacle_density=0.2)
        simulator = MultiAgentSimulator(grid_world, coordination_mode="competitive")
        
        # Add single agent
        agent = AgentClass(0, grid_world)
        start_pos = create_random_position(grid_world)
        goal = random.choice(grid_world.goals) if grid_world.goals else None
        simulator.add_agent(agent, start_pos, goal)
        
        # Run simulation
        metrics = simulator.run(max_steps=300)
        
        results[alg_name] = {
            'success': metrics['agent_metrics'][0]['reached_goal'],
            'steps': metrics['agent_metrics'][0]['steps_taken'],
            'reward': metrics['agent_metrics'][0]['total_reward'],
            'efficiency': metrics['agent_metrics'][0]['efficiency']
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<10} {'Success':<10} {'Steps':<10} {'Reward':<10} {'Efficiency':<10}")
    print("-" * 60)
    for alg_name, result in results.items():
        success_str = "Yes" if result['success'] else "No"
        print(f"{alg_name:<10} {success_str:<10} {result['steps']:<10} "
              f"{result['reward']:<10.2f} {result['efficiency']:<10.2f}")


def main():
    """Main function to run demonstrations"""
    print("\n" + "=" * 60)
    print("MULTI-AGENT GRID WORLD SIMULATOR")
    print("=" * 60)
    print("\nThis simulator demonstrates:")
    print("1. Multiple agent types (BFS, DFS, A*, RL)")
    print("2. Competitive and collaborative coordination")
    print("3. Stochastic events and uncertainty")
    print("4. Performance evaluation and visualization")
    
    # Ask user if they want live visualization
    print("\n" + "=" * 60)
    use_live = input("Enable live visualization? (y/n): ").lower().strip() == 'y'
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run competitive simulation
    competitive_sim, competitive_metrics = run_competitive_simulation(live_visualization=use_live)
    
    # Run collaborative simulation
    collaborative_sim, collaborative_metrics = run_collaborative_simulation()
    
    # Compare algorithms
    compare_algorithms()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nVisualizations saved:")
    print("  - grid_world_final.png")
    print("  - performance_metrics.png")


if __name__ == "__main__":
    main()


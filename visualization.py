"""
Visualization module for grid world simulation
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from grid_world import GridWorld, CellType
from multi_agent_simulator import MultiAgentSimulator


class GridWorldVisualizer:
    """Visualizes grid world and agent movements"""
    
    def __init__(self, grid_world: GridWorld, simulator: MultiAgentSimulator, live_mode: bool = False):
        self.grid_world = grid_world
        self.simulator = simulator
        self.fig = None
        self.ax = None
        self.live_mode = live_mode
        self.colors = {
            CellType.EMPTY: 'white',
            CellType.OBSTACLE: 'black',
            CellType.GOAL: 'green',
            CellType.AGENT: 'red',
            CellType.REWARD: 'yellow'
        }
        self.agent_colors = ['red', 'blue', 'orange', 'purple', 'cyan', 'magenta']
        
        if live_mode:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
            self.fig.show()  # Ensure window is displayed
    
    def visualize_state(self, step_data: dict = None, save_path: str = None):
        """Visualize current state of the grid"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
        
        self.ax.clear()
        
        # Draw grid
        state = self.grid_world.get_state()
        height, width = state.shape
        
        # Create color map
        color_map = np.zeros((height, width, 3))
        
        for y in range(height):
            for x in range(width):
                cell_type = CellType(state[y][x])
                if cell_type == CellType.OBSTACLE:
                    color_map[y][x] = [0, 0, 0]  # Black
                elif cell_type == CellType.GOAL:
                    color_map[y][x] = [0, 1, 0]  # Green
                elif cell_type == CellType.REWARD:
                    color_map[y][x] = [1, 1, 0]  # Yellow
                else:
                    color_map[y][x] = [1, 1, 1]  # White
        
        # Draw agents
        agent_positions = {}
        if step_data:
            agent_positions = step_data.get('agent_positions', {})
        else:
            agent_positions = self.grid_world.agents
        
        for idx, (agent_id, pos) in enumerate(agent_positions.items()):
            color_idx = agent_id % len(self.agent_colors)
            color_map[pos.y][pos.x] = plt.cm.tab10(color_idx)[:3]
        
        # Draw goals
        for goal in self.grid_world.goals:
            if goal not in agent_positions.values():
                color_map[goal.y][goal.x] = [0, 1, 0]  # Green
        
        # Draw rewards (only if not occupied by agents)
        for reward_pos in self.grid_world.rewards.keys():
            if reward_pos not in agent_positions.values():
                color_map[reward_pos.y][reward_pos.x] = [1, 1, 0]  # Yellow
        
        self.ax.imshow(color_map, origin='upper', interpolation='nearest')
        self.ax.set_title(f'Grid World - Step {self.grid_world.step_count}')
        
        # Add grid lines
        self.ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', label='Empty'),
            Patch(facecolor='black', label='Obstacle'),
            Patch(facecolor='green', label='Goal'),
            Patch(facecolor='yellow', label='Reward'),
        ]
        
        for idx, (agent_id, _) in enumerate(agent_positions.items()):
            color_idx = agent_id % len(self.agent_colors)
            color = plt.cm.tab10(color_idx)
            agent_type = type(self.simulator.agents[agent_id]).__name__
            legend_elements.append(
                Patch(facecolor=color, label=f'Agent {agent_id} ({agent_type})')
            )
        
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        elif self.live_mode:
            # In live mode, just update the display
            plt.draw()
            plt.pause(0.01)  # Small pause to allow GUI to update
        else:
            plt.show()
    
    def update_live(self, step_data: dict = None):
        """Update visualization in live mode"""
        if not self.live_mode:
            return
        
        self.visualize_state(step_data)
    
    def close(self):
        """Close the visualization window"""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()  # Turn off interactive mode
    
    def animate_simulation(self, interval: int = 100, save_path: str = None):
        """Animate the simulation"""
        if not self.simulator.step_history:
            print("No simulation history to animate. Run simulation first.")
            return
        
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        
        def animate(frame):
            if frame < len(self.simulator.step_history):
                step_data = self.simulator.step_history[frame]
                self.visualize_state(step_data)
        
        anim = animation.FuncAnimation(
            self.fig, animate, frames=len(self.simulator.step_history),
            interval=interval, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        else:
            plt.show()
    
    def plot_metrics(self, metrics: dict, save_path: str = None):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        agent_ids = list(metrics['agent_metrics'].keys())
        success_rates = [m['success_rate'] for m in metrics['agent_metrics'].values()]
        rewards = [m['total_reward'] for m in metrics['agent_metrics'].values()]
        steps = [m['steps_taken'] for m in metrics['agent_metrics'].values()]
        efficiencies = [m['efficiency'] for m in metrics['agent_metrics'].values()]
        
        # Plot 1: Success rates
        axes[0, 0].bar(agent_ids, success_rates, color='green', alpha=0.7)
        axes[0, 0].set_title('Success Rate by Agent')
        axes[0, 0].set_xlabel('Agent ID')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim([0, 1.1])
        
        # Plot 2: Total rewards
        axes[0, 1].bar(agent_ids, rewards, color='blue', alpha=0.7)
        axes[0, 1].set_title('Total Reward by Agent')
        axes[0, 1].set_xlabel('Agent ID')
        axes[0, 1].set_ylabel('Total Reward')
        
        # Plot 3: Steps taken
        axes[1, 0].bar(agent_ids, steps, color='orange', alpha=0.7)
        axes[1, 0].set_title('Steps Taken by Agent')
        axes[1, 0].set_xlabel('Agent ID')
        axes[1, 0].set_ylabel('Steps')
        
        # Plot 4: Efficiency (reward per step)
        axes[1, 1].bar(agent_ids, efficiencies, color='purple', alpha=0.7)
        axes[1, 1].set_title('Efficiency (Reward/Step) by Agent')
        axes[1, 1].set_xlabel('Agent ID')
        axes[1, 1].set_ylabel('Efficiency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()


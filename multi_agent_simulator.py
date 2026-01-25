"""
Multi-Agent Grid World Simulator
Handles coordination, competition, and performance evaluation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from grid_world import GridWorld, Position
from agents import Agent, BFSAgent, DFSAgent, AStarAgent, RLAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict


class MultiAgentSimulator:
    """Simulates multiple agents in a grid world"""
    
    def __init__(self, grid_world: GridWorld, coordination_mode: str = "competitive", 
                 visualizer=None):
        """
        Initialize simulator
        
        Args:
            grid_world: The grid world environment
            coordination_mode: "competitive" or "collaborative"
            visualizer: Optional visualizer for live updates
        """
        self.grid_world = grid_world
        self.coordination_mode = coordination_mode
        self.agents: Dict[int, Agent] = {}
        self.max_steps = 1000
        self.step_history: List[Dict] = []
        self.performance_metrics: Dict = defaultdict(list)
        self.visualizer = visualizer
    
    def add_agent(self, agent: Agent, start_pos: Position, goal: Optional[Position] = None):
        """Add an agent to the simulation"""
        self.grid_world.add_agent(agent.agent_id, start_pos, goal)
        self.agents[agent.agent_id] = agent
        
        # Initialize agent path
        if goal is None:
            goal = self.grid_world.agent_goals[agent.agent_id]
        
        # Initialize RL agent state
        if isinstance(agent, RLAgent):
            agent.last_state = start_pos
            agent.last_action = None
        else:
            agent.update_path(start_pos, goal)
    
    def _handle_collision(self, agent_id: int, desired_pos: Position) -> Position:
        """Handle collisions between agents"""
        # Check if position is occupied
        for aid, apos in self.grid_world.agents.items():
            if aid != agent_id and apos == desired_pos:
                # In competitive mode, agent waits
                # In collaborative mode, agents might coordinate
                if self.coordination_mode == "collaborative":
                    # Try to find alternative path
                    current_pos = self.grid_world.agents[agent_id]
                    neighbors = self.grid_world.get_neighbors(current_pos)
                    for neighbor in neighbors:
                        if neighbor != desired_pos and not any(
                            self.grid_world.agents[aid] == neighbor 
                            for aid in self.grid_world.agents.keys() 
                            if aid != agent_id
                        ):
                            return neighbor
                # Stay in place if no alternative
                return self.grid_world.agents[agent_id]
        return desired_pos
    
    def _coordinate_shared_goals(self):
        """Coordinate agents when they share goals (collaborative mode)"""
        if self.coordination_mode != "collaborative":
            return
        
        # Group agents by goal
        goal_groups: Dict[Position, List[int]] = defaultdict(list)
        for agent_id, goal in self.grid_world.agent_goals.items():
            goal_groups[goal].append(agent_id)
        
        # For shared goals, agents can coordinate approach
        for goal, agent_ids in goal_groups.items():
            if len(agent_ids) > 1:
                # Agents coordinate to approach from different directions
                positions = [self.grid_world.agents[aid] for aid in agent_ids]
                # Simple coordination: agents try to spread out
                pass  # Could implement more sophisticated coordination
    
    def step(self) -> bool:
        """
        Execute one simulation step
        
        Returns:
            True if simulation should continue, False if all agents reached goals
        """
        # Check if all agents reached goals
        all_reached = all(
            self.grid_world.agent_reached_goal(aid) 
            for aid in self.agents.keys()
        )
        if all_reached:
            return False
        
        # Coordinate if in collaborative mode
        if self.coordination_mode == "collaborative":
            self._coordinate_shared_goals()
        
        # Store step state
        step_data = {
            'step': self.grid_world.step_count,
            'agent_positions': {aid: pos for aid, pos in self.grid_world.agents.items()},
            'rewards_collected': {}
        }
        
        # Move each agent
        for agent_id, agent in self.agents.items():
            if self.grid_world.agent_reached_goal(agent_id):
                continue
            
            # Get next action
            if isinstance(agent, RLAgent):
                next_pos = agent.get_next_action()
            else:
                next_pos = agent.get_next_action()
                
                # Replan if path is exhausted or if agent position doesn't match path
                if next_pos is None:
                    current_pos = self.grid_world.agents[agent_id]
                    goal = self.grid_world.agent_goals[agent_id]
                    agent.update_path(current_pos, goal)
                    next_pos = agent.get_next_action()
                else:
                    # Check if agent is still on the expected path
                    current_pos = self.grid_world.agents[agent_id]
                    # More forgiving path recovery - check if we're anywhere in the remaining path
                    if agent.current_path_index < len(agent.path):
                        # Check if current position is anywhere in the remaining path
                        remaining_path = agent.path[agent.current_path_index:]
                        if current_pos in remaining_path:
                            # Found in path - jump to that position
                            try:
                                # Find the index in the full path
                                path_index = agent.path.index(current_pos, agent.current_path_index)
                                agent.current_path_index = path_index + 1
                                next_pos = agent.get_next_action()
                            except ValueError:
                                # Shouldn't happen since we checked, but replan if it does
                                goal = self.grid_world.agent_goals[agent_id]
                                agent.update_path(current_pos, goal)
                                next_pos = agent.get_next_action()
                        else:
                            # Not in remaining path - check if we're close to next step
                            if agent.current_path_index < len(agent.path):
                                expected_pos = agent.path[agent.current_path_index]
                                # For DFS, be more lenient - check if we're within 2 steps
                                if current_pos.distance(expected_pos) <= 2:
                                    # Close enough, try to continue
                                    agent.current_path_index += 1
                                    next_pos = agent.get_next_action()
                                else:
                                    # Too far off, replan
                                    goal = self.grid_world.agent_goals[agent_id]
                                    agent.update_path(current_pos, goal)
                                    next_pos = agent.get_next_action()
                            else:
                                # Path exhausted, replan
                                goal = self.grid_world.agent_goals[agent_id]
                                agent.update_path(current_pos, goal)
                                next_pos = agent.get_next_action()
                    else:
                        # Path exhausted, replan
                        goal = self.grid_world.agent_goals[agent_id]
                        agent.update_path(current_pos, goal)
                        next_pos = agent.get_next_action()
            
            if next_pos is None:
                continue
            
            # Handle collisions
            next_pos = self._handle_collision(agent_id, next_pos)
            
            # Move agent (pass agent type for stochastic probability adjustment)
            agent_type = type(agent).__name__
            success, reward = self.grid_world.move_agent(agent_id, next_pos, agent_type)
            
            if success:
                # Only advance path index if move succeeded
                if not isinstance(agent, RLAgent):
                    agent.advance_path()
                agent.record_move(reward)
                step_data['rewards_collected'][agent_id] = reward
                
                # Update RL agent Q-values and state
                if isinstance(agent, RLAgent):
                    current_pos = self.grid_world.agents[agent_id]
                    goal = self.grid_world.agent_goals[agent_id]
                    
                    # Only update Q-value if we have a previous state and action
                    if agent.last_state is not None and agent.last_action is not None:
                        agent.update_q_value(
                            agent.state_to_key(agent.last_state, goal),
                            agent.last_action,
                            reward,
                            current_pos,
                            goal
                        )
                    
                    # Update state for next iteration
                    agent.last_state = current_pos
                    agent.last_action = next_pos
                
                # Check if goal reached
                if self.grid_world.agent_reached_goal(agent_id):
                    agent.reached_goal = True
                    # Large reward for reaching goal
                    agent.total_reward += 100.0
        
        self.step_history.append(step_data)
        
        # Update live visualization if available
        if self.visualizer and self.visualizer.live_mode:
            self.visualizer.update_live(step_data)
        
        # Check max steps
        if self.grid_world.step_count >= self.max_steps:
            return False
        
        return True
    
    def run(self, max_steps: int = 1000, step_delay: float = 0.1) -> Dict:
        """
        Run the simulation
        
        Args:
            max_steps: Maximum number of steps
            step_delay: Delay between steps in seconds (for live visualization)
        
        Returns:
            Dictionary with performance metrics
        """
        self.max_steps = max_steps
        self.step_history = []
        
        step_count = 0
        import time
        while self.step() and step_count < max_steps:
            step_count += 1
            if self.visualizer and self.visualizer.live_mode:
                time.sleep(step_delay)  # Control visualization speed
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'agent_metrics': {},
            'overall': {}
        }
        
        total_steps = self.grid_world.step_count
        num_agents = len(self.agents)
        
        success_count = 0
        total_reward = 0.0
        total_path_length = 0
        
        for agent_id, agent in self.agents.items():
            reached_goal = agent.reached_goal
            success_count += int(reached_goal)
            
            # Calculate efficiency with goal-reaching bonus
            base_efficiency = agent.total_reward / max(agent.steps_taken, 1)
            
            # If agent reached goal, add efficiency bonus for doing it quickly
            if reached_goal and agent.steps_taken > 0:
                # Bonus inversely proportional to steps (faster = better)
                goal_efficiency_bonus = 50.0 / agent.steps_taken
                efficiency = base_efficiency + goal_efficiency_bonus
            else:
                efficiency = base_efficiency
            
            agent_metrics = {
                'reached_goal': reached_goal,
                'steps_taken': agent.steps_taken,
                'total_reward': agent.total_reward,
                'success_rate': 1.0 if reached_goal else 0.0,
                'efficiency': efficiency
            }
            
            metrics['agent_metrics'][agent_id] = agent_metrics
            total_reward += agent.total_reward
        
        # Overall metrics
        metrics['overall'] = {
            'success_rate': success_count / num_agents if num_agents > 0 else 0.0,
            'total_steps': total_steps,
            'average_reward': total_reward / num_agents if num_agents > 0 else 0.0,
            'coordination_score': self._calculate_coordination_score()
        }
        
        return metrics
    
    def _calculate_coordination_score(self) -> float:
        """Calculate coordination score based on agent interactions"""
        if len(self.step_history) == 0:
            return 0.0
        
        # Count collisions/conflicts
        conflicts = 0
        for step_data in self.step_history:
            positions = list(step_data['agent_positions'].values())
            # Check for collisions
            if len(positions) != len(set((p.x, p.y) for p in positions)):
                conflicts += 1
        
        # Coordination score: lower conflicts = higher score
        max_possible_conflicts = len(self.step_history)
        if max_possible_conflicts == 0:
            return 1.0
        
        coordination = 1.0 - (conflicts / max_possible_conflicts)
        return max(0.0, coordination)


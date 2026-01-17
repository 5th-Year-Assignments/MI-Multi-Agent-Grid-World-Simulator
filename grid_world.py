"""
Grid World Environment for Multi-Agent Simulation
Implements a grid-based environment with obstacles, goals, rewards, and stochastic events.
"""

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


class CellType(Enum):
    """Types of cells in the grid world"""
    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2
    AGENT = 3
    REWARD = 4


@dataclass
class Position:
    """Represents a position in the grid"""
    x: int
    y: int
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance(self, other: 'Position') -> float:
        """Calculate Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)


class GridWorld:
    """Grid world environment for multi-agent simulation"""
    
    def __init__(self, width: int = 20, height: int = 20, 
                 obstacle_density: float = 0.15,
                 num_goals: int = 3,
                 num_rewards: int = 5,
                 stochastic_prob: float = 0.1):
        """
        Initialize grid world
        
        Args:
            width: Grid width
            height: Grid height
            obstacle_density: Probability of obstacle in each cell
            num_goals: Number of goal cells
            num_rewards: Number of reward cells
            stochastic_prob: Probability of stochastic events
        """
        self.width = width
        self.height = height
        self.stochastic_prob = stochastic_prob
        self.grid = np.zeros((height, width), dtype=int)
        self.goals: List[Position] = []
        self.rewards: Dict[Position, float] = {}
        self.agents: Dict[int, Position] = {}  # agent_id -> position
        self.agent_goals: Dict[int, Position] = {}  # agent_id -> goal
        self.step_count = 0
        
        # Generate obstacles
        self._generate_obstacles(obstacle_density)
        
        # Generate goals
        self._generate_goals(num_goals)
        
        # Generate rewards
        self._generate_rewards(num_rewards)
    
    def _generate_obstacles(self, density: float):
        """Randomly place obstacles in the grid"""
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < density:
                    self.grid[y][x] = CellType.OBSTACLE.value
    
    def _generate_goals(self, num_goals: int):
        """Place goal cells in the grid"""
        attempts = 0
        while len(self.goals) < num_goals and attempts < 1000:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            
            if (self.grid[y][x] == CellType.EMPTY.value and 
                pos not in self.goals):
                self.goals.append(pos)
                self.grid[y][x] = CellType.GOAL.value
            attempts += 1
    
    def _generate_rewards(self, num_rewards: int):
        """Place reward cells in the grid"""
        attempts = 0
        while len(self.rewards) < num_rewards and attempts < 1000:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            
            if (self.grid[y][x] == CellType.EMPTY.value and 
                pos not in self.rewards and pos not in self.goals):
                self.rewards[pos] = random.uniform(1.0, 10.0)
            attempts += 1
    
    def add_agent(self, agent_id: int, start_pos: Position, goal: Optional[Position] = None):
        """Add an agent to the grid world"""
        if not self.is_valid_position(start_pos):
            raise ValueError(f"Invalid start position: {start_pos}")
        
        self.agents[agent_id] = start_pos
        
        # Assign goal if not provided
        if goal is None:
            goal = random.choice(self.goals) if self.goals else None
        self.agent_goals[agent_id] = goal
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is valid and not blocked"""
        if pos.x < 0 or pos.x >= self.width or pos.y < 0 or pos.y >= self.height:
            return False
        return self.grid[pos.y][pos.x] != CellType.OBSTACLE.value
    
    def get_neighbors(self, pos: Position) -> List[Position]:
        """Get valid neighboring positions"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        
        for dx, dy in directions:
            new_pos = Position(pos.x + dx, pos.y + dy)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def move_agent(self, agent_id: int, new_pos: Position) -> Tuple[bool, float]:
        """
        Move agent to new position
        
        Returns:
            (success, reward): Whether move succeeded and reward obtained
        """
        if agent_id not in self.agents:
            return False, 0.0
        
        current_pos = self.agents[agent_id]
        
        # Check if move is valid
        if not self.is_valid_position(new_pos):
            return False, 0.0
        
        # Check if position is occupied by another agent
        for aid, apos in self.agents.items():
            if aid != agent_id and apos == new_pos:
                return False, 0.0
        
        # Apply stochastic event (agent might not move as intended)
        if random.random() < self.stochastic_prob:
            # Randomly choose a valid neighbor instead
            neighbors = self.get_neighbors(current_pos)
            if neighbors:
                new_pos = random.choice(neighbors)
        
        # Move agent
        self.agents[agent_id] = new_pos
        
        # Check for reward
        reward = 0.0
        if new_pos in self.rewards:
            reward = self.rewards.pop(new_pos)  # Remove reward after collection
        
        self.step_count += 1
        return True, reward
    
    def agent_reached_goal(self, agent_id: int) -> bool:
        """Check if agent has reached its goal"""
        if agent_id not in self.agents or agent_id not in self.agent_goals:
            return False
        return self.agents[agent_id] == self.agent_goals[agent_id]
    
    def get_state(self) -> np.ndarray:
        """Get current state of the grid (for visualization)"""
        state = self.grid.copy()
        
        # Mark agent positions
        for agent_id, pos in self.agents.items():
            state[pos.y][pos.x] = CellType.AGENT.value
        
        return state
    
    def reset(self):
        """Reset the environment"""
        self.step_count = 0
        # Keep grid structure but reset agent positions
        # Agents should be re-added with new positions


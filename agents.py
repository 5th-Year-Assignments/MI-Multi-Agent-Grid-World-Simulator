"""
Agent implementations for grid world simulation
Includes BFS, DFS, A*, and Reinforcement Learning agents
"""

from collections import deque
import heapq
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from grid_world import GridWorld, Position, CellType


class Agent:
    """Base agent class"""
    
    def __init__(self, agent_id: int, grid_world: GridWorld):
        self.agent_id = agent_id
        self.grid_world = grid_world
        self.path: List[Position] = []
        self.current_path_index = 0
        self.total_reward = 0.0
        self.steps_taken = 0
        self.reached_goal = False
    
    def plan(self, start: Position, goal: Position) -> List[Position]:
        """Plan path from start to goal (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_next_action(self) -> Optional[Position]:
        """Get next position to move to (doesn't advance index - caller must do that)"""
        if self.current_path_index < len(self.path):
            next_pos = self.path[self.current_path_index]
            return next_pos
        return None
    
    def advance_path(self):
        """Advance to next position in path (call after successful move)"""
        if self.current_path_index < len(self.path):
            self.current_path_index += 1
    
    def update_path(self, current_pos: Position, goal: Position):
        """Replan path from current position to goal"""
        self.path = self.plan(current_pos, goal)
        self.current_path_index = 0
    
    def record_move(self, reward: float):
        """Record that agent made a move"""
        self.steps_taken += 1
        self.total_reward += reward


class BFSAgent(Agent):
    """Breadth-First Search agent"""
    
    def plan(self, start: Position, goal: Position) -> List[Position]:
        """Plan path using BFS"""
        if start == goal:
            return [goal]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal:
                return path[1:] 
            
            for neighbor in self.grid_world.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  


class DFSAgent(Agent):
    """Depth-First Search agent with iterative deepening for better paths"""
    
    def plan(self, start: Position, goal: Position) -> List[Position]:
        """Plan path using DFS with depth limit to avoid extremely long paths"""
        if start == goal:
            return [goal]
        
        # Use iterative deepening DFS - limit depth to prevent extremely long paths
        # Max depth is Manhattan distance * 2 
        max_depth = int(start.distance(goal) * 2.5)
        max_depth = max(max_depth, 50)  # Minimum depth limit
        
        stack = [(start, [start], 0)]  # (position, path, depth)
        visited_in_path = {start}
        
        while stack:
            current, path, depth = stack.pop()
            
            if current == goal:
                return path[1:]  # Exclude start position
            
            # Limit depth to prevent extremely long paths
            if depth >= max_depth:
                continue
            
            # Explore neighbors
            for neighbor in self.grid_world.get_neighbors(current):
                if neighbor not in visited_in_path:
                    visited_in_path.add(neighbor)
                    stack.append((neighbor, path + [neighbor], depth + 1))
        
        # If DFS with limit failed, try BFS as fallback (guaranteed to find path)
        # This ensures DFS always finds a path, just might use BFS if DFS path too long
        bfs_agent = BFSAgent(self.agent_id, self.grid_world)
        return bfs_agent.plan(start, goal)


class AStarAgent(Agent):
    """A* search agent with heuristic"""
    
    def heuristic(self, pos: Position, goal: Position) -> float:
        """Manhattan distance heuristic"""
        return pos.distance(goal)
    
    def plan(self, start: Position, goal: Position) -> List[Position]:
        """Plan path using A* algorithm"""
        if start == goal:
            return [goal]
        
        # Use counter to break ties in heap (prevents Position comparison errors)
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, float] = {start: 0}
        f_score: Dict[Position, float] = {start: self.heuristic(start, goal)}
        visited = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Reverse to get path from start to goal
            
            for neighbor in self.grid_world.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        return []  


class RLAgent(Agent):
    """Reinforcement Learning agent using Q-learning"""
    
    def __init__(self, agent_id: int, grid_world: GridWorld, 
                 learning_rate: float = 0.1, discount: float = 0.95, 
                 epsilon: float = 0.3, initial_q: float = 0.5):
        super().__init__(agent_id, grid_world)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.initial_q = initial_q  # Optimistic initialization
        self.q_table: Dict[Tuple[int, int, int, int], float] = {}
        self.last_state: Optional[Position] = None
        self.last_action: Optional[Position] = None
        self.steps_taken = 0  # Track steps for epsilon decay
    
    def state_to_key(self, pos: Position, goal: Position) -> Tuple[int, int, int, int]:
        """Convert state to Q-table key"""
        return (pos.x, pos.y, goal.x, goal.y)
    
    def get_q_value(self, state_key: Tuple[int, int, int, int], 
                   action: Position) -> float:
        """Get Q-value for state-action pair with optimistic initialization"""
        action_key = (action.x, action.y)
        full_key = state_key + action_key
        # Use optimistic initialization to encourage exploration
        return self.q_table.get(full_key, self.initial_q)
    
    def update_q_value(self, state_key: Tuple[int, int, int, int],
                      action: Position, reward: float, next_state: Position,
                      goal: Position):
        """Update Q-value using Q-learning update rule"""
        action_key = (action.x, action.y)
        full_key = state_key + action_key
        
        # Add reward for getting closer to goal
        current_pos = Position(state_key[0], state_key[1])
        distance_before = current_pos.distance(goal)
        distance_after = next_state.distance(goal)
        
        # Shaped reward: reward for getting closer to goal
        if distance_after < distance_before:
            reward += 0.5  # Reward for progress (increased)
        elif distance_after > distance_before:
            reward -= 0.2  # Penalty for moving away
        
        # Large reward for reaching goal
        if next_state == goal:
            reward += 50.0  # Increased goal reward
        
        # Get max Q-value for next state
        next_state_key = self.state_to_key(next_state, goal)
        max_next_q = 0.0
        for neighbor in self.grid_world.get_neighbors(next_state):
            q_val = self.get_q_value(next_state_key, neighbor)
            max_next_q = max(max_next_q, q_val)
        
        # Q-learning update
        current_q = self.q_table.get(full_key, 0.0)
        new_q = current_q + self.learning_rate * (
            reward + self.discount * max_next_q - current_q
        )
        self.q_table[full_key] = new_q
    
    def choose_action(self, current_pos: Position, goal: Position) -> Position:
        """Choose action using epsilon-greedy policy with adaptive exploration"""
        neighbors = self.grid_world.get_neighbors(current_pos)
        if not neighbors:
            return current_pos
        
        state_key = self.state_to_key(current_pos, goal)
        
        # Adaptive epsilon: decay over time but keep minimum exploration
        current_epsilon = max(0.1, self.epsilon * (0.99 ** self.steps_taken))
        
        # Epsilon-greedy: explore or exploit
        if np.random.random() < current_epsilon:
            # Exploration: choose random neighbor
            return random.choice(neighbors)
        else:
            # Exploitation: choose action with highest Q-value
            # If all Q-values are equal, add small random tie-breaking
            best_actions = []
            best_q = float('-inf')
            
            for neighbor in neighbors:
                q_val = self.get_q_value(state_key, neighbor)
                if q_val > best_q:
                    best_q = q_val
                    best_actions = [neighbor]
                elif abs(q_val - best_q) < 0.001:  # Essentially equal
                    best_actions.append(neighbor)
            
            # If multiple actions have same Q-value, choose randomly among them
            if len(best_actions) > 1:
                return random.choice(best_actions)
            return best_actions[0]
    
    def plan(self, start: Position, goal: Position) -> List[Position]:
        """RL agents don't pre-plan, they act reactively"""
        # Return empty path - RL agent will use choose_action instead
        return []
    
    def get_next_action(self) -> Optional[Position]:
        """Get next action using Q-learning policy"""
        if self.agent_id not in self.grid_world.agents:
            return None
        
        current_pos = self.grid_world.agents[self.agent_id]
        goal = self.grid_world.agent_goals.get(self.agent_id)
        
        if goal is None:
            return None
        
        return self.choose_action(current_pos, goal)


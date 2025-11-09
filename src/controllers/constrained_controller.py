"""
Constrained controller base interface for hierarchical RL system.

This module defines the interface for controllers that operate under
explicit safety constraints in a Constrained MDP framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
from .base_controller import BaseController


class ConstrainedController(BaseController):
    """
    Base class for controllers operating in Constrained MDPs.
    
    Extends BaseController with constraint tracking and cost functions.
    Separates reward (performance) from cost (safety constraints).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize constrained controller.
        
        Args:
            config: Configuration dictionary including:
                - constraint_thresholds: Dict of constraint limits
                - dual_learning_rate: Learning rate for Lagrangian multipliers
                - cost_weights: Weights for different cost components
        """
        super().__init__(config)
        
        # Constraint configuration
        self.constraint_thresholds = config.get('constraint_thresholds', {
            'ttc_min': 2.0,          # seconds
            'headway_s0': 2.0,       # meters
            'headway_T': 1.5,        # seconds
            'clearance_min': 1.5,    # meters
            'jerk_max': 3.0,         # m/s³
        })
        
        # Lagrangian dual variables (one per constraint)
        self.dual_variables = {
            'lambda_ttc': 0.0,
            'lambda_headway': 0.0,
            'lambda_clearance': 0.0,
            'lambda_jerk': 0.0,
        }
        
        # Dual learning rate
        self.dual_lr = config.get('dual_learning_rate', 0.01)
        
        # Cost tracking
        self.episode_costs = {
            'ttc': [],
            'headway': [],
            'clearance': [],
            'jerk': [],
        }
        
        # Statistics
        self.constraint_violations = {
            'ttc': 0,
            'headway': 0,
            'clearance': 0,
            'jerk': 0,
        }
        self.timesteps = 0
    
    @abstractmethod
    def compute_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute control action based on observation.
        
        Args:
            observation: Current observation
            info: Additional information
        
        Returns:
            action: Control action
        """
        pass
    
    @abstractmethod
    def compute_costs(
        self, 
        observation: np.ndarray, 
        action: np.ndarray,
        next_observation: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute constraint costs for current state-action pair.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation (optional, for derivative estimates)
        
        Returns:
            costs: Dictionary of constraint costs {constraint_name: cost_value}
        """
        pass
    
    def update_dual_variables(self, expected_costs: Dict[str, float]):
        """
        Update Lagrangian dual variables based on constraint violations.
        
        Uses projected gradient ascent:
            λ_k ← max(0, λ_k + η * (E[C_k] - d_k))
        
        Args:
            expected_costs: Expected cost for each constraint over recent episodes
        """
        for constraint_name, expected_cost in expected_costs.items():
            lambda_key = f'lambda_{constraint_name}'
            threshold = self.constraint_thresholds.get(f'{constraint_name}_threshold', 0.0)
            
            # Projected gradient ascent (project to non-negative)
            violation = expected_cost - threshold
            self.dual_variables[lambda_key] = max(
                0.0,
                self.dual_variables[lambda_key] + self.dual_lr * violation
            )
    
    def track_cost(self, constraint_name: str, cost_value: float):
        """
        Track cost for a specific constraint.
        
        Args:
            constraint_name: Name of constraint
            cost_value: Cost value to track
        """
        if constraint_name in self.episode_costs:
            self.episode_costs[constraint_name].append(cost_value)
            
            # Track violations (cost > small threshold)
            if cost_value > 1e-6:
                self.constraint_violations[constraint_name] += 1
    
    def get_expected_costs(self) -> Dict[str, float]:
        """
        Get expected (average) cost for each constraint over recent episode.
        
        Returns:
            expected_costs: Dictionary of average costs
        """
        expected_costs = {}
        for constraint_name, costs in self.episode_costs.items():
            if len(costs) > 0:
                expected_costs[constraint_name] = np.mean(costs)
            else:
                expected_costs[constraint_name] = 0.0
        return expected_costs
    
    def reset(self):
        """Reset controller state at episode start."""
        super().reset()
        
        # Reset episode costs
        self.episode_costs = {key: [] for key in self.episode_costs}
        self.timesteps = 0
    
    def get_constraint_stats(self) -> Dict[str, Any]:
        """
        Get statistics on constraint satisfaction.
        
        Returns:
            stats: Dictionary of constraint violation rates and dual variables
        """
        total_timesteps = max(1, self.timesteps)
        
        stats = {
            'violation_rates': {
                name: count / total_timesteps 
                for name, count in self.constraint_violations.items()
            },
            'dual_variables': self.dual_variables.copy(),
            'expected_costs': self.get_expected_costs(),
            'total_timesteps': self.timesteps,
        }
        
        return stats


class ManeuverPolicy(ABC):
    """
    Abstract interface for high-level maneuver decision-making.
    
    Decides discrete maneuvers: keep_lane, change_left, change_right, overtake.
    """
    
    # Maneuver constants
    KEEP_LANE = 0
    CHANGE_LEFT = 1
    CHANGE_RIGHT = 2
    OVERTAKE = 3
    
    MANEUVER_NAMES = {
        KEEP_LANE: "keep_lane",
        CHANGE_LEFT: "change_left",
        CHANGE_RIGHT: "change_right",
        OVERTAKE: "overtake",
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize maneuver policy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Maneuver state
        self.current_maneuver = self.KEEP_LANE
        self.time_in_maneuver = 0.0
        self.min_dwell_time = config.get('min_dwell_time', 2.0)  # seconds
        
        # Statistics
        self.maneuver_counts = {
            self.KEEP_LANE: 0,
            self.CHANGE_LEFT: 0,
            self.CHANGE_RIGHT: 0,
            self.OVERTAKE: 0,
        }
    
    @abstractmethod
    def select_maneuver(
        self, 
        observation: np.ndarray,
        info: Dict[str, Any] = None
    ) -> int:
        """
        Select high-level maneuver based on observation.
        
        Args:
            observation: Current observation
            info: Additional information (traffic state, etc.)
        
        Returns:
            maneuver: Integer maneuver ID (0-3)
        """
        pass
    
    def update_maneuver_state(self, maneuver: int, dt: float):
        """
        Update internal maneuver state.
        
        Args:
            maneuver: Selected maneuver
            dt: Time step (seconds)
        """
        if maneuver == self.current_maneuver:
            self.time_in_maneuver += dt
        else:
            # Check hysteresis
            if self.time_in_maneuver >= self.min_dwell_time:
                # Allow maneuver change
                self.current_maneuver = maneuver
                self.time_in_maneuver = 0.0
                self.maneuver_counts[maneuver] += 1
            # else: remain in current maneuver
    
    def get_maneuver_name(self, maneuver: int) -> str:
        """Get string name for maneuver ID."""
        return self.MANEUVER_NAMES.get(maneuver, "unknown")
    
    def reset(self):
        """Reset maneuver state."""
        self.current_maneuver = self.KEEP_LANE
        self.time_in_maneuver = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get maneuver selection statistics."""
        total = sum(self.maneuver_counts.values())
        if total == 0:
            return {'maneuver_counts': self.maneuver_counts}
        
        return {
            'maneuver_counts': self.maneuver_counts,
            'maneuver_frequencies': {
                self.get_maneuver_name(m): count / total
                for m, count in self.maneuver_counts.items()
            }
        }


class HierarchicalController(ConstrainedController):
    """
    Hierarchical controller that coordinates maneuver and trajectory policies.
    
    Two-layer architecture:
    - High-level: Maneuver policy (discrete decisions at 1 Hz)
    - Low-level: Trajectory policy (continuous control at 10 Hz)
    """
    
    def __init__(
        self, 
        maneuver_policy: ManeuverPolicy,
        trajectory_policy: ConstrainedController,
        config: Dict[str, Any] = None
    ):
        """
        Initialize hierarchical controller.
        
        Args:
            maneuver_policy: High-level maneuver policy
            trajectory_policy: Low-level trajectory policy
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.maneuver_policy = maneuver_policy
        self.trajectory_policy = trajectory_policy
        
        # Update frequencies
        self.maneuver_freq = config.get('maneuver_frequency', 1.0)  # Hz
        self.trajectory_freq = config.get('trajectory_frequency', 10.0)  # Hz
        
        # Timing
        self.dt = 1.0 / self.trajectory_freq
        self.maneuver_dt = 1.0 / self.maneuver_freq
        self.time_since_maneuver_update = 0.0
    
    def compute_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute hierarchical action.
        
        Updates maneuver at low frequency, trajectory at high frequency.
        
        Args:
            observation: Current observation
            info: Additional information
        
        Returns:
            action: Continuous control action from trajectory policy
        """
        # Update maneuver at low frequency
        if self.time_since_maneuver_update >= self.maneuver_dt:
            maneuver = self.maneuver_policy.select_maneuver(observation, info)
            self.maneuver_policy.update_maneuver_state(maneuver, self.maneuver_dt)
            self.time_since_maneuver_update = 0.0
        
        # Add current maneuver to info for trajectory policy
        if info is None:
            info = {}
        info['current_maneuver'] = self.maneuver_policy.current_maneuver
        info['time_in_maneuver'] = self.maneuver_policy.time_in_maneuver
        
        # Compute trajectory action at high frequency
        action = self.trajectory_policy.compute_action(observation, info)
        
        # Update timing
        self.time_since_maneuver_update += self.dt
        self.timesteps += 1
        
        return action
    
    def compute_costs(
        self, 
        observation: np.ndarray, 
        action: np.ndarray,
        next_observation: np.ndarray = None
    ) -> Dict[str, float]:
        """Delegate cost computation to trajectory policy."""
        return self.trajectory_policy.compute_costs(observation, action, next_observation)
    
    def reset(self):
        """Reset both policies."""
        super().reset()
        self.maneuver_policy.reset()
        self.trajectory_policy.reset()
        self.time_since_maneuver_update = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both policies."""
        stats = super().get_constraint_stats()
        stats['maneuver_stats'] = self.maneuver_policy.get_stats()
        stats['trajectory_stats'] = self.trajectory_policy.get_constraint_stats()
        return stats

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
        # Initialize attributes BEFORE calling super().__init__() 
        # because BaseController.__init__ calls reset()
        
        # Constraint configuration
        self.constraint_thresholds = config.get('constraint_thresholds', {
            'ttc_min': 1.0,          # seconds
            'headway_s0': 2.0,       # meters
            'headway_T': 1.5,        # seconds
            'clearance_min': 1.5,    # meters
            'jerk_max': 3.0,         # m/s³
        }) if config else {
            'ttc_min': 1.0,
            'headway_s0': 2.0,
            'headway_T': 1.5,
            'clearance_min': 1.5,
            'jerk_max': 3.0,
        }
        
        # Lagrangian dual variables (one per constraint)
        self.dual_variables = {
            'lambda_ttc': 0.0,
            'lambda_headway': 0.0,
            'lambda_clearance': 0.0,
            'lambda_jerk': 0.0,
        }
        
        # Dual learning rate
        self.dual_lr = config.get('dual_learning_rate', 0.005) if config else 0.005
        
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
        
        # Now call parent init (which will call reset())
        super().__init__(config)
    
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


class RuleBasedManeuverPolicy(ManeuverPolicy):
    """
    Rule-based maneuver policy implementing gap acceptance and advantage-to-change logic.
    
    Decision logic:
    1. Check if current lane is blocked (slow lead vehicle)
    2. Evaluate gap safety in adjacent lanes (TTC-based)
    3. Compute time advantage for lane changes
    4. Select maneuver with highest advantage if safe
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize rule-based maneuver policy.
        
        Args:
            config: Configuration including:
                - target_speed: Desired speed (m/s)
                - speed_diff_threshold: Minimum speed difference to trigger overtake
                - ttc_front_threshold: Minimum TTC to front vehicle in target lane
                - ttc_rear_threshold: Minimum TTC to rear vehicle in target lane
                - time_advantage_threshold: Minimum time advantage to change lanes
        """
        super().__init__(config)
        
        # Decision thresholds
        self.target_speed = config.get('target_speed', 25.0)  # m/s (~90 km/h)
        self.speed_diff_threshold = config.get('speed_diff_threshold', 3.0)  # m/s
        self.ttc_front_threshold = config.get('ttc_front_threshold', 3.0)  # seconds
        self.ttc_rear_threshold = config.get('ttc_rear_threshold', 3.0)  # seconds
        self.time_advantage_threshold = config.get('time_advantage_threshold', 2.0)  # m/s
        
        # Statistics
        self.lane_change_attempts = 0
        self.lane_change_rejections = 0
    
    def select_maneuver(
        self, 
        observation: np.ndarray,
        info: Dict[str, Any] = None
    ) -> int:
        """
        Select maneuver using rule-based logic.
        
        Args:
            observation: 18-dim observation from NADE environment:
                [0] lead_rel_distance
                [1] lead_rel_velocity
                [2] lead_ttc
                [3] ego_velocity
                [4] ego_lane_idx (normalized)
                [5] ego_lane_offset
                [6-8] right vehicle (distance, rel_vel, lateral_dist)
                [9-11] left vehicle (distance, rel_vel, lateral_dist)
                [12-17] predictions (not used in rule-based policy)
            info: Additional information
        
        Returns:
            maneuver: Selected maneuver (0-3)
        """
        info = info or {}
        
        # Extract observation features
        lead_distance = observation[0]
        lead_rel_velocity = observation[1]
        lead_ttc = observation[2]
        ego_velocity = observation[3]
        ego_lane_idx_norm = observation[4]
        
        right_distance = observation[6]
        right_rel_velocity = observation[7]
        
        left_distance = observation[9]
        left_rel_velocity = observation[10]
        
        # Estimate current lane (denormalize)
        num_lanes = 3  # NADE default
        ego_lane = int(ego_lane_idx_norm * (num_lanes - 1))
        
        # Check if current lane is blocked
        is_blocked = False
        if lead_distance < 100 and lead_distance > 0:  # Lead vehicle present
            lead_speed = ego_velocity + lead_rel_velocity
            if lead_speed < (self.target_speed - self.speed_diff_threshold):
                is_blocked = True
        
        # If not blocked, keep lane
        if not is_blocked:
            return self.KEEP_LANE
        
        # Evaluate lane change options
        lane_change_options = []
        
        # Check right lane (if not in rightmost lane)
        if ego_lane < (num_lanes - 1):
            if self._is_gap_safe(right_distance, right_rel_velocity, 'right'):
                advantage = self._compute_lane_advantage(
                    right_distance, right_rel_velocity, ego_velocity
                )
                if advantage > self.time_advantage_threshold:
                    lane_change_options.append(('right', self.CHANGE_RIGHT, advantage))
        
        # Check left lane (if not in leftmost lane)
        if ego_lane > 0:
            if self._is_gap_safe(left_distance, left_rel_velocity, 'left'):
                advantage = self._compute_lane_advantage(
                    left_distance, left_rel_velocity, ego_velocity
                )
                if advantage > self.time_advantage_threshold:
                    lane_change_options.append(('left', self.CHANGE_LEFT, advantage))
        
        # Select best option
        if lane_change_options:
            # Sort by advantage (descending)
            lane_change_options.sort(key=lambda x: x[2], reverse=True)
            self.lane_change_attempts += 1
            return lane_change_options[0][1]  # Return maneuver with highest advantage
        else:
            # No safe lane change available, keep lane
            if is_blocked:
                self.lane_change_rejections += 1
            return self.KEEP_LANE
    
    def _is_gap_safe(
        self, 
        target_lane_distance: float,
        target_lane_rel_velocity: float,
        lane_side: str
    ) -> bool:
        """
        Check if gap in target lane is safe for lane change.
        
        Args:
            target_lane_distance: Longitudinal distance to nearest vehicle in target lane
            target_lane_rel_velocity: Relative velocity with that vehicle
            lane_side: 'left' or 'right'
        
        Returns:
            is_safe: True if gap is safe
        """
        # Check if there's a vehicle in the target lane
        if abs(target_lane_distance) > 50:  # No vehicle within 50m
            return True
        
        # Vehicle ahead in target lane
        if target_lane_distance > 0:
            # Check TTC to front
            if target_lane_rel_velocity < -0.1:  # Approaching
                ttc_front = target_lane_distance / abs(target_lane_rel_velocity)
                if ttc_front < self.ttc_front_threshold:
                    return False
        
        # Vehicle behind in target lane
        else:
            # Check TTC to rear
            if target_lane_rel_velocity > 0.1:  # Vehicle catching up
                ttc_rear = abs(target_lane_distance) / target_lane_rel_velocity
                if ttc_rear < self.ttc_rear_threshold:
                    return False
        
        return True
    
    def _compute_lane_advantage(
        self,
        target_lane_distance: float,
        target_lane_rel_velocity: float,
        ego_velocity: float
    ) -> float:
        """
        Compute time advantage of changing to target lane.
        
        Args:
            target_lane_distance: Distance to lead vehicle in target lane
            target_lane_rel_velocity: Relative velocity with lead vehicle
            ego_velocity: Ego velocity
        
        Returns:
            advantage: Speed advantage (m/s) in target lane
        """
        # If no vehicle ahead in target lane, advantage is difference to target speed
        if target_lane_distance > 50:
            return self.target_speed - ego_velocity
        
        # Otherwise, advantage is speed of lead vehicle in target lane
        lead_speed_target_lane = ego_velocity + target_lane_rel_velocity
        advantage = lead_speed_target_lane - ego_velocity
        
        return max(0.0, advantage)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics including lane change attempts and rejections."""
        stats = super().get_stats()
        stats['lane_change_attempts'] = self.lane_change_attempts
        stats['lane_change_rejections'] = self.lane_change_rejections
        if self.lane_change_attempts > 0:
            stats['lane_change_success_rate'] = (
                1.0 - self.lane_change_rejections / self.lane_change_attempts
            )
        return stats


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
        # Initialize attributes BEFORE calling super().__init__() 
        # because it will call reset()
        self.maneuver_policy = maneuver_policy
        self.trajectory_policy = trajectory_policy
        
        # Update frequencies
        config = config or {}
        self.maneuver_freq = config.get('maneuver_frequency', 1.0)  # Hz
        self.trajectory_freq = config.get('trajectory_frequency', 10.0)  # Hz
        
        # Timing
        self.dt = 1.0 / self.trajectory_freq
        self.maneuver_dt = 1.0 / self.maneuver_freq
        self.time_since_maneuver_update = 0.0
        
        # Now call parent init (which will call reset())
        super().__init__(config)
    
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

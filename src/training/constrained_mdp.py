"""
Constrained MDP reward and cost functions for hierarchical RL.

This module implements the separation of performance (reward) and safety (cost)
functions for training controllers in Constrained MDPs.
"""

from typing import Dict, Any, Tuple
import numpy as np


class RewardFunction:
    """
    Reward function for performance objectives.
    
    Implements potential-based shaping for speed tracking and lane centering
    to accelerate learning while preserving optimality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize reward function.
        
        Args:
            config: Configuration with reward weights and target parameters
        """
        config = config or {}
        
        # Reward weights
        self.w_speed = config.get('w_speed', 1.0)
        self.w_lane = config.get('w_lane', 0.5)
        self.w_progress = config.get('w_progress', 0.4)
        self.w_jerk = config.get('w_jerk', 0.1)
        self.w_accel = config.get('w_accel', 0.05)
        
        # Target parameters
        self.v_target = config.get('v_target', 27.0)  # m/s (typical highway speed)
        self.v_min = config.get('v_min', 15.0)       # m/s
        self.v_max = config.get('v_max', 35.0)       # m/s
        
        # Discount factor for potential shaping
        self.gamma = config.get('gamma', 0.99)
        
        # Previous state for potential computation
        self.prev_observation = None
    
    def compute_reward(
        self, 
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any] = None
    ) -> float:
        """
        Compute total reward for transition.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            info: Additional information
        
        Returns:
            reward: Total reward value
        """
        # Extract state features
        ego_velocity = observation[3]  # Index 3 is ego velocity
        next_ego_velocity = next_observation[3]
        
        # Lane offset (if available, otherwise 0)
        lane_offset = info.get('lane_offset', 0.0) if info else 0.0
        next_lane_offset = info.get('next_lane_offset', 0.0) if info else 0.0
        
        # 1. Speed tracking reward (potential-based shaping)
        r_speed = self._speed_tracking_reward(ego_velocity, next_ego_velocity)
        
        # 2. Lane centering reward (potential-based shaping)
        r_lane = self._lane_centering_reward(lane_offset, next_lane_offset)
        
        # 3. Progress reward (encourage forward motion)
        r_progress = self.w_progress * ego_velocity * 0.1  # dt = 0.1s typical
        
        # 4. Comfort reward (penalize harsh actions)
        r_comfort = self._comfort_reward(action, info)
        
        # Total reward
        reward = r_speed + r_lane + r_progress + r_comfort
        
        return reward
    
    def _speed_tracking_reward(self, v_current: float, v_next: float) -> float:
        """
        Speed tracking reward using potential-based shaping.
        
        Potential: Φ(s) = -w_speed * (v - v_target)²
        Shaping: F(s, s') = γ * Φ(s') - Φ(s)
        
        Args:
            v_current: Current velocity
            v_next: Next velocity
        
        Returns:
            reward: Speed tracking reward
        """
        phi_current = -self.w_speed * (v_current - self.v_target) ** 2
        phi_next = -self.w_speed * (v_next - self.v_target) ** 2
        
        return self.gamma * phi_next - phi_current
    
    def _lane_centering_reward(self, offset_current: float, offset_next: float) -> float:
        """
        Lane centering reward using potential-based shaping.
        
        Potential: Φ(s) = -w_lane * |offset|²
        Shaping: F(s, s') = γ * Φ(s') - Φ(s)
        
        Args:
            offset_current: Current lane offset
            offset_next: Next lane offset
        
        Returns:
            reward: Lane centering reward
        """
        phi_current = -self.w_lane * offset_current ** 2
        phi_next = -self.w_lane * offset_next ** 2
        
        return self.gamma * phi_next - phi_current
    
    def _comfort_reward(self, action: np.ndarray, info: Dict[str, Any] = None) -> float:
        """
        Comfort reward penalizing harsh accelerations and jerk.
        
        Args:
            action: Control action [a_long, a_lat]
            info: Additional information (previous action for jerk)
        
        Returns:
            reward: Comfort reward (negative penalty)
        """
        a_long = action[0]
        
        # Acceleration penalty
        r_accel = -self.w_accel * a_long ** 2
        
        # Jerk penalty (if previous action available)
        r_jerk = 0.0
        if info and 'prev_action' in info:
            prev_a_long = info['prev_action'][0]
            dt = 0.1  # typical timestep
            jerk = (a_long - prev_a_long) / dt
            r_jerk = -self.w_jerk * jerk ** 2
        
        return r_accel + r_jerk
    
    def reset(self):
        """Reset internal state."""
        self.prev_observation = None


class CostFunction:
    """
    Cost functions for safety constraints.
    
    Implements barrier-style cost functions that penalize constraint violations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cost function.
        
        Args:
            config: Configuration with constraint thresholds
        """
        config = config or {}
        
        # Constraint thresholds
        self.ttc_min = config.get('ttc_min', 2.0)          # seconds
        self.headway_s0 = config.get('headway_s0', 2.0)   # meters
        self.headway_T = config.get('headway_T', 1.5)     # seconds
        self.clearance_min = config.get('clearance_min', 1.5)  # meters
        self.jerk_max = config.get('jerk_max', 3.0)       # m/s³
        
        # Distance threshold for TTC calculation
        self.ttc_distance_threshold = config.get('ttc_distance_threshold', 50.0)  # meters
    
    def compute_costs(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray = None,
        info: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Compute all constraint costs.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            info: Additional information
        
        Returns:
            costs: Dictionary of {constraint_name: cost_value}
        """
        costs = {}
        
        # Extract state features from observation
        # Observation format: [rel_dist, rel_vel, ttc, ego_vel, lane_occ, ...]
        rel_distance = observation[0]
        rel_velocity = observation[1]
        ttc = observation[2]
        ego_velocity = observation[3]
        
        # C1: TTC constraint
        costs['ttc'] = self._ttc_cost(ttc, rel_distance)
        
        # C2: Headway constraint
        costs['headway'] = self._headway_cost(rel_distance, ego_velocity)
        
        # C3: Lateral clearance constraint (if lane changing)
        costs['clearance'] = self._clearance_cost(observation, info)
        
        # C4: Jerk constraint
        costs['jerk'] = self._jerk_cost(action, info)
        
        return costs
    
    def _ttc_cost(self, ttc: float, rel_distance: float) -> float:
        """
        TTC constraint cost.
        
        Cost is positive if TTC < τ_min and vehicle is close enough.
        
        Args:
            ttc: Time-to-collision
            rel_distance: Relative distance to lead vehicle
        
        Returns:
            cost: TTC constraint cost
        """
        # Only apply if lead vehicle is close
        if rel_distance > self.ttc_distance_threshold:
            return 0.0
        
        # Check if TTC violates constraint
        if ttc < self.ttc_min:
            violation = self.ttc_min - ttc
            return self._barrier_function(violation)
        
        return 0.0
    
    def _headway_cost(self, rel_distance: float, ego_velocity: float) -> float:
        """
        Headway constraint cost (IDM-style).
        
        Requires: s ≥ s_0 + T_h * v_ego
        
        Args:
            rel_distance: Distance to lead vehicle
            ego_velocity: Ego vehicle velocity
        
        Returns:
            cost: Headway constraint cost
        """
        s_desired = self.headway_s0 + self.headway_T * ego_velocity
        
        if rel_distance < s_desired:
            violation = s_desired - rel_distance
            return self._barrier_function(violation)
        
        return 0.0
    
    def _clearance_cost(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any] = None
    ) -> float:
        """
        Lateral clearance constraint cost during lane changes.
        
        Args:
            observation: Current observation
            info: Additional information (maneuver state)
        
        Returns:
            cost: Clearance constraint cost
        """
        # Only apply during lane change maneuvers
        if info is None or info.get('current_maneuver', 0) == 0:  # 0 = KEEP_LANE
            return 0.0
        
        # Check if we have lateral clearance info in observation
        # Indices 5-7 are right vehicle info: [distance, rel_vel, lateral_dist]
        if len(observation) >= 8:
            lateral_distance = observation[7]  # Right vehicle lateral distance
            
            if lateral_distance < self.clearance_min:
                violation = self.clearance_min - lateral_distance
                return self._barrier_function(violation)
        
        return 0.0
    
    def _jerk_cost(self, action: np.ndarray, info: Dict[str, Any] = None) -> float:
        """
        Jerk constraint cost.
        
        Requires: |jerk| ≤ j_max
        
        Args:
            action: Current action
            info: Additional information (previous action)
        
        Returns:
            cost: Jerk constraint cost
        """
        if info is None or 'prev_action' not in info:
            return 0.0
        
        a_long = action[0]
        prev_a_long = info['prev_action'][0]
        dt = 0.1  # typical timestep
        
        jerk = abs((a_long - prev_a_long) / dt)
        
        if jerk > self.jerk_max:
            violation = jerk - self.jerk_max
            return self._barrier_function(violation)
        
        return 0.0
    
    def _barrier_function(self, violation: float) -> float:
        """
        Smooth barrier function for constraint violations.
        
        Uses quadratic penalty: ψ(ε) = ε²
        
        Args:
            violation: Amount of constraint violation (positive value)
        
        Returns:
            cost: Barrier function output
        """
        if violation <= 0:
            return 0.0
        return violation ** 2


class ConstrainedMDP:
    """
    Complete Constrained MDP formulation combining reward and cost functions.
    
    Provides unified interface for computing augmented rewards during training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Constrained MDP.
        
        Args:
            config: Configuration dictionary
        """
        self.reward_fn = RewardFunction(config)
        self.cost_fn = CostFunction(config)
        
        # Dual variables (Lagrangian multipliers)
        self.dual_variables = {
            'lambda_ttc': 0.0,
            'lambda_headway': 0.0,
            'lambda_clearance': 0.0,
            'lambda_jerk': 0.0,
        }
    
    def compute_augmented_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute augmented reward for Lagrangian SAC training.
        
        Augmented reward: r_aug = r - Σ_k λ_k * c_k
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            info: Additional information
        
        Returns:
            augmented_reward: Reward minus constraint penalties
            costs: Dictionary of individual costs
        """
        # Compute reward
        reward = self.reward_fn.compute_reward(observation, action, next_observation, info)
        
        # Compute costs
        costs = self.cost_fn.compute_costs(observation, action, next_observation, info)
        
        # Compute penalty from dual variables
        penalty = sum(
            self.dual_variables[f'lambda_{name}'] * cost_value
            for name, cost_value in costs.items()
        )
        
        # Augmented reward
        augmented_reward = reward - penalty
        
        return augmented_reward, costs
    
    def update_dual_variables(self, expected_costs: Dict[str, float], learning_rate: float = 0.01):
        """
        Update Lagrangian dual variables.
        
        Args:
            expected_costs: Expected cost for each constraint
            learning_rate: Learning rate for dual updates
        """
        for constraint_name, expected_cost in expected_costs.items():
            lambda_key = f'lambda_{constraint_name}'
            
            # Constraint threshold (aim for zero violations)
            threshold = 0.0
            
            # Projected gradient ascent
            violation = expected_cost - threshold
            self.dual_variables[lambda_key] = max(
                0.0,
                self.dual_variables[lambda_key] + learning_rate * violation
            )
    
    def get_dual_variables(self) -> Dict[str, float]:
        """Get current dual variable values."""
        return self.dual_variables.copy()
    
    def set_dual_variables(self, dual_vars: Dict[str, float]):
        """Set dual variable values."""
        self.dual_variables.update(dual_vars)
    
    def reset(self):
        """Reset internal state."""
        self.reward_fn.reset()

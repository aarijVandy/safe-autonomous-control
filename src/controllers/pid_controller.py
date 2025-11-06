"""
PID Controller for autonomous vehicle control.

This module implements a PID (Proportional-Integral-Derivative) controller
for longitudinal control (speed/acceleration) of an autonomous vehicle.
"""

import numpy as np
from typing import Any, Dict, Optional
from .base_controller import BaseController


class PIDController(BaseController):
    """
    PID controller for vehicle longitudinal control.
    
    Observation space:
        [0] relative_distance: Distance to lead vehicle (m)
        [1] relative_velocity: Velocity difference (m/s)
        [2] time_to_collision: TTC (s)
        [3] ego_velocity: Current vehicle speed (m/s)
        [4] lane_occupancy: Adjacent lane status (0 or 1)
    
    Action space:
        [0] acceleration: Control signal in [-3.0, 2.0] m/s²
    """
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.2,
        target_speed: float = 25.0,
        safe_distance: float = 20.0,
        min_ttc: float = 3.0,
        action_limits: tuple = (-3.0, 2.0),
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            target_speed: Desired cruising speed (m/s)
            safe_distance: Minimum safe following distance (m)
            min_ttc: Minimum acceptable time-to-collision (s)
            action_limits: Tuple of (min_acceleration, max_acceleration)
            config: Additional configuration parameters
        """
        super().__init__(config)
        
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Control parameters
        self.target_speed = target_speed
        self.safe_distance = safe_distance
        self.min_ttc = min_ttc
        self.action_limits = action_limits
        
        # State variables
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_action = 0.0
        
        # For jerk minimization
        self.jerk_weight = 0.5  # Weight for smoothing acceleration changes
        

        self.metrics = {
            "total_jerk": 0.0,
            "num_steps": 0,
            "ttc_violations": 0,
            "speed_errors": [],
        }
    
    def reset(self):
        """Reset controller state at the beginning of an episode."""
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_action = 0.0
        
        # Reset 
        self.metrics = {
            "total_jerk": 0.0,
            "num_steps": 0,
            "ttc_violations": 0,
            "speed_errors": [],
        }
    
    def compute_action(
        self, observation: np.ndarray, info: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Compute PID control action based on observation.
        
        The controller uses a cascaded approach:
        1. Safety check: If TTC is low or distance is small, brake
        2. Following mode: Maintain safe distance from lead vehicle
        3. Cruising mode: Maintain target speed if no lead vehicle
        
        Args:
            observation: [relative_distance, relative_velocity, ttc, ego_velocity, lane_occupancy]
            info: Additional information (optional)
        
        Returns:
            action: Acceleration command in [-3.0, 2.0] m/s²
        """
        # Parse observation
        relative_distance = observation[0]
        relative_velocity = observation[1]
        ttc = observation[2]
        ego_velocity = observation[3]
        
        # Determine control mode
        has_lead_vehicle = relative_distance < 100.0  # Lead vehicle exists
        
        # SAFETY OVERRIDE: Emergency braking if TTC is too low
        if has_lead_vehicle and ttc < self.min_ttc and relative_velocity < 0:

            error = relative_distance - self.safe_distance
            action = -2.5 if error < 0 else -1.5
            self.metrics["ttc_violations"] += 1
        
        # FOLLOWING MODE: Maintain safe distance from lead vehicle
        elif has_lead_vehicle:

            desired_distance = max(self.safe_distance, ego_velocity * 2.0)
            distance_error = relative_distance - desired_distance
            
            # consider relative velocity
            velocity_error = relative_velocity
            error = distance_error + 0.5 * velocity_error
            action = self._compute_pid(error)
        
        # CRUISING MODE: Maintain target speed
        else:
            speed_error = self.target_speed - ego_velocity
            error = speed_error
            
            action = self._compute_pid(error)
            self.metrics["speed_errors"].append(abs(speed_error))
        
        # jerk minimization (smooth transitions)
        action = self._smooth_action(action)
        
        action = np.clip(action, self.action_limits[0], self.action_limits[1])
        
        self.metrics["num_steps"] += 1
        jerk = abs(action - self.prev_action)
        self.metrics["total_jerk"] += jerk
        
        # Store for next iteration
        self.prev_action = action
        
        return np.array([action], dtype=np.float32)
    
    def _compute_pid(self, error: float) -> float:
        """
        Compute PID control output.
        """

        p_term = self.kp * error
        
        self.integral_error += error
        # clamp integral error
        max_integral = 10.0
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        i_term = self.ki * self.integral_error
        
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error

        output = p_term + i_term + d_term
        return output
    
    def _smooth_action(self, action: float) -> float:
        """
        Apply smoothing to minimize jerk.
        
        action: Raw action from PID
        
        Returns:
            smoothed_action: Smoothed action
        """
        # Exponential moving average for smooth transitions
        smoothed = self.jerk_weight * action + (1 - self.jerk_weight) * self.prev_action
        
        # Limit acceleration change rate 
        max_delta = 1.5  
        delta = smoothed - self.prev_action
        if abs(delta) > max_delta:
            smoothed = self.prev_action + np.sign(delta) * max_delta
        
        return smoothed
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the current episode.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        metrics = self.metrics.copy()
        
        # Compute average jerk
        if metrics["num_steps"] > 0:
            metrics["avg_jerk"] = metrics["total_jerk"] / metrics["num_steps"]
        else:
            metrics["avg_jerk"] = 0.0
        
        # Compute average speed error
        if len(metrics["speed_errors"]) > 0:
            metrics["avg_speed_error"] = np.mean(metrics["speed_errors"])
        else:
            metrics["avg_speed_error"] = 0.0
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """Get current PID parameters."""
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "target_speed": self.target_speed,
            "safe_distance": self.safe_distance,
            "min_ttc": self.min_ttc,
        }
    
    def set_params(self, params: Dict[str, Any]):
        """Set PID parameters."""
        if "kp" in params:
            self.kp = params["kp"]
        if "ki" in params:
            self.ki = params["ki"]
        if "kd" in params:
            self.kd = params["kd"]
        if "target_speed" in params:
            self.target_speed = params["target_speed"]
        if "safe_distance" in params:
            self.safe_distance = params["safe_distance"]
        if "min_ttc" in params:
            self.min_ttc = params["min_ttc"]

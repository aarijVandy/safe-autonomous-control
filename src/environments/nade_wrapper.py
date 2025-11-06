"""
NADE-compatible Highway Environment Wrapper.

This module provides a Gymnasium-compatible wrapper for highway-env
that is configured to match NADE (Naturalistic and Adversarial Driving Environment)
specifications:
- 3-lane highway
- Adversarial traffic behaviors (cut-ins, hard brakes, slowdowns)
- Longitudinal control focus
- Configurable traffic scenarios
"""

from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import highway_env  # Register highway-env environments


class NADEHighwayEnv(gym.Env):
    """
    NADE-compatible highway environment wrapper.
    
    This environment wraps highway-env to provide NADE-compatible features:
    - 3-lane highway configuration
    - Adversarial agent behaviors
    - Speed control and lane changing
    - Gymnasium-compatible interface
    
    Observation Space:
        - Relative distance to lead vehicle (m)
        - Relative velocity (m/s)
        - Time-to-collision (s)
        - Ego vehicle velocity (m/s)
        - (Optional) Adjacent lane occupancy
    
    Action Space:
        - Continuous: acceleration/deceleration [-3, 2] m/s²
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_lanes: int = 3,
        vehicles_count: int = 20,
        duration: float = 40.0,
        adversarial_mode: bool = False,
        adversarial_intensity: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize NADE highway environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            num_lanes: Number of lanes (default: 3 for NADE compatibility)
            vehicles_count: Number of vehicles in the environment
            duration: Episode duration in seconds
            adversarial_mode: Enable adversarial traffic behaviors
            adversarial_intensity: Intensity of adversarial behaviors [0.0, 1.0]
            config: Additional highway-env configuration overrides
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.num_lanes = num_lanes
        self.vehicles_count = vehicles_count
        self.duration = duration
        self.adversarial_mode = adversarial_mode
        self.adversarial_intensity = adversarial_intensity
        
        # Create base highway-env environment
        self._create_base_env(config)
        
        # Define observation space (5 features: rel_dist, rel_vel, ttc, ego_vel, lane_info)
        # We use normalized values for stable learning
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -30.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 30.0, 100.0, 40.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Define action space: continuous acceleration [-3, 2] m/s²
        self.action_space = spaces.Box(
            low=np.array([-3.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Metrics tracking
        self.episode_metrics = {
            "collisions": 0,
            "ttc_violations": 0,
            "avg_speed": 0.0,
            "timesteps": 0,
        }
    
    def _create_base_env(self, config: Optional[Dict[str, Any]] = None):
        """Create and configure the base highway-env environment."""
        # Base configuration for NADE compatibility
        base_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": False,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": False,  # Focus on longitudinal control
                "acceleration_range": [-3.0, 2.0],
            },
            "lanes_count": self.num_lanes,
            "vehicles_count": self.vehicles_count,
            "duration": self.duration,
            "simulation_frequency": 15,  # Hz
            "policy_frequency": 5,  # Hz (neural policy must run at 10-20 Hz)
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_lane_id": None,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1.0,
            "right_lane_reward": 0.0,  # No lateral control focus
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
        }
        
        # Apply custom config overrides
        if config:
            base_config.update(config)
        
        # Create environment
        self.env = gym.make("highway-v0", render_mode=self.render_mode)
        self.env.unwrapped.config.update(base_config)
    
    def _get_observation(self) -> np.ndarray:
        """
        Extract NADE-compatible observation from highway-env state.
        
        Returns:
            observation: [relative_distance, relative_velocity, ttc, ego_velocity, lane_occupancy]
        """
        # Extract ego vehicle state
        ego = self.env.unwrapped.vehicle
        if ego is None:
            # Vehicle not initialized yet, return default observation
            return np.array([100.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)
        
        ego_velocity = ego.speed
        
        # Find lead vehicle (closest vehicle ahead in same lane)
        lead_vehicle = None
        min_distance = float("inf")
        
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle != ego and vehicle.lane_index == ego.lane_index:
                distance = vehicle.position[0] - ego.position[0]
                if 0 < distance < min_distance:
                    min_distance = distance
                    lead_vehicle = vehicle
        
        # Calculate observation features
        if lead_vehicle is not None:
            relative_distance = lead_vehicle.position[0] - ego.position[0] - 5.0  # 5m vehicle length
            relative_velocity = lead_vehicle.speed - ego.speed
            
            # Calculate time-to-collision (TTC)
            if relative_velocity < -0.1:  # Approaching
                ttc = max(0.0, relative_distance / abs(relative_velocity))
            else:
                ttc = 100.0  # Large value if not approaching
        else:
            # No lead vehicle
            relative_distance = 100.0
            relative_velocity = 0.0
            ttc = 100.0
        
        # Check adjacent lane occupancy (simplified for now)
        lane_occupancy = 0.0
        if self.num_lanes > 1:
            # Check if there are vehicles in adjacent lanes (for future lane-change adversaries)
            for vehicle in self.env.unwrapped.road.vehicles:
                if vehicle != ego:
                    lateral_distance = abs(vehicle.lane_index[2] - ego.lane_index[2])
                    if lateral_distance == 1:  # Adjacent lane
                        longitudinal_distance = abs(vehicle.position[0] - ego.position[0])
                        if longitudinal_distance < 20.0:  # Within 20m
                            lane_occupancy = 1.0
                            break
        
        observation = np.array(
            [relative_distance, relative_velocity, ttc, ego_velocity, lane_occupancy],
            dtype=np.float32,
        )
        
        return observation
    
    def _apply_adversarial_behavior(self):
        """
        Apply adversarial behaviors to traffic vehicles if enabled.
        
        Adversarial behaviors include:
        - Sudden hard braking
        - Aggressive cut-ins
        - Unpredictable slowdowns
        """
        if not self.adversarial_mode:
            return
        
        # This is a simplified implementation
        # In practice, NADE uses trained adversarial agents
        # For now, we randomly trigger adversarial behaviors
        
        if np.random.random() < self.adversarial_intensity * 0.01:  # Low probability per timestep
            # Select a random vehicle and make it brake hard
            vehicles = [v for v in self.env.unwrapped.road.vehicles if v != self.env.unwrapped.vehicle]
            if vehicles:
                adversary = np.random.choice(vehicles)
                # Trigger hard brake by temporarily modifying its target speed
                adversary.target_speed = max(0, adversary.speed - 10)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset base environment
        self.env.reset(seed=seed, options=options)
        
        # Reset metrics
        self.episode_metrics = {
            "collisions": 0,
            "ttc_violations": 0,
            "avg_speed": 0.0,
            "timesteps": 0,
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.
        
        Args:
            action: Acceleration command [-3, 2] m/s²
        
        Returns:
            observation: Current observation
            reward: Reward value
            terminated: Whether episode is done (collision, etc.)
            truncated: Whether episode is truncated (time limit)
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply adversarial behaviors
        self._apply_adversarial_behavior()
        
        # Convert acceleration to highway-env action format
        # highway-env expects [steering, acceleration] for ContinuousAction
        highway_action = np.array([0.0, float(action[0])])
        
        # Step base environment
        _, reward, terminated, truncated, info = self.env.step(highway_action)
        
        # Get NADE-compatible observation
        observation = self._get_observation()
        
        # Update metrics
        self.episode_metrics["timesteps"] += 1
        self.episode_metrics["avg_speed"] += observation[3]  # ego_velocity
        
        if terminated and info.get("crashed", False):
            self.episode_metrics["collisions"] += 1
        
        if observation[2] < 2.0 and observation[0] < 50:  # TTC < 2s and distance < 50m
            self.episode_metrics["ttc_violations"] += 1
        
        # Add metrics to info
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        info = {
            "episode_metrics": self.episode_metrics.copy(),
            "adversarial_mode": self.adversarial_mode,
        }
        
        if self.episode_metrics["timesteps"] > 0:
            info["episode_metrics"]["avg_speed"] = (
                self.episode_metrics["avg_speed"] / self.episode_metrics["timesteps"]
            )
        
        return info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()


def create_nade_env(
    adversarial_mode: bool = False,
    render_mode: Optional[str] = None,
    **kwargs,
) -> NADEHighwayEnv:
    """
    Convenience function to create a NADE highway environment.
    
    Args:
        adversarial_mode: Enable adversarial traffic behaviors
        render_mode: Rendering mode
        **kwargs: Additional configuration parameters
    
    Returns:
        env: Configured NADEHighwayEnv instance
    
    Example:
        >>> env = create_nade_env(adversarial_mode=True)
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, truncated, info = env.step(action)
        ...     if done or truncated:
        ...         break
    """
    # Filter out non-parameter fields from config
    valid_params = {
        'num_lanes', 'vehicles_count', 'duration', 
        'adversarial_intensity', 'config'
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return NADEHighwayEnv(
        adversarial_mode=adversarial_mode,
        render_mode=render_mode,
        **filtered_kwargs,
    )

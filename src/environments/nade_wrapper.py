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
        
        # Enhanced observation space for hierarchical constrained RL (18 features):
        # [0] rel_distance_lead: relative distance to lead vehicle (m)
        # [1] rel_velocity_lead: relative velocity with lead vehicle (m/s)
        # [2] ttc_lead: time-to-collision with lead vehicle (s)
        # [3] ego_velocity: ego vehicle velocity (m/s)
        # [4] ego_lane_idx: current lane index (normalized)
        # [5] ego_lane_offset: lateral offset from lane center (m)
        # [6] right_veh_distance: distance to right-side vehicle (m)
        # [7] right_veh_rel_velocity: relative velocity with right vehicle (m/s)
        # [8] right_veh_lateral_dist: lateral distance to right vehicle (m)
        # [9] left_veh_distance: distance to left-side vehicle (m)
        # [10] left_veh_rel_velocity: relative velocity with left vehicle (m/s)
        # [11] left_veh_lateral_dist: lateral distance to left vehicle (m)
        # [12-14] lead_veh_predicted_pos: 0.5s ahead prediction [dx, dy, dv]
        # [15-17] right_veh_predicted_pos: 0.5s ahead prediction [dx, dy, dv]
        self.observation_space = spaces.Box(
            low=np.array([
                -np.inf, -30.0, 0.0, 0.0,  # lead vehicle, ego velocity
                0.0, -2.0,  # lane index, lane offset
                -np.inf, -30.0, 0.0,  # right vehicle
                -np.inf, -30.0, 0.0,  # left vehicle
                -np.inf, -5.0, -30.0,  # lead prediction
                -np.inf, -5.0, -30.0,  # right prediction
            ], dtype=np.float32),
            high=np.array([
                np.inf, 30.0, 100.0, 40.0,  # lead vehicle, ego velocity
                3.0, 2.0,  # lane index, lane offset
                np.inf, 30.0, 10.0,  # right vehicle
                np.inf, 30.0, 10.0,  # left vehicle
                np.inf, 5.0, 30.0,  # lead prediction
                np.inf, 5.0, 30.0,  # right prediction
            ], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Prediction horizon for neighbor vehicles
        self.prediction_horizon = 0.5  # seconds
        
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
        
        # Vehicle spawning parameters for continuous traffic generation
        self.spawn_interval = 150  # Spawn new vehicles every 150 steps (15 seconds at 10Hz)
        self.initial_vehicle_count = vehicles_count  # Initial vehicle count to spawn each time
        self.max_total_vehicles = vehicles_count * 3  # Maximum total vehicles allowed (to prevent overcrowding)
        self.steps_since_last_spawn = 0
    
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
            "policy_frequency": 10,  # Hz (increased from 5 to allow longer episodes)
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_lane_id": None,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1.0,
            "right_lane_reward": 0.0,  # No lateral control focus
            "high_speed_reward": 0.4,
            "reward_speed_range": [15, 20],  # Adjusted for 60 kph (16.67 m/s) traffic
            "normalize_reward": True,
            "offroad_terminal": True,
            # IDM vehicle configuration for 60 kph traffic
            "IDM_COMFORT_ACC_MAX": 1.0,  # Comfortable acceleration
            "IDM_COMFORT_ACC_MIN": -2.0,  # Comfortable deceleration
            "IDM_DESIRED_VELOCITY": 16.67,  # 60 kph in m/s
            "IDM_TIME_WANTED": 1.5,  # Desired time headway
            "IDM_DELTA": 4.0,  # Acceleration exponent
        }
        
        # Apply custom config overrides
        if config:
            base_config.update(config)
        
        # Create environment
        self.env = gym.make("highway-v0", render_mode=self.render_mode)
        self.env.unwrapped.config.update(base_config)
    
    def _get_observation(self) -> np.ndarray:
        """
        Extract enhanced observation from highway-env state.
        
        Returns:
            observation: 18-dimensional vector with:
                - Lead vehicle state (distance, velocity, TTC)
                - Ego state (velocity, lane index, lane offset)
                - Right/left vehicle states
                - 0.5s-ahead predictions for lead and right vehicles
        """
        # Extract ego vehicle state
        ego = self.env.unwrapped.vehicle
        if ego is None:
            # Vehicle not initialized yet, return default observation (18 features)
            return np.zeros(18, dtype=np.float32)
        
        ego_velocity = ego.speed
        ego_lane = ego.lane_index[2] if isinstance(ego.lane_index, tuple) else 0
        ego_position = ego.position
        
        # Get lane offset (distance from lane center)
        try:
            lane = ego.lane
            ego_lane_offset = ego.position[1] - lane.position(ego.position[0], 0)[1]
        except:
            ego_lane_offset = 0.0
        
        # Find relevant vehicles (lead, right, left)
        lead_vehicle = None
        right_vehicle = None
        left_vehicle = None
        min_lead_distance = float("inf")
        min_right_distance = float("inf")
        min_left_distance = float("inf")
        
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle == ego:
                continue
            
            vehicle_lane = vehicle.lane_index[2] if isinstance(vehicle.lane_index, tuple) else 0
            longitudinal_distance = vehicle.position[0] - ego_position[0]
            
            # Check for lead vehicle (ahead in same lane)
            if vehicle_lane == ego_lane and 0 < longitudinal_distance < min_lead_distance:
                min_lead_distance = longitudinal_distance
                lead_vehicle = vehicle
            
            # Check for right-side vehicle
            if vehicle_lane == ego_lane + 1:  # One lane to the right
                abs_long_dist = abs(longitudinal_distance)
                if abs_long_dist < 30.0 and abs_long_dist < min_right_distance:
                    min_right_distance = abs_long_dist
                    right_vehicle = vehicle
            
            # Check for left-side vehicle
            if vehicle_lane == ego_lane - 1:  # One lane to the left
                abs_long_dist = abs(longitudinal_distance)
                if abs_long_dist < 30.0 and abs_long_dist < min_left_distance:
                    min_left_distance = abs_long_dist
                    left_vehicle = vehicle
        
        # Calculate observation features for lead vehicle
        if lead_vehicle is not None:
            lead_rel_distance = lead_vehicle.position[0] - ego_position[0] - 5.0  # 5m vehicle length
            lead_rel_velocity = lead_vehicle.speed - ego_velocity
            
            # Calculate time-to-collision (TTC)
            if lead_rel_velocity < -0.1:  # Approaching
                lead_ttc = max(0.0, lead_rel_distance / abs(lead_rel_velocity))
            else:
                lead_ttc = 100.0  # Large value if not approaching
            
            # Predict lead vehicle position after prediction_horizon
            lead_pred_dx = lead_rel_distance + lead_rel_velocity * self.prediction_horizon
            lead_pred_dy = 0.0  # Assume constant lane
            lead_pred_dv = lead_rel_velocity  # Constant velocity assumption
        else:
            lead_rel_distance = 100.0
            lead_rel_velocity = 0.0
            lead_ttc = 100.0
            lead_pred_dx = 100.0
            lead_pred_dy = 0.0
            lead_pred_dv = 0.0
        
        # Calculate observation features for right-side vehicle
        if right_vehicle is not None:
            right_veh_distance = right_vehicle.position[0] - ego_position[0]
            right_veh_rel_velocity = right_vehicle.speed - ego_velocity
            right_veh_lateral_dist = abs((right_vehicle.lane_index[2] if isinstance(right_vehicle.lane_index, tuple) else 0) - ego_lane) * 4.0
            
            # Predict right vehicle position after prediction_horizon
            right_pred_dx = right_veh_distance + right_veh_rel_velocity * self.prediction_horizon
            right_pred_dy = 0.0  # Simplified: assume no lane change during prediction
            right_pred_dv = right_veh_rel_velocity
        else:
            right_veh_distance = 100.0
            right_veh_rel_velocity = 0.0
            right_veh_lateral_dist = 10.0
            right_pred_dx = 100.0
            right_pred_dy = 0.0
            right_pred_dv = 0.0
        
        # Calculate observation features for left-side vehicle
        if left_vehicle is not None:
            left_veh_distance = left_vehicle.position[0] - ego_position[0]
            left_veh_rel_velocity = left_vehicle.speed - ego_velocity
            left_veh_lateral_dist = abs((left_vehicle.lane_index[2] if isinstance(left_vehicle.lane_index, tuple) else 0) - ego_lane) * 4.0
        else:
            left_veh_distance = 100.0
            left_veh_rel_velocity = 0.0
            left_veh_lateral_dist = 10.0
        
        # Construct 18-dimensional observation
        observation = np.array([
            # Lead vehicle (3 features)
            lead_rel_distance,
            lead_rel_velocity,
            lead_ttc,
            # Ego state (3 features)
            ego_velocity,
            float(ego_lane) / max(1, self.num_lanes - 1),  # Normalized lane index
            ego_lane_offset,
            # Right vehicle (3 features)
            right_veh_distance,
            right_veh_rel_velocity,
            right_veh_lateral_dist,
            # Left vehicle (3 features)
            left_veh_distance,
            left_veh_rel_velocity,
            left_veh_lateral_dist,
            # Lead vehicle prediction (3 features)
            lead_pred_dx,
            lead_pred_dy,
            lead_pred_dv,
            # Right vehicle prediction (3 features)
            right_pred_dx,
            right_pred_dy,
            right_pred_dv,
        ], dtype=np.float32)
        
        return observation
    
    def _spawn_new_vehicles(self):
        """
        Spawn new vehicles to maintain traffic density throughout the simulation.
        This ensures continuous traffic flow rather than just initial vehicle generation.
        Vehicles are spawned at 60 kph (16.67 m/s) with small variations.
        
        Every 150 timesteps, spawns the same number of vehicles as the initial vehicle count
        to maintain consistent traffic density throughout the episode.
        Also removes vehicles that are too far from ego to prevent accumulation.
        """
        # Check if it's time to spawn
        if self.steps_since_last_spawn < self.spawn_interval:
            return
        
        # Reset spawn counter
        self.steps_since_last_spawn = 0
        
        # Get ego vehicle
        ego = self.env.unwrapped.vehicle
        if ego is None:
            return
        
        road = self.env.unwrapped.road
        
        # Clean up vehicles that are too far from ego (>250m behind or >300m ahead)
        # This prevents vehicle accumulation and maintains consistent traffic density
        # Note: We use a smaller distance for removal to ensure we don't remove visible traffic
        vehicles_to_remove = []
        for vehicle in road.vehicles:
            if vehicle == ego:
                continue
            
            longitudinal_distance = vehicle.position[0] - ego.position[0]
            
            # Remove vehicles that are too far behind or ahead
            # These thresholds ensure vehicles are only removed when truly out of range
            if longitudinal_distance < -250.0 or longitudinal_distance > 300.0:
                vehicles_to_remove.append(vehicle)
        
        # Remove far-away vehicles
        for vehicle in vehicles_to_remove:
            try:
                road.vehicles.remove(vehicle)
            except ValueError:
                # Vehicle already removed, skip
                pass
        
        # Count current vehicles (excluding ego)
        current_vehicles = [v for v in road.vehicles if v != ego]
        
        # Spawn the same number as initial vehicle count
        vehicles_needed = self.initial_vehicle_count
        
        try:
            # Import vehicle classes
            from highway_env.vehicle.behavior import IDMVehicle
            from highway_env.road.road import Road, RoadNetwork
            
            road = self.env.unwrapped.road
            
            # Spawn vehicles ahead AND behind ego vehicle in random lanes
            for i in range(vehicles_needed):
                # Choose a random lane
                lane_idx = np.random.randint(0, self.num_lanes)
                
                # Alternate spawning ahead and behind for better distribution
                if i % 2 == 0:
                    # Spawn ahead of ego vehicle (100-200m ahead)
                    spawn_distance = ego.position[0] + np.random.uniform(100, 200)
                else:
                    # Spawn behind ego vehicle (80-150m behind)
                    spawn_distance = ego.position[0] - np.random.uniform(80, 150)
                
                # Target speed: 60 kph (16.67 m/s) with minimal variation to maintain speed
                # Tighter range to prevent speed decay
                target_speed = np.random.uniform(16.0, 17.3)  # 57.6-62.3 kph
                
                # Get the lane
                try:
                    # Create new vehicle using highway-env's make_on_lane
                    new_vehicle = IDMVehicle.make_on_lane(
                        road=road,
                        lane_index=("0", "1", lane_idx),
                        longitudinal=spawn_distance,
                        speed=target_speed,
                    )
                    
                    # Set IDM parameters for consistent 60 kph driving behavior
                    # These parameters ensure spawned vehicles maintain their speed
                    new_vehicle.target_speed = target_speed
                    if hasattr(new_vehicle, 'COMFORT_ACC_MAX'):
                        new_vehicle.COMFORT_ACC_MAX = 1.0  # Comfortable acceleration
                    if hasattr(new_vehicle, 'COMFORT_ACC_MIN'):
                        new_vehicle.COMFORT_ACC_MIN = -2.0  # Comfortable braking
                    if hasattr(new_vehicle, 'TIME_WANTED'):
                        new_vehicle.TIME_WANTED = 1.5  # Desired time headway (seconds)
                    if hasattr(new_vehicle, 'DELTA'):
                        new_vehicle.DELTA = 4.0  # Acceleration exponent
                    if hasattr(new_vehicle, 'DESIRED_VELOCITY'):
                        new_vehicle.DESIRED_VELOCITY = 16.67  # 60 kph in m/s
                    
                    # Check if spawn position is safe (no collision with existing vehicles)
                    safe_spawn = True
                    for vehicle in current_vehicles:
                        if vehicle == ego:
                            continue
                        distance = abs(vehicle.position[0] - new_vehicle.position[0])
                        lateral_dist = abs(vehicle.position[1] - new_vehicle.position[1])
                        
                        # Check both longitudinal and lateral distance
                        if distance < 20.0 and lateral_dist < 3.0:  # Too close
                            safe_spawn = False
                            break
                    
                    # Add vehicle if safe
                    if safe_spawn:
                        road.vehicles.append(new_vehicle)
                
                except (AttributeError, IndexError, KeyError, ValueError) as e:
                    # Lane doesn't exist or other error, skip this spawn
                    pass
        
        except ImportError:
            # highway_env not available or different structure
            pass
    
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
        
        # TODO: fix this afterwards
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
        
        # Reset vehicle spawning counter
        self.steps_since_last_spawn = 0
        
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
        
        # Spawn new vehicles to maintain traffic density
        self._spawn_new_vehicles()
        self.steps_since_last_spawn += 1
        
        # Apply adversarial behaviors
        self._apply_adversarial_behavior()
        
        # Convert acceleration from [-3, 2] m/s² to normalized action [-1, 1]
        # highway-env's ContinuousAction expects normalized values
        # Normalize: action_norm = (action - min) / (max - min) * 2 - 1
        # For acceleration range [-3, 2]: min=-3, max=2, range=5
        accel_value = float(action[0])
        normalized_accel = ((accel_value - (-3.0)) / 5.0) * 2.0 - 1.0
        normalized_accel = np.clip(normalized_accel, -1.0, 1.0)
        
        # highway-env expects [steering, acceleration] for ContinuousAction
        highway_action = np.array([0.0, normalized_accel])
        
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

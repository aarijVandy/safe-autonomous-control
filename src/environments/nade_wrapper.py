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
        
        # === TRAINING CURRICULUM PARAMETERS ===
        self.episode_count = 0  # Track total episodes
        self.LATERAL_UNLOCK_EPISODE = 0  # Freeze lateral control for first 200 episodes
        self.ADVERSARIAL_UNLOCK_EPISODE = 10  # Delay adversarial traffic until episode 300
        self.ttc_violations_history = []  # Track TTC violations for performance-based unlock
        
        # Lane change state machine - MUCH MORE CONSERVATIVE
        self.LANE_CHANGE_IDLE = 0
        self.LANE_CHANGE_EXECUTING = 1
        self.LANE_CHANGE_COOLDOWN = 2
        
        self.lane_change_state = self.LANE_CHANGE_IDLE
        self.lane_change_timer = 0
        self.lane_change_duration = 20  # 2 seconds at 10Hz - much slower, safer lane changes
        self.lane_change_cooldown_duration = 50  # 5 seconds cooldown - prevent rapid lane switching
        self.target_lane = None
        self.lane_change_start_lane = None
        self.lane_change_aborted = False  # Track if lane change was aborted for safety
        
        # === STEERING LIMITS ===
        self.MAX_STEERING = 0.15  # Increased to allow meaningful lane changes
        
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
        
        # Define action space: 2D [acceleration, lane_change_command]
        # acceleration: continuous [-3, 2] m/s²
        # lane_change_command: continuous [-1, 1] where:
        #   < -0.33: change left
        #   > 0.33: change right
        #   [-0.33, 0.33]: keep lane (small corrections allowed)
        self.action_space = spaces.Box(
            low=np.array([-3.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32),
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
                "lateral": True,  # Enable lateral control for lane changes
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
        
        NOTE: Delayed until episode ADVERSARIAL_UNLOCK_EPISODE for curriculum learning.
        """
        # === CURRICULUM LEARNING: DELAY ADVERSARIAL TRAFFIC ===
        # Don't apply adversarial behaviors until agent has learned stable driving
        if self.episode_count < self.ADVERSARIAL_UNLOCK_EPISODE:
            return
        
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
        
        # Increment episode counter for curriculum learning
        self.episode_count += 1
        
        # Reset metrics
        self.episode_metrics = {
            "collisions": 0,
            "ttc_violations": 0,
            "avg_speed": 0.0,
            "timesteps": 0,
        }
        
        # Reset vehicle spawning counter
        self.steps_since_last_spawn = 0
        
        # === CRITICAL: Reset per-episode timestep counter ===
        self.episode_timesteps = 0  # Track timesteps within this episode
        
        # Reset lane change state machine
        self.lane_change_state = self.LANE_CHANGE_IDLE
        self.lane_change_timer = 0
        self.target_lane = None
        self.lane_change_start_lane = None
        
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
        
        # Increment per-episode timestep counter
        self.episode_timesteps += 1
        
        # Spawn new vehicles to maintain traffic density
        self._spawn_new_vehicles()
        self.steps_since_last_spawn += 1
        
        # Apply adversarial behaviors
        self._apply_adversarial_behavior()
        
        # Extract actions: [acceleration, lane_change_command]
        accel_value = float(action[0])  # [-3, 2] m/s²
        lane_cmd = float(action[1])     # [-1, 1] continuous
        
        # === CRITICAL: FREEZE ALL STEERING FOR FIRST 100 TIMESTEPS ===
        # This prevents ANY turning in early timesteps to learn pure longitudinal control
        FREEZE_STEERING_TIMESTEPS = 0
        steering_frozen = self.episode_timesteps < FREEZE_STEERING_TIMESTEPS
        
        # === CURRICULUM LEARNING: FREEZE LATERAL CONTROL ===
        # Force lateral control to zero for first N episodes to learn longitudinal stability
        if self.episode_count < self.LATERAL_UNLOCK_EPISODE or steering_frozen:
            lane_cmd = 0.0  # Disable lateral action entirely
            # Agent must learn safe following, speed control, and stability first
        
        # Convert acceleration from [-3, 2] m/s² to normalized action [-1, 1]
        # highway-env's ContinuousAction expects normalized values
        # Normalize: action_norm = (action - min) / (max - min) * 2 - 1
        # For acceleration range [-3, 2]: min=-3, max=2, range=5
        normalized_accel = ((accel_value - (-3.0)) / 5.0) * 2.0 - 1.0
        normalized_accel = np.clip(normalized_accel, -1.0, 1.0)
        
        # Get current lane to prevent going off-road
        ego = self.env.unwrapped.vehicle
        if ego is not None:
            ego_lane_idx = ego.lane_index[2] if isinstance(ego.lane_index, tuple) else 0
            num_lanes = self.num_lanes
        else:
            ego_lane_idx = 1  # Default to middle lane if ego not available
            num_lanes = self.num_lanes
        
        # STRICT LANE BOUNDARY ENFORCEMENT - clamp to valid range
        if ego_lane_idx < 0:
            ego_lane_idx = 0
            # Force vehicle back to valid lane
            if ego is not None and hasattr(ego, 'lane_index') and isinstance(ego.lane_index, tuple):
                ego.lane_index = (ego.lane_index[0], ego.lane_index[1], 0)
        elif ego_lane_idx >= num_lanes:
            ego_lane_idx = num_lanes - 1
            # Force vehicle back to valid lane
            if ego is not None and hasattr(ego, 'lane_index') and isinstance(ego.lane_index, tuple):
                ego.lane_index = (ego.lane_index[0], ego.lane_index[1], num_lanes - 1)
        
        # Get observation for safety checks and assistive control
        observation = self._get_observation()
        lane_offset = observation[5]  # Lateral offset from lane center
        right_veh_distance = observation[6]
        left_veh_distance = observation[9]
        
        # Lane change state machine with gradual steering and SAFETY ABORT
        steering = 0.0
        agent_steering = 0.0  # Track agent's steering request separately
        
        # === ABSOLUTE STEERING FREEZE FOR FIRST 100 TIMESTEPS ===
        if steering_frozen:
            # Agent lateral control is FROZEN for first 100 timesteps
            # BUT: Apply VERY strong stabilization to keep vehicle in lane center
            # Use a PD controller with aggressive gains to prevent any lane departure
            k_p = 0.35  # VERY strong proportional gain for rapid centering
            k_d = 0.20  # VERY strong derivative gain to kill lateral velocity
            lateral_velocity = ego.velocity[1] if ego is not None and hasattr(ego, 'velocity') else 0.0
            steering = -k_p * lane_offset - k_d * lateral_velocity
            # Allow very large steering during freeze - safety through lane-keeping
            # This is safe because it aggressively centers the vehicle
            steering = np.clip(steering, -0.3, 0.3)  # Very aggressive limit for stabilization
            agent_steering = 0.0  # Agent has no control
        elif self.lane_change_state == self.LANE_CHANGE_IDLE:
            # === LANE-CENTERING ASSISTIVE CONTROLLER ===
            # Add PD controller to naturally return vehicle to lane center
            k_p = 0.02  # Proportional gain
            k_d = 0.01  # Derivative gain
            lateral_velocity = ego.velocity[1] if ego is not None and hasattr(ego, 'velocity') else 0.0
            assistive_steering = -k_p * lane_offset - k_d * lateral_velocity
            
            # Combine agent steering with assistive steering
            # Removed 0.1 multiplier to allow meaningful steering from agent
            agent_steering = np.clip(lane_cmd * 0.4, -0.2, 0.2)  # Allow reasonable steering range
            steering = agent_steering + assistive_steering
            
            # === HARD SAFETY FILTER: BLOCK STEERING NEAR LANE BOUNDARIES ===
            if abs(lane_offset) > 2.2:  # Relaxed from 1.6 to allow more maneuvering
                steering = 0.0  # Cancel all steering to prevent going off-road
            
            # Not currently changing lanes - check if new lane change is requested
            # VERY CONSERVATIVE: Only respond to strong lane change commands (> 0.7 threshold)
            if lane_cmd < -0.7 and ego_lane_idx > 0:
                # Request to change LEFT - COMPREHENSIVE SAFETY CHECKS
                lead_distance = observation[0]
                lead_ttc = observation[2]
                
                # CRITICAL SAFETY CHECK: Cancel if surrounding vehicles too close
                if left_veh_distance < 12.0:  # Vehicle too close on left
                    steering = 0.0  # REJECT - unsafe gap
                # Only allow lane change if current situation is safe
                elif lead_distance > 40 or lead_ttc > 5.0:
                    self.lane_change_state = self.LANE_CHANGE_EXECUTING
                    self.lane_change_timer = 0
                    self.target_lane = ego_lane_idx - 1
                    self.lane_change_start_lane = ego_lane_idx
                    self.lane_change_aborted = False
                    steering = -self.MAX_STEERING  # Use conservative steering limit
                else:
                    steering = 0.0  # Unsafe current lane - reject lane change
            elif lane_cmd > 0.7 and ego_lane_idx < num_lanes - 1:
                # Request to change RIGHT - COMPREHENSIVE SAFETY CHECKS
                lead_distance = observation[0]
                lead_ttc = observation[2]
                
                # CRITICAL SAFETY CHECK: Cancel if surrounding vehicles too close
                if right_veh_distance < 12.0:  # Vehicle too close on right
                    steering = 0.0  # REJECT - unsafe gap
                # Only allow lane change if current situation is safe
                elif lead_distance > 40 or lead_ttc > 5.0:
                    self.lane_change_state = self.LANE_CHANGE_EXECUTING
                    self.lane_change_timer = 0
                    self.target_lane = ego_lane_idx + 1
                    self.lane_change_start_lane = ego_lane_idx
                    self.lane_change_aborted = False
                    steering = self.MAX_STEERING  # Use conservative steering limit
                else:
                    steering = 0.0  # Unsafe current lane - reject lane change
        
        elif self.lane_change_state == self.LANE_CHANGE_EXECUTING:
            # Currently executing a lane change - CONTINUOUS SAFETY MONITORING
            self.lane_change_timer += 1
            progress = min(1.0, self.lane_change_timer / self.lane_change_duration)
            
            # COMPREHENSIVE SAFETY ABORT CHECKS
            lead_distance = observation[0]
            lead_ttc = observation[2]
            
            # Abort if ANY of these conditions occur:
            # 1. Lead vehicle becomes dangerously close
            if lead_distance > 0 and lead_distance < 30 and lead_ttc < 3.0:
                self.lane_change_state = self.LANE_CHANGE_COOLDOWN
                self.lane_change_timer = 0
                self.lane_change_aborted = True
                steering = 0.0
            # 2. Surrounding vehicles become too close during lane change
            elif right_veh_distance < 12.0 or left_veh_distance < 12.0:
                self.lane_change_state = self.LANE_CHANGE_COOLDOWN
                self.lane_change_timer = 0
                self.lane_change_aborted = True
                steering = 0.0
            # 3. Vehicle drifting too far from lane center (about to go off-road)
            elif abs(lane_offset) > 2.0:
                self.lane_change_state = self.LANE_CHANGE_COOLDOWN
                self.lane_change_timer = 0
                self.lane_change_aborted = True
                steering = 0.0
            # Check if we've reached the target lane
            elif ego_lane_idx == self.target_lane:
                self.lane_change_state = self.LANE_CHANGE_COOLDOWN
                self.lane_change_timer = 0
                steering = 0.0
            elif self.lane_change_timer >= self.lane_change_duration:
                self.lane_change_state = self.LANE_CHANGE_COOLDOWN
                self.lane_change_timer = 0
                steering = 0.0
            else:
                # Continue with VERY CONSERVATIVE steering
                direction = 1.0 if self.target_lane > self.lane_change_start_lane else -1.0
                # Use sine wave with MAX_STEERING limit 
                steering_magnitude = self.MAX_STEERING * np.sin(progress * np.pi)
                steering = direction * steering_magnitude
        
        elif self.lane_change_state == self.LANE_CHANGE_COOLDOWN:
            # In cooldown period - no lane changes allowed
            self.lane_change_timer += 1
            steering = 0.0  # Keep lane strictly
            
            if self.lane_change_timer >= self.lane_change_cooldown_duration:
                # Cooldown complete - return to idle
                self.lane_change_state = self.LANE_CHANGE_IDLE
                self.lane_change_timer = 0
                self.lane_change_aborted = False
        
        # === FINAL SAFETY CLAMP: Enforce maximum steering limit ===
        # During steering freeze, stabilization is already clamped to [-0.15, 0.15]
        # After freeze, apply normal conservative steering limits
        if not steering_frozen:
            steering = np.clip(steering, -self.MAX_STEERING, self.MAX_STEERING)
        
        # CRITICAL: highway-env ContinuousAction expects [acceleration, steering] order!
        # NOT [steering, acceleration] as you might expect
        # See highway_env/envs/common/action.py:ContinuousAction.get_action()
        highway_action = np.array([normalized_accel, steering])
        
        # Step base environment
        _, base_reward, terminated, truncated, info = self.env.step(highway_action)
        
        # Add lane change state to info for debugging
        info['lane_change_state'] = self.lane_change_state
        info['lane_change_timer'] = self.lane_change_timer
        info['target_lane'] = self.target_lane
        info['current_lane'] = ego_lane_idx
        info['steering'] = steering
        info['lane_cmd'] = lane_cmd
        info['episode_timesteps'] = self.episode_timesteps
        info['steering_frozen'] = steering_frozen
        
        # Get NADE-compatible observation
        observation = self._get_observation()
        
        # === Apply MODERATE Penalties for Unsafe Behaviors ===
        # Use smaller penalties that discourage violations but don't destroy learning signal
        reward = base_reward
        penalty_applied = False
        
        # === STRONG PENALTY FOR LARGE LATERAL ACCELERATIONS (PREVENT ZIG-ZAG) ===
        if abs(steering) > 0.03:
            lateral_penalty = -1.0 * abs(steering)
            reward += lateral_penalty
            info['lateral_accel_penalty'] = lateral_penalty
        
        # === STRONG LANE-CENTER REWARD ===
        # Exponential reward for staying near lane center
        lane_offset = observation[5]  # ego_lane_offset
        lane_center_reward = 1.5 * np.exp(-abs(lane_offset) * 0.7)
        reward += lane_center_reward
        info['lane_center_reward'] = lane_center_reward
        
        # LANE CHANGE PENALTY - Apply during lane change execution
        if self.lane_change_state == self.LANE_CHANGE_EXECUTING:
            # Continuous penalty throughout lane change to discourage it
            LANE_PENALTY = 2.0  # INCREASED from 0.5 to strongly discourage
            reward -= LANE_PENALTY  # -2.0 per timestep during lane change
            info['lane_change_execution_penalty'] = -LANE_PENALTY

        # Check for collision - TERMINATE EPISODE IMMEDIATELY
        if info.get("crashed", False):
            self.episode_metrics["collisions"] += 1
            reward = -100.0  # Large penalty for collision
            # TERMINATE episode on collision
            terminated = True
            penalty_applied = True
            info['collision_penalty'] = True
            info['termination_reason'] = 'collision'
        
        # Check for going off-road / out of highway bounds - TERMINATE EPISODE
        if ego is not None and not penalty_applied:
            vehicle_lane_idx = ego.lane_index[2] if isinstance(ego.lane_index, tuple) else 0
            # Check if lane index is outside valid range [0, num_lanes-1]
            if vehicle_lane_idx < 0 or vehicle_lane_idx >= self.num_lanes:
                reward = -100.0  # Large penalty for going off-road
                # TERMINATE episode when going off-road
                terminated = True
                penalty_applied = True
                info['off_road_penalty'] = True
                info['termination_reason'] = 'off_road'
                self.episode_metrics['collisions'] += 1
        
        # Check for dangerous lane offset (too far from lane center)
        if not penalty_applied:  # Only if no major penalty already applied
            lane_offset = observation[5]  # ego_lane_offset
            if abs(lane_offset) > 2.5:  # More than 2.5m from center (approaching edge)
                # REDUCED penalty - moderate discouragement
                offset_penalty = -10.0 * (abs(lane_offset) - 2.5)  # Max ~-15 for 4m offset
                reward += offset_penalty
                info['lane_offset_penalty'] = offset_penalty
        
        # Penalize critically low TTC (near-collision) more heavily
        if not penalty_applied:  # Only if no major penalty already applied
            ttc = observation[2]
            lead_distance = observation[0]
            if ttc < 1.0 and lead_distance > 0 and lead_distance < 30:  # Critical: TTC < 1s and very close
                # REDUCED penalty for near-collision
                critical_penalty = -30.0 * (1.0 - ttc)  # Max -30 for TTC=0
                reward += critical_penalty
                info['critical_ttc_penalty'] = critical_penalty
            elif ttc < 2.0 and lead_distance > 0 and lead_distance < 50:  # Warning: TTC < 2s
                # Moderate penalty for dangerous following
                warning_penalty = -50.0 * (2.0 - ttc)
                reward += warning_penalty
                info['ttc_warning_penalty'] = warning_penalty
        
        # === SURVIVAL BONUS - Reward for staying alive ===
        # If no penalty was applied (no crash, no off-road), give survival bonus
        if not penalty_applied:
            survival_bonus = 1  # per timestep for staying safe
            reward += survival_bonus
            info['survival_bonus'] = survival_bonus
            
            # Extra bonus for being in a valid lane and well-centered
            lane_offset = observation[5]
            if abs(lane_offset) < 1.5:  # Well within lane
                centering_bonus = 0.3
                reward += centering_bonus
                info['centering_bonus'] = centering_bonus
        
        # Update metrics
        self.episode_metrics["timesteps"] += 1
        self.episode_metrics["avg_speed"] += observation[3]  # ego_velocity
        
        if observation[2] < 2.0 and observation[0] < 50:  # TTC < 2s and distance < 50m
            self.episode_metrics["ttc_violations"] += 1
        
        # Track TTC violations for performance-based curriculum
        # Can be used to unlock lateral control based on performance instead of episodes
        if terminated or truncated:
            violation_rate = (self.episode_metrics["ttc_violations"] / 
                            max(1, self.episode_metrics["timesteps"]))
            self.ttc_violations_history.append(violation_rate)
            # Keep only last 50 episodes
            if len(self.ttc_violations_history) > 50:
                self.ttc_violations_history.pop(0)
        
        # Add metrics to info (including curriculum status)
        info.update(self._get_info())
        info['episode_count'] = self.episode_count
        info['lateral_control_enabled'] = self.episode_count >= self.LATERAL_UNLOCK_EPISODE
        info['adversarial_enabled'] = self.episode_count >= self.ADVERSARIAL_UNLOCK_EPISODE
        
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

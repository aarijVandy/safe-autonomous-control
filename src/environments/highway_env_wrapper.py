"""
Highway Environment Wrapper with Adversarial Scenario Support.

This module wraps the highway-env environment to support adversarial
scenario generation and enhanced metrics collection.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .adversarial_scenarios import (
    AdversarialScenarioGenerator,
    ScenarioMetrics,
    AdversarialEventType,
)


class HighwayEnvWrapper(gym.Wrapper):
    """
    Wrapper for highway-env with adversarial scenario support.
    
    This wrapper:
    - Integrates with AdversarialScenarioGenerator
    - Applies adversarial events during episodes
    - Collects enhanced metrics
    - Provides consistent interface for RL training
    """
    
    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        scenario_generator: Optional[AdversarialScenarioGenerator] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            env_config: Configuration for highway-env
            scenario_generator: Adversarial scenario generator
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        # Default environment configuration
        if env_config is None:
            env_config = self._get_default_config()
        
        # Create base environment
        base_env = gym.make('highway-v0', render_mode=render_mode, **env_config)
        super().__init__(base_env)
        
        # Scenario generator
        if scenario_generator is None:
            scenario_generator = AdversarialScenarioGenerator(mode="normal")
        self.scenario_generator = scenario_generator
        
        # Current scenario
        self.current_scenario = None
        self.scenario_metrics = ScenarioMetrics()
        
        # Adversarial event tracking
        self.active_events = []
        self.current_timestep = 0
        
        # Previous state for computing derived metrics
        self.prev_action = None
        self.prev_position = None
        self.prev_lane = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default highway-env configuration."""
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": False,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": False,
            },
            "lanes_count": 4,
            "vehicles_count": 10,
            "duration": 40,  # seconds
            "simulation_frequency": 15,  # Hz
            "policy_frequency": 5,  # Hz
            "collision_reward": -1.0,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with new scenario.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Reset metrics
        self.scenario_metrics.reset()
        self.current_timestep = 0
        self.active_events = []
        self.prev_action = None
        self.prev_position = None
        self.prev_lane = None
        
        # Sample new scenario
        self.current_scenario = self.scenario_generator.sample_scenario()
        
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Apply scenario configuration to environment
        self._apply_scenario_config()
        
        # Add scenario info
        info["scenario"] = {
            "is_adversarial": self.current_scenario.is_adversarial,
            "difficulty": self.current_scenario.difficulty.value,
            "num_vehicles": self.current_scenario.num_vehicles,
            "traffic_density": self.current_scenario.traffic_density,
            "num_adversarial_events": len(self.current_scenario.adversarial_events),
        }
        
        return obs, info
    
    def _apply_scenario_config(self):
        """Apply scenario configuration to the environment."""
        if not hasattr(self.env.unwrapped, 'road'):
            return
        
        road = self.env.unwrapped.road
        
        # Set vehicle speeds (if accessible)
        if hasattr(road, 'vehicles') and len(road.vehicles) > 0:
            for i, vehicle in enumerate(road.vehicles[:len(self.current_scenario.vehicle_speeds)]):
                target_speed = self.current_scenario.vehicle_speeds[i]
                if hasattr(vehicle, 'target_speed'):
                    vehicle.target_speed = target_speed
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with adversarial event handling.
        
        Args:
            action: Action to execute
        
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_timestep += 1
        
        # Check and trigger adversarial events
        self._check_adversarial_events()
        
        # Execute action in base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Compute additional metrics
        self._update_metrics(obs, action, reward, info)
        
        # Add scenario metrics to info
        if terminated or truncated:
            info["scenario_metrics"] = self.scenario_metrics.get_summary()
            info["scenario_info"] = {
                "is_adversarial": self.current_scenario.is_adversarial,
                "difficulty": self.current_scenario.difficulty.value,
                "events_triggered": len(self.active_events),
            }
        
        return obs, reward, terminated, truncated, info
    
    def _check_adversarial_events(self):
        """Check if any adversarial events should be triggered."""
        if not self.current_scenario or not self.current_scenario.adversarial_events:
            return
        
        # Convert timestep to seconds
        dt = 1.0 / self.env.unwrapped.config["policy_frequency"]
        current_time = self.current_timestep * dt
        
        # Check each event
        for event in self.current_scenario.adversarial_events:
            if event in self.active_events:
                continue
            
            # Trigger if time has come
            if current_time >= event.trigger_time:
                self._trigger_event(event)
                self.active_events.append(event)
    
    def _trigger_event(self, event):
        """
        Trigger an adversarial event.
        
        Args:
            event: AdversarialEvent to trigger
        """
        if not hasattr(self.env.unwrapped, 'road'):
            return
        
        road = self.env.unwrapped.road
        if event.target_vehicle_idx >= len(road.vehicles):
            return
        
        target_vehicle = road.vehicles[event.target_vehicle_idx]
        
        # Apply event based on type
        if event.event_type == AdversarialEventType.AGGRESSIVE_CUT_IN:
            # Force lane change with minimal gap
            params = event.parameters
            if hasattr(target_vehicle, 'act'):
                # Trigger aggressive lane change behavior
                target_lane = target_vehicle.lane_index[2] + params["target_lane_offset"]
                target_vehicle.target_lane_index = (
                    target_vehicle.lane_index[0],
                    target_vehicle.lane_index[1],
                    np.clip(target_lane, 0, len(road.network.graph[target_vehicle.lane_index[0]][target_vehicle.lane_index[1]]) - 1)
                )
        
        elif event.event_type == AdversarialEventType.BRAKE_CHECK:
            # Apply sudden braking
            params = event.parameters
            if hasattr(target_vehicle, 'act'):
                # Override vehicle behavior to brake hard
                target_vehicle.target_speed = max(0, target_vehicle.speed + params["deceleration"])
        
        # Note: Full implementation would require modifying vehicle behavior
        # This is a simplified version showing the structure
    
    def _update_metrics(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any],
    ):
        """
        Update scenario metrics.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            info: Step information
        """
        # Check for collision
        collision = info.get("crashed", False) or reward == -1.0
        
        # Estimate TTC from observation (if using kinematics observation)
        ttc = float('inf')
        if len(obs.shape) == 2 and obs.shape[1] >= 5:
            # obs shape: (num_vehicles, features)
            # features: [presence, x, y, vx, vy]
            for i in range(1, len(obs)):  # Skip ego vehicle
                if obs[i, 0] > 0:  # Vehicle present
                    rel_x = obs[i, 1]
                    rel_vx = obs[i, 3]
                    if rel_x > 0 and rel_vx < 0:  # Vehicle ahead and approaching
                        vehicle_ttc = -rel_x / rel_vx
                        ttc = min(ttc, vehicle_ttc)
        
        # Compute jerk if we have previous action
        jerk = 0.0
        if self.prev_action is not None:
            jerk = np.abs(action[0] - self.prev_action[0])
        
        # Update metrics
        self.scenario_metrics.update(
            collision=collision,
            ttc=ttc,
            ttc_threshold=2.0,
            jerk=jerk,
        )
        
        self.prev_action = action
    
    def get_scenario_stats(self) -> Dict[str, Any]:
        """Get scenario generator statistics."""
        return self.scenario_generator.get_stats()
    
    def set_mode(self, mode: str):
        """
        Change scenario generation mode.
        
        Args:
            mode: New mode ('normal', 'adversarial', 'mixed')
        """
        self.scenario_generator.mode = mode
    
    def set_difficulty(self, difficulty: str):
        """
        Change difficulty level.
        
        Args:
            difficulty: New difficulty ('easy', 'medium', 'hard', 'extreme')
        """
        from .adversarial_scenarios import DifficultyLevel
        self.scenario_generator.difficulty = DifficultyLevel(difficulty)
        self.scenario_generator.params = self.scenario_generator.DIFFICULTY_PARAMS[
            self.scenario_generator.difficulty
        ]
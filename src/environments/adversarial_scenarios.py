"""
Adversarial Scenario Generator for Highway Driving.

This module defines adversarial events and scenarios for testing and training
robust autonomous driving controllers.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


class AdversarialEventType(Enum):
    """Types of adversarial events."""
    AGGRESSIVE_CUT_IN = "aggressive_cut_in"
    BRAKE_CHECK = "brake_check"
    ERRATIC_LANE_CHANGE = "erratic_lane_change"
    SPEED_VARIANCE = "speed_variance"
    MULTI_VEHICLE_CONFLICT = "multi_vehicle_conflict"
    SUDDEN_SLOWDOWN = "sudden_slowdown"
    UNSAFE_MERGE = "unsafe_merge"


class DifficultyLevel(Enum):
    """Difficulty levels for scenarios."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class AdversarialEvent:
    """Configuration for a single adversarial event."""
    event_type: AdversarialEventType
    trigger_time: float  # When to trigger (seconds into episode)
    duration: float  # How long the event lasts
    severity: float  # Event intensity [0, 1]
    target_vehicle_idx: int  # Which vehicle performs the action
    parameters: Dict[str, Any]  # Event-specific parameters


@dataclass
class ScenarioConfig:
    """Configuration for a complete scenario."""
    num_vehicles: int
    traffic_density: str  # "low", "medium", "high"
    vehicle_speeds: List[float]  # Initial speeds for each vehicle
    vehicle_positions: List[Tuple[float, int]]  # (longitudinal, lane) for each vehicle
    adversarial_events: List[AdversarialEvent]
    is_adversarial: bool
    difficulty: DifficultyLevel


class AdversarialScenarioGenerator:
    """
    Generator for normal and adversarial driving scenarios.
    
    This class creates scenarios with varying difficulty levels, from normal
    highway driving to challenging adversarial situations.
    """
    
    # Difficulty parameters
    DIFFICULTY_PARAMS = {
        DifficultyLevel.EASY: {
            "adversarial_ratio": 0.2,
            "event_severity_range": (0.2, 0.4),
            "event_frequency": 0.1,  # events per 100 timesteps
            "min_ttc": 3.0,
            "cut_in_gap_range": (12, 20),  # meters
            "brake_decel_range": (-4, -3),  # m/s²
        },
        DifficultyLevel.MEDIUM: {
            "adversarial_ratio": 0.5,
            "event_severity_range": (0.4, 0.7),
            "event_frequency": 0.3,
            "min_ttc": 2.0,
            "cut_in_gap_range": (8, 15),
            "brake_decel_range": (-6, -4),
        },
        DifficultyLevel.HARD: {
            "adversarial_ratio": 0.7,
            "event_severity_range": (0.6, 0.9),
            "event_frequency": 0.5,
            "min_ttc": 1.5,
            "cut_in_gap_range": (5, 12),
            "brake_decel_range": (-8, -5),
        },
        DifficultyLevel.EXTREME: {
            "adversarial_ratio": 0.9,
            "event_severity_range": (0.8, 1.0),
            "event_frequency": 0.7,
            "min_ttc": 1.0,
            "cut_in_gap_range": (3, 8),
            "brake_decel_range": (-10, -7),
        },
    }
    
    def __init__(
        self,
        mode: str = "normal",  # "normal", "adversarial", "mixed"
        difficulty: str = "medium",
        adversarial_ratio: float = 0.5,  # For mixed mode
        episode_length: int = 1000,  # timesteps
        seed: Optional[int] = None,
    ):
        """
        Initialize the scenario generator.
        
        Args:
            mode: Generation mode - "normal", "adversarial", or "mixed"
            difficulty: Difficulty level - "easy", "medium", "hard", "extreme"
            adversarial_ratio: Ratio of adversarial scenarios in mixed mode
            episode_length: Length of episodes in timesteps
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.difficulty = DifficultyLevel(difficulty)
        self.adversarial_ratio = adversarial_ratio
        self.episode_length = episode_length
        
        self.rng = np.random.default_rng(seed)
        
        # Get difficulty parameters
        self.params = self.DIFFICULTY_PARAMS[self.difficulty]
        
        # Statistics
        self.scenarios_generated = 0
        self.adversarial_scenarios_generated = 0
    
    def sample_scenario(self) -> ScenarioConfig:
        """
        Sample a scenario configuration.
        
        Returns:
            ScenarioConfig with vehicle setup and adversarial events
        """
        self.scenarios_generated += 1
        
        # Determine if this should be adversarial
        if self.mode == "normal":
            is_adversarial = False
        elif self.mode == "adversarial":
            is_adversarial = True
        else:  # mixed
            is_adversarial = self.rng.random() < self.adversarial_ratio
        
        if is_adversarial:
            self.adversarial_scenarios_generated += 1
        
        # Generate traffic density
        if is_adversarial:
            traffic_density = self.rng.choice(["medium", "high"], p=[0.4, 0.6])
        else:
            traffic_density = self.rng.choice(["low", "medium"], p=[0.6, 0.4])
        
        # Sample number of vehicles based on density
        if traffic_density == "low":
            num_vehicles = self.rng.integers(3, 6)
        elif traffic_density == "medium":
            num_vehicles = self.rng.integers(6, 10)
        else:  # high
            num_vehicles = self.rng.integers(10, 15)
        
        # Generate vehicle initial conditions
        vehicle_speeds = self._generate_vehicle_speeds(num_vehicles, is_adversarial)
        vehicle_positions = self._generate_vehicle_positions(num_vehicles, traffic_density)
        
        # Generate adversarial events if applicable
        adversarial_events = []
        if is_adversarial:
            adversarial_events = self._generate_adversarial_events(num_vehicles)
        
        return ScenarioConfig(
            num_vehicles=num_vehicles,
            traffic_density=traffic_density,
            vehicle_speeds=vehicle_speeds,
            vehicle_positions=vehicle_positions,
            adversarial_events=adversarial_events,
            is_adversarial=is_adversarial,
            difficulty=self.difficulty,
        )
    
    def _generate_vehicle_speeds(
        self, num_vehicles: int, is_adversarial: bool
    ) -> List[float]:
        """
        Generate initial speeds for vehicles.
        
        Args:
            num_vehicles: Number of vehicles in scenario
            is_adversarial: Whether this is an adversarial scenario
        
        Returns:
            List of initial speeds in m/s
        """
        if is_adversarial:
            # More speed variance in adversarial scenarios
            mean_speed = self.rng.uniform(20, 30)
            std_dev = self.rng.uniform(3, 8)
        else:
            # Normal traffic: consistent speeds
            mean_speed = self.rng.uniform(23, 27)
            std_dev = self.rng.uniform(1, 3)
        
        speeds = self.rng.normal(mean_speed, std_dev, num_vehicles)
        speeds = np.clip(speeds, 15, 35)  # Highway speed limits
        
        return speeds.tolist()
    
    def _generate_vehicle_positions(
        self, num_vehicles: int, traffic_density: str
    ) -> List[Tuple[float, int]]:
        """
        Generate initial positions for vehicles.
        
        Args:
            num_vehicles: Number of vehicles
            traffic_density: Traffic density level
        
        Returns:
            List of (longitudinal_position, lane_id) tuples
        """
        # Highway has typically 3-4 lanes
        num_lanes = 4
        
        # Set spacing based on density
        if traffic_density == "low":
            min_spacing = 40
            max_spacing = 80
        elif traffic_density == "medium":
            min_spacing = 25
            max_spacing = 50
        else:  # high
            min_spacing = 15
            max_spacing = 35
        
        positions = []
        
        # Ego vehicle always starts at (0, lane 1 or 2)
        ego_lane = self.rng.integers(1, 3)
        positions.append((0.0, ego_lane))
        
        # Generate other vehicles
        for i in range(1, num_vehicles):
            # Random lane
            lane = self.rng.integers(0, num_lanes)
            
            # Random longitudinal position (both ahead and behind ego)
            if self.rng.random() < 0.6:  # 60% ahead
                long_pos = self.rng.uniform(min_spacing, min_spacing * 3)
            else:  # 40% behind
                long_pos = -self.rng.uniform(min_spacing, min_spacing * 2)
            
            positions.append((long_pos, lane))
        
        return positions
    
    def _generate_adversarial_events(
        self, num_vehicles: int
    ) -> List[AdversarialEvent]:
        """
        Generate adversarial events for the scenario.
        
        Args:
            num_vehicles: Number of vehicles in the scenario
        
        Returns:
            List of adversarial events
        """
        events = []
        
        # Calculate number of events based on difficulty
        event_frequency = self.params["event_frequency"]
        num_events = int(self.rng.poisson(event_frequency * self.episode_length / 100))
        num_events = min(num_events, 5)  # Cap at 5 events per episode
        
        if num_events == 0:
            return events
        
        # Sample event types
        event_types = self.rng.choice(
            list(AdversarialEventType),
            size=num_events,
            replace=True,
        )
        
        # Generate each event
        for event_type in event_types:
            event = self._create_adversarial_event(event_type, num_vehicles)
            if event:
                events.append(event)
        
        # Sort by trigger time
        events.sort(key=lambda e: e.trigger_time)
        
        return events
    
    def _create_adversarial_event(
        self, event_type: AdversarialEventType, num_vehicles: int
    ) -> Optional[AdversarialEvent]:
        """
        Create a specific adversarial event.
        
        Args:
            event_type: Type of adversarial event
            num_vehicles: Number of vehicles in scenario
        
        Returns:
            AdversarialEvent configuration
        """
        # Sample severity
        severity_min, severity_max = self.params["event_severity_range"]
        severity = self.rng.uniform(severity_min, severity_max)
        
        # Sample trigger time (not too early, not too late)
        trigger_time = self.rng.uniform(
            self.episode_length * 0.2,
            self.episode_length * 0.8
        )
        
        # Select target vehicle (not ego, which is always index 0)
        if num_vehicles < 2:
            return None
        target_vehicle = self.rng.integers(1, num_vehicles)
        
        # Create event based on type
        if event_type == AdversarialEventType.AGGRESSIVE_CUT_IN:
            gap_min, gap_max = self.params["cut_in_gap_range"]
            gap_distance = gap_min + (1 - severity) * (gap_max - gap_min)
            
            return AdversarialEvent(
                event_type=event_type,
                trigger_time=trigger_time,
                duration=self.rng.uniform(2, 5),
                severity=severity,
                target_vehicle_idx=target_vehicle,
                parameters={
                    "gap_distance": gap_distance,
                    "relative_speed": self.rng.uniform(-5, 5),
                    "target_lane_offset": self.rng.choice([-1, 1]),  # Left or right
                }
            )
        
        elif event_type == AdversarialEventType.BRAKE_CHECK:
            decel_min, decel_max = self.params["brake_decel_range"]
            deceleration = decel_min + severity * (decel_min - decel_max)
            
            return AdversarialEvent(
                event_type=event_type,
                trigger_time=trigger_time,
                duration=self.rng.uniform(1, 3),
                severity=severity,
                target_vehicle_idx=target_vehicle,
                parameters={
                    "deceleration": deceleration,
                    "recovery_time": self.rng.uniform(2, 4),
                }
            )
        
        elif event_type == AdversarialEventType.ERRATIC_LANE_CHANGE:
            return AdversarialEvent(
                event_type=event_type,
                trigger_time=trigger_time,
                duration=self.rng.uniform(5, 10),
                severity=severity,
                target_vehicle_idx=target_vehicle,
                parameters={
                    "num_changes": int(2 + severity * 3),
                    "warning_time": (1 - severity) * 2,  # Less warning = more erratic
                    "completion_time": self.rng.uniform(1, 3),
                }
            )
        
        elif event_type == AdversarialEventType.SPEED_VARIANCE:
            return AdversarialEvent(
                event_type=event_type,
                trigger_time=trigger_time,
                duration=self.rng.uniform(10, 20),
                severity=severity,
                target_vehicle_idx=target_vehicle,
                parameters={
                    "speed_oscillation_amplitude": 5 + severity * 10,  # m/s
                    "oscillation_period": 3 + (1 - severity) * 3,  # seconds
                }
            )
        
        elif event_type == AdversarialEventType.SUDDEN_SLOWDOWN:
            return AdversarialEvent(
                event_type=event_type,
                trigger_time=trigger_time,
                duration=self.rng.uniform(3, 8),
                severity=severity,
                target_vehicle_idx=target_vehicle,
                parameters={
                    "speed_reduction": 10 + severity * 15,  # m/s reduction
                    "deceleration_rate": -3 - severity * 4,  # m/s²
                }
            )
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        adversarial_percentage = 0.0
        if self.scenarios_generated > 0:
            adversarial_percentage = (
                100 * self.adversarial_scenarios_generated / self.scenarios_generated
            )
        
        return {
            "total_scenarios": self.scenarios_generated,
            "adversarial_scenarios": self.adversarial_scenarios_generated,
            "adversarial_percentage": adversarial_percentage,
            "mode": self.mode,
            "difficulty": self.difficulty.value,
            "adversarial_ratio": self.adversarial_ratio,
        }
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.scenarios_generated = 0
        self.adversarial_scenarios_generated = 0


class ScenarioMetrics:
    """Metrics tracker for scenario evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.collisions = 0
        self.ttc_violations = 0  # TTC < threshold
        self.lane_changes_successful = 0
        self.lane_changes_attempted = 0
        self.overtakes_successful = 0
        self.overtakes_attempted = 0
        self.min_ttc = float('inf')
        self.total_jerk = 0.0
        self.timesteps = 0
        self.episode_complete = False
    
    def update(
        self,
        collision: bool = False,
        ttc: float = float('inf'),
        ttc_threshold: float = 2.0,
        lane_change_attempted: bool = False,
        lane_change_successful: bool = False,
        overtake_attempted: bool = False,
        overtake_successful: bool = False,
        jerk: float = 0.0,
    ):
        """Update metrics for current timestep."""
        self.timesteps += 1
        
        if collision:
            self.collisions += 1
        
        if ttc < ttc_threshold:
            self.ttc_violations += 1
        
        self.min_ttc = min(self.min_ttc, ttc)
        
        if lane_change_attempted:
            self.lane_changes_attempted += 1
            if lane_change_successful:
                self.lane_changes_successful += 1
        
        if overtake_attempted:
            self.overtakes_attempted += 1
            if overtake_successful:
                self.overtakes_successful += 1
        
        self.total_jerk += abs(jerk)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        lane_change_rate = 0.0
        if self.lane_changes_attempted > 0:
            lane_change_rate = self.lane_changes_successful / self.lane_changes_attempted
        
        overtake_rate = 0.0
        if self.overtakes_attempted > 0:
            overtake_rate = self.overtakes_successful / self.overtakes_attempted
        
        avg_jerk = 0.0
        if self.timesteps > 0:
            avg_jerk = self.total_jerk / self.timesteps
        
        return {
            "collision_rate": self.collisions,
            "ttc_violation_rate": self.ttc_violations / max(1, self.timesteps),
            "min_ttc": self.min_ttc if self.min_ttc != float('inf') else 0.0,
            "lane_change_success_rate": lane_change_rate,
            "overtake_success_rate": overtake_rate,
            "avg_jerk": avg_jerk,
            "timesteps": self.timesteps,
        }
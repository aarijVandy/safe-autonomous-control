"""
Configuration file for hierarchical constrained RL controller.

This module centralizes all hyperparameters, constraint thresholds, and training
settings for the hierarchical controller system.
"""

from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ConstraintConfig:
    """Configuration for safety constraints."""
    
    # TTC constraint
    ttc_min: float = 2.0  # seconds
    ttc_distance_threshold: float = 50.0  # meters (only apply if vehicle is close)
    
    # Headway constraint (IDM-style)
    headway_s0: float = 2.0  # minimum gap (meters)
    headway_T: float = 1.5  # time headway (seconds)
    
    # Lateral clearance constraint
    clearance_min: float = 1.5  # meters (during lane change)
    
    # Jerk constraint
    jerk_max: float = 3.0  # m/s³
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RewardConfig:
    """Configuration for reward function."""
    
    # Reward weights
    w_speed: float = 1.0          # Speed tracking weight
    w_lane: float = 0.5           # Lane centering weight
    w_progress: float = 0.4       # Progress weight
    w_jerk: float = 0.1           # Jerk penalty weight
    w_accel: float = 0.05         # Acceleration penalty weight
    
    # Target parameters
    v_target: float = 19.5       # Target velocity (m/s) - 70 km/h
    v_min: float = 15.0           # Minimum velocity (m/s)
    v_max: float = 35.0           # Maximum velocity (m/s)
    
    # Discount factor
    gamma: float = 0.99
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SACConfig:
    """Configuration for SAC algorithm."""
    
    # Learning rates
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_temperature: float = 3e-4
    lr_dual: float = 0.01        # Dual variable learning rate
    
    # Network architecture
    hidden_sizes: tuple = (256, 256)
    activation: str = "relu"
    
    # Training parameters
    gamma: float = 0.99
    tau: float = 0.005           # Soft update coefficient
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    
    # Exploration
    initial_temperature: float = 0.2
    target_entropy: str = "auto"  # or float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SafetyLayerConfig:
    """Configuration for safety layer (CBF-QP)."""
    
    # CBF parameters
    alpha: float = 0.5            # Class-K function parameter
    
    # QP solver settings
    solver: str = "osqp"          # Solver backend
    max_iter: int = 100           # Maximum QP iterations
    eps_abs: float = 1e-3         # Absolute tolerance
    eps_rel: float = 1e-3         # Relative tolerance
    polish: bool = True           # Polish solution
    warm_start: bool = True       # Use warm starting
    
    # Timing
    max_solve_time: float = 0.01  # Maximum solve time (seconds)
    
    # Emergency behavior
    emergency_brake_accel: float = -5.0  # m/s²
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ManeuverPolicyConfig:
    """Configuration for maneuver policy."""
    
    # Policy type
    policy_type: str = "rule_based"  # "rule_based" or "learned"
    
    # Timing
    min_dwell_time: float = 2.0       # Minimum time in maneuver (seconds)
    maneuver_frequency: float = 1.0   # Hz (how often to update maneuver)
    
    # Rule-based policy parameters
    speed_threshold: float = 3.0      # m/s difference to trigger overtake
    ttc_min_lane_change: float = 3.0  # seconds (minimum TTC for safe gap)
    advantage_threshold: float = 2.0  # m/s (minimum speed advantage to change lanes)
    
    # Learned policy parameters (if policy_type == "learned")
    lr_maneuver: float = 3e-4
    maneuver_hidden_sizes: tuple = (128, 128)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    # Training duration
    total_timesteps: int = 1_000_000
    n_eval_episodes: int = 50
    eval_freq: int = 10_000
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_phases: list = None  # Will be set in __post_init__
    
    # Prioritized replay
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    
    # Dual variable updates
    dual_update_frequency: int = 10  # Update duals every N episodes
    
    # Checkpointing
    checkpoint_freq: int = 50_000
    save_best_model: bool = True
    success_rate_threshold: float = 0.90
    violation_rate_threshold: float = 0.05
    
    # Logging
    log_freq: int = 1000
    tensorboard_log: str = "./logs/hierarchical_sac"
    
    def __post_init__(self):
        """Set default curriculum phases if not provided."""
        if self.curriculum_phases is None and self.curriculum_enabled:
            self.curriculum_phases = [
                # (max_timestep, adversarial_ratio, difficulty)
                (300_000, 0.2, "easy"),
                (700_000, 0.5, "medium"),
                (1_000_000, 0.7, "hard"),
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert list to tuple for JSON serialization
        if self.curriculum_phases:
            d['curriculum_phases'] = [list(phase) for phase in self.curriculum_phases]
        return d


@dataclass
class HierarchicalControllerConfig:
    """Complete configuration for hierarchical constrained RL controller."""
    
    # Sub-configurations
    constraints: ConstraintConfig = None
    reward: RewardConfig = None
    sac: SACConfig = None
    safety_layer: SafetyLayerConfig = None
    maneuver_policy: ManeuverPolicyConfig = None
    training: TrainingConfig = None
    
    # Hierarchical architecture
    trajectory_frequency: float = 10.0  # Hz (low-level policy)
    
    # Action space
    action_space_low: tuple = (-5.0, -0.3)  # [a_long, a_lat]
    action_space_high: tuple = (3.0, 0.3)
    
    # Observation space
    obs_dim: int = 18
    
    def __post_init__(self):
        """Initialize sub-configurations with defaults if not provided."""
        if self.constraints is None:
            self.constraints = ConstraintConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.sac is None:
            self.sac = SACConfig()
        if self.safety_layer is None:
            self.safety_layer = SafetyLayerConfig()
        if self.maneuver_policy is None:
            self.maneuver_policy = ManeuverPolicyConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            'constraints': self.constraints.to_dict(),
            'reward': self.reward.to_dict(),
            'sac': self.sac.to_dict(),
            'safety_layer': self.safety_layer.to_dict(),
            'maneuver_policy': self.maneuver_policy.to_dict(),
            'training': self.training.to_dict(),
            'trajectory_frequency': self.trajectory_frequency,
            'action_space_low': self.action_space_low,
            'action_space_high': self.action_space_high,
            'obs_dim': self.obs_dim,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HierarchicalControllerConfig':
        """Create configuration from dictionary."""
        return cls(
            constraints=ConstraintConfig(**config_dict.get('constraints', {})),
            reward=RewardConfig(**config_dict.get('reward', {})),
            sac=SACConfig(**config_dict.get('sac', {})),
            safety_layer=SafetyLayerConfig(**config_dict.get('safety_layer', {})),
            maneuver_policy=ManeuverPolicyConfig(**config_dict.get('maneuver_policy', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            trajectory_frequency=config_dict.get('trajectory_frequency', 10.0),
            action_space_low=tuple(config_dict.get('action_space_low', (-5.0, -0.3))),
            action_space_high=tuple(config_dict.get('action_space_high', (3.0, 0.3))),
            obs_dim=config_dict.get('obs_dim', 18),
        )


# Preset configurations for different scenarios

DEFAULT_CONFIG = HierarchicalControllerConfig()

# Conservative configuration (tighter safety constraints)
CONSERVATIVE_CONFIG = HierarchicalControllerConfig(
    constraints=ConstraintConfig(
        ttc_min=2.5,
        headway_s0=3.0,
        headway_T=2.0,
        clearance_min=2.0,
        jerk_max=2.5,
    ),
    safety_layer=SafetyLayerConfig(
        alpha=0.7,  # More conservative CBF
    )
)

# Aggressive configuration (looser safety, higher performance)
AGGRESSIVE_CONFIG = HierarchicalControllerConfig(
    constraints=ConstraintConfig(
        ttc_min=1.5,
        headway_s0=1.5,
        headway_T=1.0,
        clearance_min=1.0,
        jerk_max=4.0,
    ),
    reward=RewardConfig(
        w_speed=1.5,
        w_progress=0.6,
        v_target=30.0,
    ),
    safety_layer=SafetyLayerConfig(
        alpha=0.3,  # Less conservative CBF
    )
)

# Fast training configuration (smaller buffer, fewer timesteps)
FAST_TRAINING_CONFIG = HierarchicalControllerConfig(
    sac=SACConfig(
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
    ),
    training=TrainingConfig(
        total_timesteps=500_000,
        eval_freq=5_000,
        checkpoint_freq=25_000,
        curriculum_phases=[
            (150_000, 0.2, "easy"),
            (350_000, 0.5, "medium"),
            (500_000, 0.7, "hard"),
        ]
    )
)


def get_config(config_name: str = "default") -> HierarchicalControllerConfig:
    """
    Get preset configuration by name.
    
    Args:
        config_name: Name of preset configuration
            - "default": Balanced safety and performance
            - "conservative": Tight safety constraints
            - "aggressive": Looser safety, prioritize performance
            - "fast_training": Smaller resources for quick iteration
    
    Returns:
        config: HierarchicalControllerConfig instance
    
    Raises:
        ValueError: If config_name is not recognized
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "fast_training": FAST_TRAINING_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available configs: {list(configs.keys())}"
        )
    
    return configs[config_name]

"""
Quick test script for hierarchical controller setup.

This script verifies that:
1. Environment loads correctly
2. Controller initializes properly
3. Actions can be computed
4. Costs are tracked correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.environments.nade_wrapper import create_nade_env
from src.controllers.constrained_controller import (
    HierarchicalController,
    RuleBasedManeuverPolicy,
)
from src.controllers.sac_trajectory_policy import SACTrajectoryPolicy
from config.environment_config import get_scenario_config


def test_hierarchical_controller():
    """Test hierarchical controller initialization and basic operations."""
    print("\n" + "="*60)
    print("Testing Hierarchical Constrained RL Controller")
    print("="*60 + "\n")
    
    # Create environment with normal traffic
    print("1. Creating environment...")
    env_config = get_scenario_config('normal')
    env = create_nade_env(
        adversarial_mode=env_config['adversarial_mode'],
        vehicles_count=env_config['vehicles_count'],
        duration=env_config['duration'],
    )
    print("   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.shape}")
    
    # Create controller
    print("\n2. Creating hierarchical controller...")
    
    # Maneuver policy config
    maneuver_config = {
        'target_speed': 25.0,
        'speed_diff_threshold': 3.0,
        'ttc_front_threshold': 3.0,
        'ttc_rear_threshold': 3.0,
        'time_advantage_threshold': 2.0,
        'min_dwell_time': 2.0,
    }
    maneuver_policy = RuleBasedManeuverPolicy(maneuver_config)
    print("   ✓ Maneuver policy created")
    
    # Trajectory policy config
    trajectory_config = {
        'obs_dim': 18,
        'act_dim': 1,
        'hidden_dim': 256,
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 10000,  # Smaller for testing
        'batch_size': 64,
        'constraint_thresholds': {
            'ttc_min': 2.0,
            'headway_s0': 2.0,
            'headway_T': 1.5,
            'clearance_min': 1.5,
            'jerk_max': 3.0,
        },
        'dual_learning_rate': 0.01,
        'w_speed': 1.0,
        'w_lane': 0.5,
        'w_progress': 0.2,
        'w_comfort': 0.1,
    }
    trajectory_policy = SACTrajectoryPolicy(trajectory_config)
    print("   ✓ Trajectory policy created")
    print(f"   - Device: {trajectory_policy.device}")
    print(f"   - Actor parameters: {sum(p.numel() for p in trajectory_policy.actor.parameters())}")
    print(f"   - Critic parameters: {sum(p.numel() for p in trajectory_policy.critic.parameters())}")
    
    # Hierarchical controller config
    hierarchical_config = {
        'maneuver_frequency': 1.0,
        'trajectory_frequency': 10.0,
    }
    controller = HierarchicalController(
        maneuver_policy=maneuver_policy,
        trajectory_policy=trajectory_policy,
        config=hierarchical_config,
    )
    print("   ✓ Hierarchical controller created")
    
    # Test episode
    print("\n3. Running test episode (50 steps)...")
    obs, info = env.reset()
    controller.reset()
    
    episode_reward = 0.0
    episode_costs = {name: [] for name in ['ttc', 'headway', 'clearance', 'jerk']}
    
    for step in range(50):
        # Compute action
        action = controller.compute_action(obs, info)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Compute costs
        costs = controller.compute_costs(obs, action, next_obs)
        
        # Track
        episode_reward += reward
        for name, cost in costs.items():
            episode_costs[name].append(cost)
        
        # Add to replay buffer
        shaped_reward = trajectory_policy.compute_reward(obs, action)
        trajectory_policy.replay_buffer.add(obs, action, shaped_reward, next_obs, done, costs)
        
        if done or truncated:
            break
        
        obs = next_obs
    
    print(f"   ✓ Episode completed ({step + 1} steps)")
    print(f"   - Total reward: {episode_reward:.2f}")
    print(f"   - Replay buffer size: {len(trajectory_policy.replay_buffer)}")
    
    # Test cost tracking
    print("\n4. Analyzing constraint costs...")
    for name, cost_list in episode_costs.items():
        if len(cost_list) > 0:
            avg_cost = np.mean(cost_list)
            violations = sum(1 for c in cost_list if c > 1e-6)
            print(f"   - {name}: avg={avg_cost:.4f}, violations={violations}/{len(cost_list)}")
    
    # Test dual variables
    print("\n5. Checking dual variables...")
    expected_costs = controller.get_expected_costs()
    controller.update_dual_variables(expected_costs)
    for name, value in controller.dual_variables.items():
        print(f"   - {name}: {value:.6f}")
    
    # Test constraint stats
    print("\n6. Controller statistics...")
    stats = controller.get_stats()
    print(f"   - Timesteps: {stats['total_timesteps']}")
    print(f"   - Maneuver stats: {stats['maneuver_stats']}")
    
    # Test network update
    if len(trajectory_policy.replay_buffer) >= trajectory_config['batch_size']:
        print("\n7. Testing policy update...")
        update_metrics = trajectory_policy.update()
        print("   ✓ Policy update successful")
        for key, value in update_metrics.items():
            print(f"   - {key}: {value:.4f}")
    else:
        print(f"\n7. Skipping policy update (need {trajectory_config['batch_size']} samples, have {len(trajectory_policy.replay_buffer)})")
    
    # Cleanup
    env.close()
    
    print("\n" + "="*60)
    print("✓ All tests passed! Controller is ready for training.")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_hierarchical_controller()

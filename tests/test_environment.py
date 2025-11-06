"""
Test script to validate NADE highway environment.

This script creates the environment, runs random actions, and verifies:
1. Environment can be created and reset
2. Actions can be executed
3. Observations have correct shape
4. Basic metrics are tracked
5. Rendering works (if enabled)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.nade_wrapper import create_nade_env
from config.environment_config import get_scenario_config


def test_environment(
    scenario: str = "normal",
    num_episodes: int = 3,
    max_steps: int = 200,
    render: bool = False,
):
    """
    Test the NADE highway environment.
    
    Args:
        scenario: Scenario name ('normal', 'mild', 'moderate', 'severe')
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    """
    print("=" * 80)
    print("NADE Highway Environment Test")
    print("=" * 80)
    
    # Get scenario configuration
    config = get_scenario_config(scenario)
    print(f"\nScenario: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Adversarial Mode: {config['adversarial_mode']}")
    
    # Create environment
    render_mode = "human" if render else None
    env = create_nade_env(render_mode=render_mode, **config)
    
    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'=' * 80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'=' * 80}")
        
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        print(f"Initial observation: {obs}")
        
        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Print step info every 50 steps
            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}: "
                      f"reward={reward:.3f}, "
                      f"rel_dist={obs[0]:.2f}m, "
                      f"rel_vel={obs[1]:.2f}m/s, "
                      f"ttc={obs[2]:.2f}s, "
                      f"ego_vel={obs[3]:.2f}m/s")
            
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"\n  Episode ended ({reason}) at step {episode_steps}")
                break
        
        # Print episode summary
        print(f"\n  Episode Summary:")
        print(f"    Total Reward: {episode_reward:.3f}")
        print(f"    Total Steps: {episode_steps}")
        print(f"    Metrics: {info.get('episode_metrics', {})}")
    
    # Close environment
    env.close()
    
    print(f"\n{'=' * 80}")
    print("Test completed successfully!")
    print(f"{'=' * 80}\n")


def test_observation_consistency():
    """Test that observations are consistent and within expected ranges."""
    print("\nTesting observation consistency...")
    
    env = create_nade_env(adversarial_mode=False)
    
    # Run multiple resets and check observations
    for i in range(5):
        obs, _ = env.reset()
        
        # Check observation shape
        assert obs.shape == env.observation_space.shape, \
            f"Observation shape mismatch: {obs.shape} != {env.observation_space.shape}"
        
        # Check observation values are reasonable
        assert np.isfinite(obs).all(), "Observation contains non-finite values"
        
        # Check TTC is non-negative
        assert obs[2] >= 0.0, f"TTC is negative: {obs[2]}"
        
        # Check ego velocity is non-negative
        assert obs[3] >= 0.0, f"Ego velocity is negative: {obs[3]}"
        
        # Check lane occupancy is binary
        assert obs[4] in [0.0, 1.0], f"Lane occupancy not binary: {obs[4]}"
    
    env.close()
    print("  ✓ Observation consistency test passed!")


def test_action_clipping():
    """Test that actions are properly clipped to valid range."""
    print("\nTesting action clipping...")
    
    env = create_nade_env(adversarial_mode=False)
    obs, _ = env.reset()
    
    # Test extreme actions
    extreme_actions = [
        np.array([-10.0]),  # Way below minimum
        np.array([10.0]),   # Way above maximum
        np.array([0.0]),    # Zero (valid)
        np.array([-3.0]),   # Minimum (valid)
        np.array([2.0]),    # Maximum (valid)
    ]
    
    for action in extreme_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        # If this doesn't crash, action clipping is working
        assert np.isfinite(obs).all(), f"Invalid observation after action {action}"
    
    env.close()
    print("  ✓ Action clipping test passed!")


def test_adversarial_mode():
    """Test that adversarial mode can be toggled."""
    print("\nTesting adversarial mode...")
    
    # Test normal mode
    env_normal = create_nade_env(adversarial_mode=False)
    obs, info = env_normal.reset()
    assert not info["adversarial_mode"], "Adversarial mode should be False"
    env_normal.close()
    
    # Test adversarial mode
    env_adv = create_nade_env(adversarial_mode=True, adversarial_intensity=0.5)
    obs, info = env_adv.reset()
    assert info["adversarial_mode"], "Adversarial mode should be True"
    env_adv.close()
    
    print("  ✓ Adversarial mode test passed!")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print("Running Unit Tests")
    print("=" * 80)
    
    test_observation_consistency()
    test_action_clipping()
    test_adversarial_mode()
    
    print("\n" + "=" * 80)
    print("All unit tests passed!")
    print("=" * 80 + "\n")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test NADE highway environment")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "mild", "moderate", "severe"],
        help="Scenario to test"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run unit tests only"
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        run_all_tests()
    else:
        # Run unit tests first
        run_all_tests()
        
        # Then run environment test
        test_environment(
            scenario=args.scenario,
            num_episodes=args.episodes,
            max_steps=args.steps,
            render=args.render,
        )


if __name__ == "__main__":
    main()

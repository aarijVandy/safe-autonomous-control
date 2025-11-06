"""
Test script for PID controller.

This script tests the PID controller in the NADE environment
to verify it works correctly before optimization.
"""

import sys
import os
import argparse
import numpy as np
from gymnasium.wrappers import RecordVideo

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.nade_wrapper import create_nade_env
from src.controllers.pid_controller import PIDController
from config.environment_config import get_scenario_config


def test_pid_controller(
    scenario: str = "normal",
    num_episodes: int = 3,
    render: bool = True,
    kp: float = 0.5,
    ki: float = 0.1,
    kd: float = 0.2,
    record_video: bool = False,
    video_folder: str = None,
):
    """
    Test PID controller in the environment.
    
    Args:
        scenario: Environment scenario
        num_episodes: Number of test episodes
        render: Whether to render the environment
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        record_video: Whether to record video of episodes
        video_folder: Folder to save videos (default: media/)
    """
    print("\n" + "=" * 70)
    print("PID CONTROLLER TEST")
    print("=" * 70)
    print(f"Scenario: {scenario}")
    print(f"Episodes: {num_episodes}")
    print(f"PID Gains: Kp={kp}, Ki={ki}, Kd={kd}")
    if record_video:
        print(f"Recording: Enabled (saving to {video_folder or 'media/'})")
    print("=" * 70 + "\n")
    
    # Create environment
    scenario_config = get_scenario_config(scenario)
    
    # Determine render mode
    if record_video:
        # For video recording, we need rgb_array mode
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None
    
    env = create_nade_env(
        adversarial_mode=scenario_config.get("adversarial_mode", False),
        render_mode=render_mode,
        vehicles_count=scenario_config.get("vehicles_count", 15),
        duration=scenario_config.get("duration", 40.0),
        adversarial_intensity=scenario_config.get("adversarial_intensity", 0.5),
    )
    
    # Wrap with RecordVideo if requested
    if record_video:
        if video_folder is None:
            video_folder = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "media"
            )
        
        # Create video folder if it doesn't exist
        os.makedirs(video_folder, exist_ok=True)
        
        # Add RecordVideo wrapper
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"pid_{scenario}_kp{kp}_ki{ki}_kd{kd}",
            episode_trigger=lambda x: True,  # Record all episodes
        )
        print(f"Video recording initialized. Videos will be saved to: {video_folder}\n")
    
    # Create controller
    controller = PIDController(kp=kp, ki=ki, kd=kd)
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    collision_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        controller.reset()
        
        done = truncated = False
        episode_reward = 0.0
        episode_length = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 70)
        
        while not (done or truncated):
            # Compute action
            action = controller.compute_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Print state every 20 steps
            if episode_length % 20 == 0:
                print(
                    f"  Step {episode_length}: "
                    f"Speed={obs[3]:.1f} m/s, "
                    f"Distance={obs[0]:.1f} m, "
                    f"TTC={obs[2]:.1f} s, "
                    f"Action={action[0]:.2f}"
                )
            
            if done:
                if info.get("episode_metrics", {}).get("collisions", 0) > 0:
                    collision_count += 1
                    print(f"  ✗ Collision at step {episode_length}!")
                break
        
        # Get controller metrics
        controller_metrics = controller.get_metrics()
        
        # Print episode summary
        print(f"\nEpisode Summary:")
        print(f"  Total Reward:    {episode_reward:.2f}")
        print(f"  Episode Length:  {episode_length} steps")
        print(f"  Avg Jerk:        {controller_metrics['avg_jerk']:.4f}")
        print(f"  Avg Speed Error: {controller_metrics.get('avg_speed_error', 0):.2f} m/s")
        print(f"  TTC Violations:  {controller_metrics['ttc_violations']}")
        print(f"  Collision:       {'Yes' if done and collision_count > 0 else 'No'}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Average Reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length:     {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Success Rate:       {(num_episodes - collision_count) / num_episodes:.1%}")
    print(f"Collision Rate:     {collision_count / num_episodes:.1%}")
    print("=" * 70 + "\n")
    
    env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test PID controller")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "mild", "moderate", "severe"],
        help="Environment scenario",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of test episodes"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of episodes",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default=None,
        help="Folder to save videos (default: media/)",
    )
    parser.add_argument("--kp", type=float, default=0.5, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.1, help="Integral gain")
    parser.add_argument("--kd", type=float, default=0.2, help="Derivative gain")
    
    args = parser.parse_args()
    
    test_pid_controller(
        scenario=args.scenario,
        num_episodes=args.episodes,
        render=args.render,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        record_video=args.record_video,
        video_folder=args.video_folder,
    )


if __name__ == "__main__":
    main()

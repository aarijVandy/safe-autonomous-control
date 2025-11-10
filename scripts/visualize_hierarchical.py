#!/usr/bin/env python3
"""
Hierarchical Controller Visualization Script.

This script provides comprehensive visualization of trained hierarchical controller performance:
1. Real-time environment rendering
2. Time-series plots (speed, distance, TTC, actions)
3. Performance metrics dashboard
4. Constraint violation tracking
5. Maneuver decision visualization
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.nade_wrapper import create_nade_env
from src.controllers.constrained_controller import (
    HierarchicalController,
    RuleBasedManeuverPolicy,
)
from src.controllers.sac_trajectory_policy import SACTrajectoryPolicy
from config.environment_config import get_scenario_config
from config.hierarchical_config import get_config


class HierarchicalVisualizer:
    """Visualizer for hierarchical controller performance."""
    
    def __init__(
        self,
        checkpoint_path: str,
        scenario: str = "normal",
        render_env: bool = True,
        max_steps: int = 1000,
        record_video: bool = True,
        video_dir: str = "media",
    ):
        """
        Initialize visualizer.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            scenario: Environment scenario
            render_env: Whether to render the environment
            max_steps: Maximum simulation steps
            record_video: Whether to record video
            video_dir: Directory to save videos
        """
        self.checkpoint_path = checkpoint_path
        self.scenario = scenario
        self.max_steps = max_steps
        self.record_video = record_video
        self.video_dir = Path(video_dir)
        
        # Create video directory if recording
        if record_video:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        scenario_config = get_scenario_config(scenario)
        
        # Determine render mode
        if record_video:
            render_mode = "rgb_array"
        elif render_env:
            render_mode = "human"
        else:
            render_mode = None
        
        self.env = create_nade_env(
            adversarial_mode=scenario_config.get("adversarial_mode", False),
            render_mode=render_mode,
            vehicles_count=scenario_config.get("vehicles_count", 15),
            duration=scenario_config.get("duration", 40.0),
            adversarial_intensity=scenario_config.get("adversarial_intensity", 0.5),
        )
        
        # Wrap with video recorder if requested
        if record_video:
            checkpoint_name = Path(checkpoint_path).stem
            video_prefix = f"hierarchical_{scenario}_{checkpoint_name}"
            self.env = RecordVideo(
                self.env,
                video_folder=str(self.video_dir),
                name_prefix=video_prefix,
                episode_trigger=lambda x: True,  # Record every episode
            )
            print(f"Video recording enabled. Videos will be saved to: {self.video_dir}")
        
        # Create and load controller
        self.controller = self._create_and_load_controller()
        
        # Data storage
        self.reset_data()
    
    def _create_and_load_controller(self) -> HierarchicalController:
        """Create hierarchical controller and load trained weights."""
        # Get default configuration
        config = get_config("default")
        
        # Create maneuver policy
        maneuver_config = config.maneuver_policy.to_dict()
        maneuver_policy = RuleBasedManeuverPolicy(maneuver_config)
        
        # Create trajectory policy (action_dim=2 for longitudinal + lateral control)
        trajectory_config = {
            'obs_dim': 18,
            'act_dim': 2,  # UPDATED: 2D action space [acceleration, lane_change]
            'hidden_dim': 256,
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 1000000,
            'batch_size': 256,
            # Constraint thresholds
            'constraint_thresholds': {
                'ttc_min': 2.0,
                'headway_s0': 2.0,
                'headway_T': 1.5,
                'clearance_min': 1.5,
                'jerk_max': 3.0,
            },
            # Dual learning
            'dual_learning_rate': 0.005,
            # Reward weights
            'w_speed': 2.0,
            'w_lane': 0.3,
            'w_progress': 1.0,
            'w_comfort': 0.05,
            # Exploration
            'exploration_bonus': True,
            'exploration_decay': 0.9999,
        }
        trajectory_policy = SACTrajectoryPolicy(trajectory_config)
        
        # Load trained weights
        print(f"Loading model from: {self.checkpoint_path}")
        trajectory_policy.load(self.checkpoint_path)
        trajectory_policy.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        
        # Create hierarchical controller
        hierarchical_config = {
            'maneuver_frequency': config.maneuver_policy.maneuver_frequency,
            'trajectory_frequency': config.trajectory_frequency,
        }
        controller = HierarchicalController(
            maneuver_policy=maneuver_policy,
            trajectory_policy=trajectory_policy,
            config=hierarchical_config,
        )
        
        return controller
    
    def reset_data(self):
        """Reset data storage."""
        self.data = {
            "time": [],
            "ego_speed": [],
            "relative_distance": [],
            "relative_velocity": [],
            "ttc": [],
            "action": [],  # Longitudinal action only (1D)
            "reward": [],
            "maneuver": [],
            "ttc_violation": [],
            "headway_violation": [],
            "jerk": [],
        }
    
    def run_episode(self, verbose: bool = True) -> Dict:
        """
        Run one episode and collect data.
        
        Args:
            verbose: Print progress
        
        Returns:
            episode_info: Dictionary with episode statistics
        """
        obs, info = self.env.reset()
        self.controller.reset()
        self.reset_data()
        
        done = truncated = False
        episode_reward = 0.0
        episode_length = 0
        
        prev_action = 0.0  # Initialize as scalar for 1D action
        ttc_violations = 0
        headway_violations = 0
        
        if verbose:
            print(f"\nRunning episode with Hierarchical RL Controller")
            print(f"Scenario: {self.scenario}")
            print(f"Max steps: {self.max_steps}")
            print("-" * 80)
        
        while not (done or truncated) and episode_length < self.max_steps:
            # Compute action
            action = self.controller.compute_action(obs, info)
            
            # Handle 2D action [acceleration, lane_change]
            if isinstance(action, np.ndarray):
                action_accel = float(action[0]) if len(action) > 0 else 0.0
                action_lane = float(action[1]) if len(action) > 1 else 0.0
            else:
                action_accel = float(action)
                action_lane = 0.0
            
            # Debug: print details every 50 steps with lane change info
            if verbose and episode_length % 50 == 0:
                ego_speed = obs[3]
                lead_dist = obs[0]
                ttc_val = obs[2]
                lane_state = info.get('lane_change_state', -1)
                current_lane = info.get('current_lane', -1)
                target_lane = info.get('target_lane', None)
                state_names = {0: 'IDLE', 1: 'EXECUTING', 2: 'COOLDOWN'}
                state_name = state_names.get(lane_state, 'UNKNOWN')
                print(f"    DEBUG Step {episode_length}: ego_speed={ego_speed:.1f}, lead_dist={lead_dist:.1f}, ttc={ttc_val:.1f}, accel={action_accel:.3f}, lane_cmd={action_lane:.3f}, current_lane={current_lane}, state={state_name}, target={target_lane}")
            
            # Compute jerk (ensure it's a scalar)
            jerk = float(abs(action_accel - prev_action))
            
            # Check violations
            ttc_violated = obs[2] < 2.0 and obs[0] < 50.0  # TTC < 2s and close
            if ttc_violated:
                ttc_violations += 1
            
            # IDM headway: s0 + v*T
            required_headway = 2.0 + obs[3] * 1.5
            headway_violated = obs[0] < required_headway
            if headway_violated:
                headway_violations += 1
            
            # Store data
            self.data["time"].append(episode_length)
            self.data["ego_speed"].append(obs[3])
            self.data["relative_distance"].append(obs[0])
            self.data["relative_velocity"].append(obs[1])
            self.data["ttc"].append(obs[2])
            self.data["action"].append(action_accel)  # Store longitudinal action for plotting
            self.data["maneuver"].append(self.controller.maneuver_policy.current_maneuver)
            self.data["ttc_violation"].append(1 if ttc_violated else 0)
            self.data["headway_violation"].append(1 if headway_violated else 0)
            self.data["jerk"].append(jerk)
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            self.data["reward"].append(reward)
            prev_action = action_accel
            
            # Print progress
            if verbose and episode_length % 100 == 0:
                print(
                    f"  Step {episode_length:4d}: "
                    f"Speed={obs[3]:5.1f} m/s, "
                    f"Dist={obs[0]:5.1f} m, "
                    f"TTC={obs[2]:5.1f} s, "
                    f"Maneuver={self.controller.maneuver_policy.current_maneuver}, "
                    f"Accel={action_accel:5.2f}, Lane={action_lane:5.2f}"
                )
        
        # Get controller statistics
        controller_stats = self.controller.get_stats()
        
        episode_info = {
            "reward": episode_reward,
            "length": episode_length,
            "collision": done and info.get("episode_metrics", {}).get("collisions", 0) > 0,
            "ttc_violations": ttc_violations,
            "headway_violations": headway_violations,
            "avg_jerk": np.mean(self.data["jerk"]) if self.data["jerk"] else 0.0,
            "max_jerk": np.max(self.data["jerk"]) if self.data["jerk"] else 0.0,
            "avg_speed": np.mean(self.data["ego_speed"]) if self.data["ego_speed"] else 0.0,
            "controller_stats": controller_stats,
        }
        
        if verbose:
            print("\n" + "-" * 80)
            print("Episode complete!")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length} steps")
            print(f"  Avg speed: {episode_info['avg_speed']:.2f} m/s")
            print(f"  Avg jerk: {episode_info['avg_jerk']:.4f}")
            print(f"  Max jerk: {episode_info['max_jerk']:.4f}")
            print(f"  TTC violations: {ttc_violations}")
            print(f"  Headway violations: {headway_violations}")
            print(f"  Collision: {'Yes' if episode_info['collision'] else 'No'}")
            
            if self.record_video:
                print(f"\n  Video saved to: {self.video_dir}/")
        
        return episode_info
    
    def plot_results(self, save_path: str = None):
        """
        Create comprehensive plots of the episode.
        
        Args:
            save_path: Optional path to save the figure
        """
        # Convert lists to numpy arrays
        time = np.array(self.data["time"])
        ego_speed = np.array(self.data["ego_speed"])
        relative_distance = np.array(self.data["relative_distance"])
        relative_velocity = np.array(self.data["relative_velocity"])
        ttc = np.array(self.data["ttc"])
        action = np.array(self.data["action"])
        reward = np.array(self.data["reward"])
        ttc_violation = np.array(self.data["ttc_violation"])
        headway_violation = np.array(self.data["headway_violation"])
        jerk = np.array(self.data["jerk"])
        
        # Create figure with subplots (3x2 layout for 1D action)
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"Hierarchical RL Controller Performance ({self.scenario} scenario)",
            fontsize=16,
            fontweight="bold",
        )
        
        # 1. Speed profile
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(time, ego_speed, "b-", linewidth=2, label="Ego Speed")
        target_speed = 16.67  # 60 km/h to match traffic speed
        ax1.axhline(
            y=target_speed,
            color="r",
            linestyle="--",
            label=f"Target ({target_speed} m/s)",
        )
        ax1.set_xlabel("Time (steps)")
        ax1.set_ylabel("Speed (m/s)")
        ax1.set_title("Vehicle Speed Profile")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Following distance with constraints
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(time, relative_distance, "g-", linewidth=2, label="Relative Distance")
        # Plot required headway
        required_headway = 2.0 + ego_speed * 1.5
        ax2.plot(time, required_headway, "r--", linewidth=1.5, label="Required Headway (IDM)")
        ax2.fill_between(time, 0, required_headway, alpha=0.1, color="red")
        ax2.set_xlabel("Time (steps)")
        ax2.set_ylabel("Distance (m)")
        ax2.set_title("Relative Distance & Headway Constraint")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Time-to-collision with violations
        ax3 = plt.subplot(3, 2, 3)
        ttc_clamped = np.clip(ttc, 0, 20)  # Clamp for better visualization
        ax3.plot(time, ttc_clamped, "orange", linewidth=2, label="TTC")
        ax3.axhline(y=2.0, color="r", linestyle="--", linewidth=1.5, label="Min TTC (2.0s)")
        ax3.fill_between(time, 0, 2.0, alpha=0.2, color="red")
        # Mark violations
        violation_times = time[ttc_violation.astype(bool)]
        violation_ttc = ttc_clamped[ttc_violation.astype(bool)]
        ax3.scatter(violation_times, violation_ttc, color="red", s=20, alpha=0.5, label="Violations")
        ax3.set_xlabel("Time (steps)")
        ax3.set_ylabel("TTC (s)")
        ax3.set_title(f"Time-to-Collision (Violations: {np.sum(ttc_violation)})")
        ax3.set_ylim([0, 20])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Control action (longitudinal acceleration only)
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(time, action, "purple", linewidth=2)
        ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax4.fill_between(
            time, action, 0, 
            where=(action >= 0), 
            alpha=0.3, color="green", label="Acceleration"
        )
        ax4.fill_between(
            time, action, 0, 
            where=(action < 0), 
            alpha=0.3, color="red", label="Braking"
        )
        ax4.set_xlabel("Time (steps)")
        ax4.set_ylabel("Acceleration (m/s²)")
        ax4.set_title("Control Action (Longitudinal)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Jerk (acceleration smoothness)
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(time, jerk, "brown", linewidth=2)
        ax5.axhline(
            y=3.0,
            color="r",
            linestyle="--",
            label="Max Jerk Constraint (3.0)",
        )
        ax5.axhline(
            y=np.mean(jerk),
            color="b",
            linestyle="--",
            label=f"Avg: {np.mean(jerk):.3f}",
        )
        ax5.set_xlabel("Time (steps)")
        ax5.set_ylabel("Jerk (|Δa|)")
        ax5.set_title("Jerk (Control Smoothness)")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative reward
        ax6 = plt.subplot(3, 2, 6)
        cumulative_reward = np.cumsum(reward)
        ax6.plot(time, cumulative_reward, "darkblue", linewidth=2)
        ax6.set_xlabel("Time (steps)")
        ax6.set_ylabel("Cumulative Reward")
        ax6.set_title(f"Cumulative Reward (Total: {cumulative_reward[-1]:.1f})")
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_plot(self, multiple_runs: List[Dict]):
        """
        Create comparison plot for multiple scenarios.
        
        Args:
            multiple_runs: List of dictionaries with run data
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Hierarchical Controller: Multi-Scenario Comparison", fontsize=16, fontweight="bold")
        
        colors = ["b", "g", "r", "orange", "purple"]
        
        for idx, run_data in enumerate(multiple_runs):
            color = colors[idx % len(colors)]
            label = run_data.get("label", f"Run {idx+1}")
            
            # Speed
            axes[0, 0].plot(run_data["time"], run_data["ego_speed"], color=color, alpha=0.7, label=label)
            
            # Distance
            axes[0, 1].plot(run_data["time"], run_data["relative_distance"], color=color, alpha=0.7, label=label)
            
            # TTC
            ttc_clamped = np.clip(run_data["ttc"], 0, 20)
            axes[0, 2].plot(run_data["time"], ttc_clamped, color=color, alpha=0.7, label=label)
            
            # Action (longitudinal only)
            axes[1, 0].plot(run_data["time"], run_data["action"], color=color, alpha=0.7, label=label)
            
            # Jerk
            axes[1, 1].plot(run_data["time"], run_data["jerk"], color=color, alpha=0.7, label=label)
            
            # Cumulative reward
            cumulative_reward = np.cumsum(run_data["reward"])
            axes[1, 2].plot(run_data["time"], cumulative_reward, color=color, alpha=0.7, label=label)
        
        # Configure subplots
        axes[0, 0].set_title("Speed Profile")
        axes[0, 0].set_xlabel("Time (steps)")
        axes[0, 0].set_ylabel("Speed (m/s)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title("Following Distance")
        axes[0, 1].set_xlabel("Time (steps)")
        axes[0, 1].set_ylabel("Distance (m)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_title("Time-to-Collision")
        axes[0, 2].set_xlabel("Time (steps)")
        axes[0, 2].set_ylabel("TTC (s)")
        axes[0, 2].set_ylim([0, 20])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_title("Longitudinal Control Action")
        axes[1, 0].set_xlabel("Time (steps)")
        axes[1, 0].set_ylabel("Acceleration (m/s²)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title("Jerk")
        axes[1, 1].set_xlabel("Time (steps)")
        axes[1, 1].set_ylabel("Jerk (||Δa||)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].set_title("Cumulative Reward")
        axes[1, 2].set_xlabel("Time (steps)")
        axes[1, 2].set_ylabel("Cumulative Reward")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize hierarchical controller performance")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/hierarchical_sac_ep299_ts55390.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "mild", "moderate", "severe"],
        help="Environment scenario",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save plot to file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple scenarios",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of the episode",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="media",
        help="Directory to save videos",
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Available checkpoints:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for ckpt in sorted(checkpoints_dir.glob("*.pt")):
                print(f"  {ckpt}")
        return
    
    if args.compare:
        # Compare multiple scenarios
        print("\nComparing multiple scenarios...")
        scenarios = ["normal", "mild", "moderate"]
        
        runs = []
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario}")
            visualizer = HierarchicalVisualizer(
                checkpoint_path=str(checkpoint_path),
                scenario=scenario,
                render_env=False,
                max_steps=args.max_steps,
                record_video=False,  # Don't record videos during comparison
            )
            episode_info = visualizer.run_episode(verbose=False)
            run_data = visualizer.data.copy()
            run_data["label"] = f"{scenario.capitalize()} (R={episode_info['reward']:.1f})"
            runs.append(run_data)
            visualizer.close()
            
            print(f"  Reward: {episode_info['reward']:.2f}")
            print(f"  TTC violations: {episode_info['ttc_violations']}")
            print(f"  Collision: {episode_info['collision']}")
        
        # Create comparison plot
        visualizer = HierarchicalVisualizer(
            checkpoint_path=str(checkpoint_path),
            scenario="normal",
            render_env=False,
            max_steps=args.max_steps,
            record_video=False,
        )
        visualizer.create_comparison_plot(runs)
        visualizer.close()
    else:
        # Single scenario
        visualizer = HierarchicalVisualizer(
            checkpoint_path=str(checkpoint_path),
            scenario=args.scenario,
            render_env=args.render,
            max_steps=args.max_steps,
            record_video=args.record_video,
            video_dir=args.video_dir,
        )
        
        try:
            # Run episode
            episode_info = visualizer.run_episode()
            
            # Plot results
            visualizer.plot_results(save_path=args.save)
        
        finally:
            visualizer.close()


if __name__ == "__main__":
    main()

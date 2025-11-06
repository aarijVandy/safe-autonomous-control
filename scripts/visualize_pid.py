#!/usr/bin/env python3
"""
Advanced PID Controller Visualization Script.

This script provides comprehensive visualization of PID controller performance:
1. Real-time environment rendering
2. Time-series plots (speed, distance, TTC, actions)
3. Performance metrics dashboard
4. Trajectory analysis
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments.nade_wrapper import create_nade_env
from src.controllers.pid_controller import PIDController
from config.environment_config import get_scenario_config


class PIDVisualizer:
    """Visualizer for PID controller performance."""
    
    def __init__(
        self,
        scenario: str = "normal",
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.2,
        render_env: bool = True,
    ):
        """
        Initialize visualizer.
        
        Args:
            scenario: Environment scenario
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            render_env: Whether to render the environment
        """
        self.scenario = scenario
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Create environment
        scenario_config = get_scenario_config(scenario)
        self.env = create_nade_env(
            adversarial_mode=scenario_config.get("adversarial_mode", False),
            render_mode="human" if render_env else None,
            vehicles_count=scenario_config.get("vehicles_count", 15),
            duration=scenario_config.get("duration", 40.0),
            adversarial_intensity=scenario_config.get("adversarial_intensity", 0.5),
        )
        
        # Create controller
        self.controller = PIDController(kp=kp, ki=ki, kd=kd)
        
        # Data storage
        self.reset_data()
    
    def reset_data(self):
        """Reset data storage."""
        self.data = {
            "time": [],
            "ego_speed": [],
            "relative_distance": [],
            "relative_velocity": [],
            "ttc": [],
            "action": [],
            "reward": [],
            "p_term": [],
            "i_term": [],
            "d_term": [],
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
        
        if verbose:
            print(f"\nRunning episode with PID gains: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
            print("-" * 70)
        
        while not (done or truncated):
            # Compute action
            action = self.controller.compute_action(obs)
            
            # Store data
            self.data["time"].append(episode_length)
            self.data["ego_speed"].append(obs[3])
            self.data["relative_distance"].append(obs[0])
            self.data["relative_velocity"].append(obs[1])
            self.data["ttc"].append(obs[2])
            self.data["action"].append(action[0])
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            self.data["reward"].append(reward)
            
            # Print progress
            if verbose and episode_length % 20 == 0:
                print(
                    f"  Step {episode_length}: "
                    f"Speed={obs[3]:.1f} m/s, "
                    f"Dist={obs[0]:.1f} m, "
                    f"TTC={obs[2]:.1f} s, "
                    f"Action={action[0]:.2f}"
                )
        
        # Get controller metrics
        controller_metrics = self.controller.get_metrics()
        
        episode_info = {
            "reward": episode_reward,
            "length": episode_length,
            "collision": done and info.get("episode_metrics", {}).get("collisions", 0) > 0,
            "avg_jerk": controller_metrics.get("avg_jerk", 0.0),
            "avg_speed_error": controller_metrics.get("avg_speed_error", 0.0),
            "ttc_violations": controller_metrics.get("ttc_violations", 0),
        }
        
        if verbose:
            print("\n" + "-" * 70)
            print("Episode complete!")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length} steps")
            print(f"  Avg jerk: {episode_info['avg_jerk']:.4f}")
            print(f"  Collision: {'Yes' if episode_info['collision'] else 'No'}")
        
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
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"PID Controller Performance (Kp={self.kp}, Ki={self.ki}, Kd={self.kd})",
            fontsize=16,
            fontweight="bold",
        )
        
        # 1. Speed profile
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(time, ego_speed, "b-", linewidth=2, label="Ego Speed")
        ax1.axhline(
            y=self.controller.target_speed,
            color="r",
            linestyle="--",
            label=f"Target ({self.controller.target_speed} m/s)",
        )
        ax1.set_xlabel("Time (steps)")
        ax1.set_ylabel("Speed (m/s)")
        ax1.set_title("Vehicle Speed Profile")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Following distance
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(time, relative_distance, "g-", linewidth=2)
        ax2.axhline(
            y=self.controller.safe_distance,
            color="r",
            linestyle="--",
            label=f"Safe Distance ({self.controller.safe_distance} m)",
        )
        ax2.set_xlabel("Time (steps)")
        ax2.set_ylabel("Distance (m)")
        ax2.set_title("Relative Distance to Lead Vehicle")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Time-to-collision
        ax3 = plt.subplot(3, 2, 3)
        ttc_clamped = np.clip(ttc, 0, 20)  # Clamp for better visualization
        ax3.plot(time, ttc_clamped, "orange", linewidth=2)
        ax3.axhline(
            y=self.controller.min_ttc,
            color="r",
            linestyle="--",
            label=f"Min TTC ({self.controller.min_ttc} s)",
        )
        ax3.fill_between(time, 0, self.controller.min_ttc, alpha=0.2, color="red")
        ax3.set_xlabel("Time (steps)")
        ax3.set_ylabel("TTC (s)")
        ax3.set_title("Time-to-Collision")
        ax3.set_ylim([0, 20])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Control action (acceleration)
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(time, action, "purple", linewidth=2)
        ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax4.fill_between(time, action, 0, where=(action >= 0), alpha=0.3, color="green", label="Acceleration")
        ax4.fill_between(time, action, 0, where=(action < 0), alpha=0.3, color="red", label="Braking")
        ax4.set_xlabel("Time (steps)")
        ax4.set_ylabel("Acceleration (m/s²)")
        ax4.set_title("Control Action")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Jerk (rate of change of acceleration)
        ax5 = plt.subplot(3, 2, 5)
        jerk = np.diff(action, prepend=action[0])
        ax5.plot(time, np.abs(jerk), "brown", linewidth=2)
        ax5.set_xlabel("Time (steps)")
        ax5.set_ylabel("Jerk (|Δa|)")
        ax5.set_title("Jerk (Acceleration Smoothness)")
        ax5.axhline(
            y=np.mean(np.abs(jerk)),
            color="b",
            linestyle="--",
            label=f"Avg: {np.mean(np.abs(jerk)):.3f}",
        )
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative reward
        ax6 = plt.subplot(3, 2, 6)
        cumulative_reward = np.cumsum(reward)
        ax6.plot(time, cumulative_reward, "darkblue", linewidth=2)
        ax6.set_xlabel("Time (steps)")
        ax6.set_ylabel("Cumulative Reward")
        ax6.set_title("Cumulative Reward")
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_plot(self, multiple_runs: List[Dict]):
        """
        Create comparison plot for multiple PID configurations.
        
        Args:
            multiple_runs: List of dictionaries with run data
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PID Controller Comparison", fontsize=16, fontweight="bold")
        
        colors = ["b", "g", "r", "orange", "purple"]
        
        for idx, run_data in enumerate(multiple_runs):
            color = colors[idx % len(colors)]
            label = f"Kp={run_data['kp']}, Ki={run_data['ki']}, Kd={run_data['kd']}"
            
            # Speed
            axes[0, 0].plot(run_data["time"], run_data["ego_speed"], color=color, alpha=0.7, label=label)
            
            # Distance
            axes[0, 1].plot(run_data["time"], run_data["relative_distance"], color=color, alpha=0.7, label=label)
            
            # Action
            axes[1, 0].plot(run_data["time"], run_data["action"], color=color, alpha=0.7, label=label)
            
            # Jerk
            jerk = np.diff(run_data["action"], prepend=run_data["action"][0])
            axes[1, 1].plot(run_data["time"], np.abs(jerk), color=color, alpha=0.7, label=label)
        
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
        
        axes[1, 0].set_title("Control Action")
        axes[1, 0].set_xlabel("Time (steps)")
        axes[1, 0].set_ylabel("Acceleration (m/s²)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title("Jerk")
        axes[1, 1].set_xlabel("Time (steps)")
        axes[1, 1].set_ylabel("Jerk (|Δa|)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize PID controller performance")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "mild", "moderate", "severe"],
        help="Environment scenario",
    )
    parser.add_argument("--kp", type=float, default=0.5, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.1, help="Integral gain")
    parser.add_argument("--kd", type=float, default=0.2, help="Derivative gain")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--save", type=str, help="Save plot to file")
    parser.add_argument("--compare", action="store_true", help="Compare multiple PID configurations")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple configurations
        print("\nComparing multiple PID configurations...")
        configs = [
            {"kp": 0.3, "ki": 0.05, "kd": 0.1},
            {"kp": 0.5, "ki": 0.1, "kd": 0.2},
            {"kp": 0.8, "ki": 0.15, "kd": 0.3},
        ]
        
        runs = []
        for config in configs:
            visualizer = PIDVisualizer(
                scenario=args.scenario,
                kp=config["kp"],
                ki=config["ki"],
                kd=config["kd"],
                render_env=False,
            )
            visualizer.run_episode(verbose=False)
            run_data = visualizer.data.copy()
            run_data.update(config)
            runs.append(run_data)
            visualizer.close()
        
        # Create comparison plot
        visualizer = PIDVisualizer(scenario=args.scenario, kp=0.5, ki=0.1, kd=0.2, render_env=False)
        visualizer.create_comparison_plot(runs)
        visualizer.close()
    else:
        # Single configuration
        visualizer = PIDVisualizer(
            scenario=args.scenario,
            kp=args.kp,
            ki=args.ki,
            kd=args.kd,
            render_env=args.render,
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

"""
PID Controller Optimization Script.

This script optimizes PID controller parameters using grid search or
Bayesian optimization to maximize performance on three key metrics:
1. Speed to turn (response time)
2. Stay on road (collision avoidance)
3. Minimize jerks (acceleration smoothness)
"""

import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.environments.nade_wrapper import create_nade_env
from src.controllers.pid_controller import PIDController
from config.environment_config import get_scenario_config


class PIDOptimizer:
    """
    Optimizer for PID controller parameters.
    
    Optimizes Kp, Ki, Kd gains to maximize performance on:
    - Response time (speed to turn)
    - Collision avoidance (stay on road)
    - Jerk minimization (smooth control)
    """
    
    def __init__(
        self,
        scenario: str = "normal",
        num_episodes: int = 5,
        optimization_method: str = "grid_search",
        render: bool = False,
    ):
        """
        Initialize the optimizer.
        
        Args:
            scenario: Environment scenario ('normal', 'mild', 'moderate', 'severe')
            num_episodes: Number of episodes to evaluate each parameter set
            optimization_method: 'grid_search' or 'random_search'
            render: Whether to render the environment
        """
        self.scenario = scenario
        self.num_episodes = num_episodes
        self.optimization_method = optimization_method
        self.render = render
        
        # Create environment
        scenario_config = get_scenario_config(scenario)
        self.env = create_nade_env(
            adversarial_mode=scenario_config.get("adversarial_mode", False),
            render_mode="human" if render else None,
            vehicles_count=scenario_config.get("vehicles_count", 15),
            duration=scenario_config.get("duration", 40.0),
            adversarial_intensity=scenario_config.get("adversarial_intensity", 0.5),
        )
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def evaluate_params(
        self, kp: float, ki: float, kd: float, verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a set of PID parameters.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            verbose: Print episode details
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        # Create controller with these parameters
        controller = PIDController(kp=kp, ki=ki, kd=kd)
        
        # Metrics accumulators
        total_reward = 0.0
        total_collisions = 0
        total_success = 0
        total_jerk = 0.0
        total_speed_error = 0.0
        total_response_time = 0.0
        episode_lengths = []
        
        for episode in range(self.num_episodes):
            obs, info = self.env.reset()
            controller.reset()
            
            done = truncated = False
            episode_reward = 0.0
            episode_length = 0
            response_achieved = False
            response_time = 0
            
            # Track when vehicle reaches target speed (response time)
            target_speed = controller.target_speed
            speed_threshold = 0.9 * target_speed  # 90% of target
            
            while not (done or truncated):
                # Compute action
                action = controller.compute_action(obs)
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track response time (time to reach 90% of target speed)
                if not response_achieved and obs[3] >= speed_threshold:
                    response_achieved = True
                    response_time = episode_length
                
                if done and info.get("episode_metrics", {}).get("collisions", 0) > 0:
                    # Collision occurred
                    break
            
            # Get controller metrics
            controller_metrics = controller.get_metrics()
            
            # Accumulate metrics
            total_reward += episode_reward
            episode_lengths.append(episode_length)
            
            # Check for collision
            if info.get("episode_metrics", {}).get("collisions", 0) > 0:
                total_collisions += 1
            else:
                total_success += 1
            
            # Jerk metric
            total_jerk += controller_metrics.get("avg_jerk", 0.0)
            
            # Speed error
            total_speed_error += controller_metrics.get("avg_speed_error", 0.0)
            
            # Response time
            if response_achieved:
                total_response_time += response_time
            else:
                # Penalize if never reached target speed
                total_response_time += episode_length * 2
            
            if verbose:
                print(
                    f"  Episode {episode + 1}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, "
                    f"Collision={'Yes' if done else 'No'}, "
                    f"Jerk={controller_metrics.get('avg_jerk', 0.0):.3f}"
                )
        
        # Compute average metrics
        metrics = {
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "avg_reward": total_reward / self.num_episodes,
            "success_rate": total_success / self.num_episodes,
            "collision_rate": total_collisions / self.num_episodes,
            "avg_jerk": total_jerk / self.num_episodes,
            "avg_speed_error": total_speed_error / self.num_episodes,
            "avg_response_time": total_response_time / self.num_episodes,
            "avg_episode_length": np.mean(episode_lengths),
        }
        
        # Compute composite score (weighted sum of normalized metrics)
        # Higher is better
        score = self._compute_score(metrics)
        metrics["composite_score"] = score
        
        return metrics
    
    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute composite score from metrics.
        
        Optimization objectives:
        1. Speed to turn (response time): Lower is better
        2. Stay on road (collision avoidance): Higher success rate is better
        3. Minimize jerks: Lower jerk is better
        
        Args:
            metrics: Dictionary of performance metrics
        
        Returns:
            score: Composite score (higher is better)
        """
        # Weights for each objective
        w_response = 0.2      # Response time weight
        w_collision = 0.5     # Collision avoidance weight (most important)
        w_jerk = 0.2          # Jerk minimization weight
        w_speed_error = 0.1   # Speed tracking weight
        
        # Normalize metrics to [0, 1] range
        # Success rate (already in [0, 1])
        collision_score = metrics["success_rate"]
        
        # Response time (lower is better, normalize to [0, 1])
        # Assume good response time is < 50 steps, bad is > 200 steps
        response_time = metrics["avg_response_time"]
        response_score = max(0, 1 - (response_time - 20) / 180)  # Linear interpolation
        
        # Jerk (lower is better, normalize to [0, 1])
        # Assume good jerk is < 0.1, bad is > 1.0
        jerk = metrics["avg_jerk"]
        jerk_score = max(0, 1 - jerk / 1.0)
        
        # Speed error (lower is better)
        speed_error = metrics["avg_speed_error"]
        speed_score = max(0, 1 - speed_error / 10.0)
        
        # Composite score
        score = (
            w_collision * collision_score
            + w_response * response_score
            + w_jerk * jerk_score
            + w_speed_error * speed_score
        )
        
        return score
    
    def grid_search(
        self,
        kp_range: Tuple[float, float, int] = (0.1, 2.0, 5),
        ki_range: Tuple[float, float, int] = (0.0, 0.5, 5),
        kd_range: Tuple[float, float, int] = (0.0, 1.0, 5),
    ):
        """
        Perform grid search over PID parameter space.
        
        Args:
            kp_range: (min, max, num_points) for Kp
            ki_range: (min, max, num_points) for Ki
            kd_range: (min, max, num_points) for Kd
        """
        print("\n" + "=" * 70)
        print("PID CONTROLLER OPTIMIZATION - GRID SEARCH")
        print("=" * 70)
        print(f"Scenario: {self.scenario}")
        print(f"Episodes per configuration: {self.num_episodes}")
        print(f"Kp range: {kp_range}")
        print(f"Ki range: {ki_range}")
        print(f"Kd range: {kd_range}")
        print("=" * 70 + "\n")
        
        # Generate parameter grid
        kp_values = np.linspace(kp_range[0], kp_range[1], kp_range[2])
        ki_values = np.linspace(ki_range[0], ki_range[1], ki_range[2])
        kd_values = np.linspace(kd_range[0], kd_range[1], kd_range[2])
        
        total_configs = len(kp_values) * len(ki_values) * len(kd_values)
        print(f"Testing {total_configs} parameter configurations...\n")
        
        config_num = 0
        for kp in kp_values:
            for ki in ki_values:
                for kd in kd_values:
                    config_num += 1
                    print(f"[{config_num}/{total_configs}] Testing Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")
                    
                    # Evaluate this configuration
                    metrics = self.evaluate_params(kp, ki, kd, verbose=False)
                    self.results.append(metrics)
                    
                    # Print results
                    print(
                        f"  → Score: {metrics['composite_score']:.3f} | "
                        f"Success: {metrics['success_rate']:.1%} | "
                        f"Response: {metrics['avg_response_time']:.1f} steps | "
                        f"Jerk: {metrics['avg_jerk']:.3f}"
                    )
                    
                    # Update best
                    if metrics["composite_score"] > self.best_score:
                        self.best_score = metrics["composite_score"]
                        self.best_params = metrics
                        print(f"  ✓ NEW BEST!")
                    
                    print()
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        self._print_best_params()
    
    def random_search(
        self,
        num_iterations: int = 50,
        kp_range: Tuple[float, float] = (0.1, 2.0),
        ki_range: Tuple[float, float] = (0.0, 0.5),
        kd_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Perform random search over PID parameter space.
        
        Args:
            num_iterations: Number of random configurations to test
            kp_range: (min, max) for Kp
            ki_range: (min, max) for Ki
            kd_range: (min, max) for Kd
        """
        print("\n" + "=" * 70)
        print("PID CONTROLLER OPTIMIZATION - RANDOM SEARCH")
        print("=" * 70)
        print(f"Scenario: {self.scenario}")
        print(f"Episodes per configuration: {self.num_episodes}")
        print(f"Iterations: {num_iterations}")
        print("=" * 70 + "\n")
        
        for i in range(num_iterations):
            # Sample random parameters
            kp = np.random.uniform(kp_range[0], kp_range[1])
            ki = np.random.uniform(ki_range[0], ki_range[1])
            kd = np.random.uniform(kd_range[0], kd_range[1])
            
            print(f"[{i + 1}/{num_iterations}] Testing Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
            
            # Evaluate this configuration
            metrics = self.evaluate_params(kp, ki, kd, verbose=False)
            self.results.append(metrics)
            
            # Print results
            print(
                f"  → Score: {metrics['composite_score']:.3f} | "
                f"Success: {metrics['success_rate']:.1%} | "
                f"Response: {metrics['avg_response_time']:.1f} steps | "
                f"Jerk: {metrics['avg_jerk']:.3f}"
            )
            
            # Update best
            if metrics["composite_score"] > self.best_score:
                self.best_score = metrics["composite_score"]
                self.best_params = metrics
                print(f"  ✓ NEW BEST!")
            
            print()
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        self._print_best_params()
    
    def _print_best_params(self):
        """Print the best parameters found."""
        if self.best_params is None:
            print("No parameters evaluated yet.")
            return
        
        print("\nBEST PID PARAMETERS:")
        print("-" * 70)
        print(f"Kp: {self.best_params['kp']:.4f}")
        print(f"Ki: {self.best_params['ki']:.4f}")
        print(f"Kd: {self.best_params['kd']:.4f}")
        print(f"\nComposite Score: {self.best_params['composite_score']:.4f}")
        print(f"\nPerformance Metrics:")
        print(f"  Success Rate:        {self.best_params['success_rate']:.2%}")
        print(f"  Collision Rate:      {self.best_params['collision_rate']:.2%}")
        print(f"  Avg Response Time:   {self.best_params['avg_response_time']:.1f} steps")
        print(f"  Avg Jerk:            {self.best_params['avg_jerk']:.4f}")
        print(f"  Avg Speed Error:     {self.best_params['avg_speed_error']:.2f} m/s")
        print(f"  Avg Episode Length:  {self.best_params['avg_episode_length']:.1f} steps")
        print(f"  Avg Reward:          {self.best_params['avg_reward']:.2f}")
    
    def save_results(self, filename: str = None):
        """
        Save optimization results to JSON file.
        
        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pid_optimization_{self.scenario}_{timestamp}.json"
        
        # Create output directory
        output_dir = "optimization_results"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data
        data = {
            "scenario": self.scenario,
            "num_episodes": self.num_episodes,
            "optimization_method": self.optimization_method,
            "timestamp": datetime.now().isoformat(),
            "best_params": self.best_params,
            "all_results": self.results,
        }
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        
        # Also save best params to config file
        config_file = os.path.join("config", f"pid_params_{self.scenario}.json")
        with open(config_file, "w") as f:
            json.dump(
                {
                    "kp": self.best_params["kp"],
                    "ki": self.best_params["ki"],
                    "kd": self.best_params["kd"],
                    "performance": {
                        "success_rate": self.best_params["success_rate"],
                        "avg_jerk": self.best_params["avg_jerk"],
                        "avg_response_time": self.best_params["avg_response_time"],
                    },
                },
                f,
                indent=2,
            )
        print(f"Best parameters saved to: {config_file}")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    """Main function to run PID optimization."""
    parser = argparse.ArgumentParser(description="Optimize PID controller parameters")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "mild", "moderate", "severe"],
        help="Environment scenario",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid_search",
        choices=["grid_search", "random_search"],
        help="Optimization method",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per parameter configuration",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for random search",
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = PIDOptimizer(
        scenario=args.scenario,
        num_episodes=args.episodes,
        optimization_method=args.method,
        render=args.render,
    )
    
    try:
        # Run optimization
        if args.method == "grid_search":
            # Coarse grid for faster results, can be refined
            optimizer.grid_search(
                kp_range=(0.2, 1.5, 4),
                ki_range=(0.0, 0.3, 4),
                kd_range=(0.1, 0.8, 4),
            )
        else:
            optimizer.random_search(num_iterations=args.iterations)
        
        # Save results
        optimizer.save_results()
    
    finally:
        optimizer.close()


if __name__ == "__main__":
    main()

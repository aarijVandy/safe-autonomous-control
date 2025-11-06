#!/usr/bin/env python3
"""
Quick demo script for PID controller optimization.

This script runs a quick optimization demo with reduced parameters
for fast demonstration purposes.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.optimize_pid import PIDOptimizer


def main():
    """Run a quick PID optimization demo."""
    print("\n" + "=" * 70)
    print("PID CONTROLLER OPTIMIZATION - QUICK DEMO")
    print("=" * 70)
    print("\nThis is a quick demo with reduced parameters for fast results.")
    print("For full optimization, use: python src/training/optimize_pid.py\n")
    print("=" * 70 + "\n")
    
    # Create optimizer with reduced parameters for quick demo
    optimizer = PIDOptimizer(
        scenario="normal",
        num_episodes=3,  # Fewer episodes for speed
        optimization_method="grid_search",
        render=False,
    )
    
    try:
        # Run grid search with coarser grid
        print("Running grid search with 3x3x3 = 27 configurations...")
        print("Each configuration tested for 3 episodes.\n")
        
        optimizer.grid_search(
            kp_range=(0.3, 1.0, 3),  # 3 values
            ki_range=(0.05, 0.2, 3),  # 3 values
            kd_range=(0.1, 0.5, 3),  # 3 values
        )
        
        # Save results
        optimizer.save_results(filename="pid_optimization_quick_demo.json")
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE!")
        print("=" * 70)
        print("\nTo run full optimization with more episodes and finer grid:")
        print("  python src/training/optimize_pid.py --scenario normal --method grid_search --episodes 10")
        print("\nTo test the optimized controller:")
        print("  python tests/test_pid_controller.py --scenario normal --kp <best_kp> --ki <best_ki> --kd <best_kd> --render")
        print("\n" + "=" * 70 + "\n")
    
    finally:
        optimizer.close()


if __name__ == "__main__":
    main()

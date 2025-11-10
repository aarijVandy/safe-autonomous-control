"""
Training script for Hierarchical Constrained RL Controller.

This script trains a hierarchical controller combining:
- Rule-based maneuver policy (high-level discrete decisions)
- SAC trajectory policy (low-level continuous control with Lagrangian constraints)

Training starts with normal (easy) traffic and can be extended to adversarial scenarios
using curriculum learning.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from src.environments.nade_wrapper import create_nade_env
from src.controllers.constrained_controller import (
    HierarchicalController,
    RuleBasedManeuverPolicy,
)
from src.controllers.sac_trajectory_policy import SACTrajectoryPolicy
from config.environment_config import get_scenario_config


class HierarchicalTrainer:
    """
    Trainer for hierarchical constrained RL controller.
    
    Handles:
    - Training loop with episode management
    - TensorBoard logging
    - Model checkpointing
    - Constraint violation tracking
    - Curriculum learning (optional)
    """
    
    def __init__(
        self,
        env_config: dict,
        controller_config: dict,
        training_config: dict,
        log_dir: str = 'runs',
        checkpoint_dir: str = 'checkpoints',
    ):
        """
        Initialize trainer.
        
        Args:
            env_config: Environment configuration
            controller_config: Controller configuration
            training_config: Training hyperparameters
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.env_config = env_config
        self.controller_config = controller_config
        self.training_config = training_config
        
        # Create directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = create_nade_env(**env_config)
        self.controller = self._create_controller()
        run_name = f"hierarchical_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(self.log_dir / run_name))
        
        # Training state
        self.episode = 0
        self.total_timesteps = 0
        self.best_reward = -np.inf
        self.best_constraint_satisfaction = 0.0
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.constraint_violations = []
    
    def _create_controller(self) -> HierarchicalController:
        """Create hierarchical controller with maneuver and trajectory policies."""
        # Create maneuver policy
        maneuver_config = self.controller_config.get('maneuver_policy', {})
        maneuver_policy = RuleBasedManeuverPolicy(maneuver_config)
        
        # Create trajectory policy
        trajectory_config = self.controller_config.get('trajectory_policy', {})
        trajectory_policy = SACTrajectoryPolicy(trajectory_config)
        
        # Create hierarchical controller
        hierarchical_config = self.controller_config.get('hierarchical', {})
        controller = HierarchicalController(
            maneuver_policy=maneuver_policy,
            trajectory_policy=trajectory_policy,
            config=hierarchical_config,
        )
        
        return controller
    
    def train_episode(self) -> dict:
        """
        Run one training episode.
        
        Returns:
            episode_info: Dictionary with episode statistics
        """
        obs, info = self.env.reset()
        self.controller.reset()
        
        episode_reward = 0.0
        episode_costs = {
            'ttc': [],
            'headway': [],
            'clearance': [],
            'jerk': [],
        }
        episode_length = 0
        done = False
        truncated = False
        
        # Episode loop
        while not (done or truncated):
            # Select action using hierarchical controller
            action = self.controller.compute_action(obs, info)
            
            # Step environment
            next_obs, env_reward, done, truncated, info = self.env.step(action)
            
            # Compute costs
            costs = self.controller.compute_costs(obs, action, next_obs)
            
            # Compute shaped reward (overriding env reward for better learning)
            shaped_reward = self.controller.trajectory_policy.compute_reward(obs, action)
            
            # Add transition to replay buffer
            self.controller.trajectory_policy.replay_buffer.add(
                obs, action, shaped_reward, next_obs, done, costs
            )
            
            # Track statistics
            episode_reward += shaped_reward
            for constraint_name, cost_value in costs.items():
                episode_costs[constraint_name].append(cost_value)
            
            # Update policy (if enough samples)
            if len(self.controller.trajectory_policy.replay_buffer) > self.training_config['batch_size']:
                update_metrics = self.controller.trajectory_policy.update()
                
                # Log update metrics
                if update_metrics and episode_length % 10 == 0:
                    for key, value in update_metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, self.total_timesteps)
            
            # Update state
            obs = next_obs
            episode_length += 1
            self.total_timesteps += 1
        
        # Episode complete - update dual variables
        expected_costs = self.controller.get_expected_costs()
        self.controller.update_dual_variables(expected_costs)
        
        # Compile episode info
        episode_info = {
            'reward': episode_reward,
            'length': episode_length,
            'expected_costs': expected_costs,
            'avg_costs': {
                name: np.mean(costs) if len(costs) > 0 else 0.0
                for name, costs in episode_costs.items()
            },
            'violation_counts': {
                name: sum(1 for c in costs if c > 1e-6)
                for name, costs in episode_costs.items()
            },
            'collision': info.get('episode_metrics', {}).get('collisions', 0) > 0,
        }
        
        return episode_info
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            eval_metrics: Dictionary of evaluation metrics
        """
        self.controller.trajectory_policy.eval()
        
        eval_rewards = []
        eval_lengths = []
        eval_collisions = 0
        eval_constraint_violations = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            self.controller.reset()
            
            episode_reward = 0.0
            episode_violations = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.controller.compute_action(obs, info)
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                costs = self.controller.compute_costs(obs, action, next_obs)
                episode_reward += reward
                episode_violations += sum(1 for c in costs.values() if c > 1e-6)
                
                obs = next_obs
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(info.get('episode_metrics', {}).get('timesteps', 0))
            eval_collisions += int(info.get('episode_metrics', {}).get('collisions', 0) > 0)
            eval_constraint_violations.append(episode_violations)
        
        self.controller.trajectory_policy.train()
        
        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'collision_rate': eval_collisions / num_episodes,
            'mean_violations': np.mean(eval_constraint_violations),
            'success_rate': 1.0 - (eval_collisions / num_episodes),
        }
        
        return metrics
    
    def save_checkpoint(self, filename: str = None):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename (default: auto-generated)
        """
        if filename is None:
            filename = f'hierarchical_sac_ep{self.episode}_ts{self.total_timesteps}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        self.controller.trajectory_policy.save(str(checkpoint_path))
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, total_episodes: int):
        """
        Main training loop.
        
        Args:
            total_episodes: Total number of training episodes
        """
        print(f"\n{'='*60}")
        print("Starting Hierarchical Constrained RL Training")
        print(f"{'='*60}")
        print(f"Environment: {self.env_config.get('name', 'unknown')}")
        print(f"Total episodes: {total_episodes}")
        print(f"Observation dim: {self.controller.trajectory_policy.obs_dim}")
        print(f"Action dim: {self.controller.trajectory_policy.act_dim}")
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        # Training loop
        for episode in tqdm(range(total_episodes), desc="Training"):
            self.episode = episode
            
            # Train one episode
            episode_info = self.train_episode()
            
            # Log episode metrics
            self.writer.add_scalar('episode/reward', episode_info['reward'], episode)
            self.writer.add_scalar('episode/length', episode_info['length'], episode)
            self.writer.add_scalar('episode/collision', int(episode_info['collision']), episode)
            
            # Log costs
            for constraint_name, cost in episode_info['avg_costs'].items():
                self.writer.add_scalar(f'cost/{constraint_name}', cost, episode)
            
            # Log constraint violations
            total_violations = sum(episode_info['violation_counts'].values())
            self.writer.add_scalar('episode/total_violations', total_violations, episode)
            
            # Log dual variables
            for name, value in self.controller.dual_variables.items():
                self.writer.add_scalar(f'dual/{name}', value, episode)
            
            # Track statistics
            self.episode_rewards.append(episode_info['reward'])
            self.episode_lengths.append(episode_info['length'])
            self.constraint_violations.append(total_violations)
            
            # Periodic evaluation
            if (episode + 1) % self.training_config.get('eval_frequency', 50) == 0:
                eval_metrics = self.evaluate(num_episodes=10)
                
                print(f"\n[Episode {episode + 1}] Evaluation:")
                print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"  Success Rate: {eval_metrics['success_rate']:.2%}")
                print(f"  Collision Rate: {eval_metrics['collision_rate']:.2%}")
                print(f"  Mean Violations: {eval_metrics['mean_violations']:.1f}")
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f'eval/{key}', value, episode)
                
                # Save best model
                if eval_metrics['success_rate'] > self.best_constraint_satisfaction:
                    self.best_constraint_satisfaction = eval_metrics['success_rate']
                    self.save_checkpoint('best_model.pt')
                    print(f"  ✓ New best model saved! (Success rate: {eval_metrics['success_rate']:.2%})")
            
            # Periodic checkpointing
            if (episode + 1) % self.training_config.get('checkpoint_frequency', 100) == 0:
                self.save_checkpoint()
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        
        final_eval = self.evaluate(num_episodes=20)
        print("\nFinal Evaluation (20 episodes):")
        for key, value in final_eval.items():
            print(f"  {key}: {value}")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Close
        self.writer.close()
        self.env.close()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Hierarchical Constrained RL Controller')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--scenario', type=str, default='normal', help='Traffic scenario (normal, mild, moderate, severe)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Model checkpoint directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Environment configuration (normal traffic for initial training)
    env_scenario = get_scenario_config(args.scenario)
    env_config = {
        'adversarial_mode': env_scenario['adversarial_mode'],
        'vehicles_count': env_scenario['vehicles_count'],
        'duration': env_scenario['duration'],
        'adversarial_intensity': env_scenario.get('adversarial_intensity', 0.0),
    }
    
    # Controller configuration
    controller_config = {
        'maneuver_policy': {
            'target_speed': 19.44,  # 70 km/h
            'speed_diff_threshold': 3.0,
            'ttc_front_threshold': 3.0,
            'ttc_rear_threshold': 3.0,
            'time_advantage_threshold': 2.0,
            'min_dwell_time': 2.0,
        },
        'trajectory_policy': {
            'obs_dim': 18,
            'act_dim': 1,
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
            'dual_learning_rate': 0.005,  # Reduced for more stable learning
            # Reward weights (rebalanced)
            'w_speed': 2.0,
            'w_lane': 0.3,
            'w_progress': 1.0,
            'w_comfort': 0.05,
            # Exploration
            'exploration_bonus': True,
            'exploration_decay': 0.9999,
        },
        'hierarchical': {
            'maneuver_frequency': 1.0,  # Hz
            'trajectory_frequency': 10.0,  # Hz
        },
    }
    
    # Training configuration
    training_config = {
        'batch_size': 256,
        'eval_frequency': 50,
        'checkpoint_frequency': 100,
    }
    
    # Create trainer
    trainer = HierarchicalTrainer(
        env_config=env_config,
        controller_config=controller_config,
        training_config=training_config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Train
    trainer.train(total_episodes=args.episodes)


if __name__ == '__main__':
    main()

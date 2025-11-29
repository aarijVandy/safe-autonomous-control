"""
SAC-based Trajectory Policy 

This module implements a Soft Actor-Critic (SAC) policy for continuous control
with explicit safety constraints enforced via Lagrangian multipliers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque
import random

from .constrained_controller import ConstrainedController


class ReplayBuffer:
    """Experience replay buffer for SAC training."""
    
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension
            act_dim: Action dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Preallocate arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 4), dtype=np.float32)  # 4 constraint types
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        costs: Dict[str, float]
    ):
        """Add transition to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        # Pack costs into array
        self.costs[self.ptr] = [
            costs.get('ttc', 0.0),
            costs.get('headway', 0.0),
            costs.get('clearance', 0.0),
            costs.get('jerk', 0.0),
        ]
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch = (
            torch.FloatTensor(self.observations[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_observations[idxs]),
            torch.FloatTensor(self.dones[idxs]),
            torch.FloatTensor(self.costs[idxs]),
        )
        
        return batch
    
    def __len__(self):
        return self.size


class Actor(nn.Module):
    """Stochastic policy network (Gaussian) for SAC."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        """
        Initialize actor network.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        
        # Action bounds for highway driving
        # For 2D actions: [acceleration, steering]
        # acceleration: [-3, 2] m/s² → scale=2.5, bias=-0.5
        # steering: [-1, 1] → scale=1.0, bias=0.0
        if act_dim == 1:
            self.action_scale = torch.FloatTensor([2.5])
            self.action_bias = torch.FloatTensor([-0.5])
        else:
            self.action_scale = torch.FloatTensor([2.5, 1.0])
            self.action_bias = torch.FloatTensor([-0.5, 0.0])
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
        
        # Initialize weights to bias toward zero steering (straight driving)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights to bias toward straight driving (zero steering).
        
        The mean output layer is initialized with very small weights so that
        untrained policy outputs near-zero actions in the normalized space,
        which translates to:
        - Small acceleration (close to 0 after denormalization)
        - Zero steering (straight driving)
        
        This prevents random turning behavior before the policy is trained.
        """
        # Initialize mean output layer with very small weights
        # This makes untrained policy output ~0 in normalized space [-1, 1]
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mean.bias, 0.0)
        
        # Initialize log_std output layer conservatively
        # Start with lower exploration noise
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.constant_(self.log_std.bias, -2.0)  # Start with low variance
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean and log_std.
        
        Args:
            obs: Observation tensor
        
        Returns:
            mean: Action mean
            log_std: Log standard deviation
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy with reparameterization trick.
        
        Args:
            obs: Observation tensor
        
        Returns:
            action: Sampled action (tanh-squashed)
            log_prob: Log probability of action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample with reparameterization
        
        # Tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log prob with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob


class Critic(nn.Module):
    """Q-function network for SAC."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        """
        Initialize critic network.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc4 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning two Q-values (for clipped double Q-learning).
        
        Args:
            obs: Observation tensor
            action: Action tensor
        
        Returns:
            q1: Q-value from first network
            q2: Q-value from second network
        """
        x = torch.cat([obs, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2


class SACTrajectoryPolicy(ConstrainedController):
    """
    SAC-based trajectory policy with Lagrangian constraint enforcement.
    
    Implements Soft Actor-Critic with:
    - Augmented reward: r_aug = r - Σ λ_k * c_k
    - Automatic temperature tuning
    - Clipped double Q-learning
    - Target network updates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SAC trajectory policy.
        
        Args:
            config: Configuration including:
                - obs_dim: Observation dimension
                - act_dim: Action dimension
                - hidden_dim: Hidden layer size
                - lr: Learning rate
                - gamma: Discount factor
                - tau: Target network update rate
                - alpha: Entropy temperature (or 'auto')
                - buffer_size: Replay buffer capacity
                - batch_size: Training batch size
        """
        super().__init__(config)
        
        # Network dimensions
        self.obs_dim = config.get('obs_dim', 18)
        self.act_dim = config.get('act_dim', 1)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Training hyperparameters
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 256)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.actor = Actor(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.act_dim, self.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Automatic entropy tuning
        self.target_entropy = -self.act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        
        # Replay buffer
        buffer_size = config.get('buffer_size', 1000000)
        self.replay_buffer = ReplayBuffer(buffer_size, self.obs_dim, self.act_dim)
        
        # Training state
        self.training_mode = True
        self.updates = 0
        
        # Reward shaping parameters (rebalanced for better learning)
        self.w_speed = config.get('w_speed', 2.0)        # Increased for stronger signal
        self.w_lane = config.get('w_lane', 0.3)          # Reduced, less critical
        self.w_progress = config.get('w_progress', 1.0)  # Increased to encourage progress
        self.w_comfort = config.get('w_comfort', 0.05)   # Reduced, comfort is secondary
        
        # Exploration bonus parameters
        self.exploration_bonus_enabled = config.get('exploration_bonus', True)
        self.exploration_decay = config.get('exploration_decay', 0.9999)
        
        # Previous state for jerk computation
        self.prev_action = None
    
    def compute_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute action using current policy.
        
        Args:
            observation: Current observation
            info: Additional information (includes current_maneuver from hierarchical controller)
        
        Returns:
            action: Control action [acceleration, lane_change_command]
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get maneuver decision from hierarchical controller
        current_maneuver = info.get('current_maneuver', 0) if info else 0
        
        if self.training_mode:
            # Sample action from policy
            with torch.no_grad():
                action, _ = self.actor.sample(obs_tensor)
            action = action.cpu().numpy()[0]
        else:
            # Use deterministic policy (mean) - no bias, trust the learned policy
            with torch.no_grad():
                mean, _ = self.actor.forward(obs_tensor)
                action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
            action = action.cpu().numpy()[0]
        
        # Handle action dimensionality
        if self.act_dim == 1:
            # Legacy 1D action (longitudinal only) - add neutral lateral component
            action = np.array([action[0], 0.0])
        elif self.act_dim == 2:
            # 2D action: override lateral component based on maneuver policy
            # action[0] = longitudinal (from RL policy)
            # action[1] = lateral (influenced by maneuver policy)
            
            # Get current lane from observation
            ego_lane_norm = observation[4]  # Normalized lane index
            num_lanes = 3
            ego_lane = int(ego_lane_norm * (num_lanes - 1))
            
            # Maneuver constants: KEEP_LANE=0, CHANGE_LEFT=1, CHANGE_RIGHT=2
            if current_maneuver == 1:  # CHANGE_LEFT
                # Only allow left change if not in leftmost lane (lane 0)
                if ego_lane > 0:
                    action[1] = -0.8  # Strong left signal
                else:
                    action[1] = 0.0  # Already at left boundary, keep lane
            elif current_maneuver == 2:  # CHANGE_RIGHT
                # Only allow right change if not in rightmost lane (lane 2)
                if ego_lane < (num_lanes - 1):
                    action[1] = 0.8   # Strong right signal
                else:
                    action[1] = 0.0  # Already at right boundary, keep lane
            elif current_maneuver == 0:  # KEEP_LANE
                # Allow moderate corrections from RL policy for obstacle avoidance
                # Also enforce lane boundaries
                if ego_lane == 0:
                    # Leftmost lane: only allow neutral or right corrections
                    action[1] = np.clip(action[1], 0.0, 0.5)
                elif ego_lane == num_lanes - 1:
                    # Rightmost lane: only allow neutral or left corrections
                    action[1] = np.clip(action[1], -0.5, 0.0)
                else:
                    # Middle lane: allow moderate corrections both ways
                    action[1] = np.clip(action[1], -0.5, 0.5)
        
        # Clip to valid ranges
        action[0] = np.clip(action[0], -3.0, 2.0)   # acceleration
        if len(action) > 1:
            action[1] = np.clip(action[1], -1.0, 1.0)  # lane change command
        
        return action
    
    def compute_costs(
        self, 
        observation: np.ndarray, 
        action: np.ndarray,
        next_observation: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute constraint costs using barrier functions.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
        
        Returns:
            costs: Dictionary of constraint costs
        """
        costs = {}
        
        # Extract relevant features from observation
        lead_distance = observation[0]
        lead_rel_velocity = observation[1]
        ttc = observation[2]
        ego_velocity = observation[3]
        
        right_distance = observation[6]
        right_lateral = observation[8]
        
        left_distance = observation[9]
        left_lateral = observation[11]
        
        # TTC constraint: TTC ≥ τ_min
        tau_min = self.constraint_thresholds['ttc_min']
        if lead_distance < 100 and lead_distance > 0:  # Lead vehicle present
            if ttc < tau_min:
                violation = tau_min - ttc
                costs['ttc'] = violation  # Linear barrier function
            else:
                costs['ttc'] = 0.0
        else:
            costs['ttc'] = 0.0
        
        # Headway constraint: s ≥ s_0 + T_h * v_ego
        s_0 = self.constraint_thresholds['headway_s0']
        T_h = self.constraint_thresholds['headway_T']
        if lead_distance < 100 and lead_distance > 0:
            desired_headway = s_0 + T_h * ego_velocity
            if lead_distance < desired_headway:
                violation = desired_headway - lead_distance
                costs['headway'] = violation  # Linear barrier function
            else:
                costs['headway'] = 0.0
        else:
            costs['headway'] = 0.0
        
        # Lateral clearance constraint during lane change
        d_min = self.constraint_thresholds['clearance_min']
        clearance_violations = []
        
        # Check right side
        if right_lateral < d_min and abs(right_distance) < 20:
            clearance_violations.append(d_min - right_lateral)
        
        # Check left side
        if left_lateral < d_min and abs(left_distance) < 20:
            clearance_violations.append(d_min - left_lateral)
        
        if clearance_violations:
            costs['clearance'] = sum(clearance_violations)  # Linear barrier function
        else:
            costs['clearance'] = 0.0
        
        # Jerk constraint: |jerk| ≤ j_max
        j_max = self.constraint_thresholds['jerk_max']
        if self.prev_action is not None and next_observation is not None:
            dt = 0.1  # 10 Hz control frequency
            jerk = (action[0] - self.prev_action[0]) / dt
            if abs(jerk) > j_max:
                violation = abs(jerk) - j_max
                costs['jerk'] = violation  # Linear barrier function
            else:
                costs['jerk'] = 0.0
        else:
            costs['jerk'] = 0.0
        
        # Track costs
        for constraint_name, cost_value in costs.items():
            self.track_cost(constraint_name, cost_value)
        
        # Update previous action
        self.prev_action = action.copy()
        
        return costs
    
    def compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        Compute shaped reward using potential functions with safety recovery bonuses.
        
        Args:
            observation: Current observation
            action: Action taken
        
        Returns:
            reward: Shaped reward value
        """
        # Extract features
        lead_distance = observation[0]
        lead_rel_velocity = observation[1]
        ttc = observation[2]
        ego_velocity = observation[3]
        lane_offset = observation[5]
        
        # Speed tracking reward - IMPROVED: More positive signal
        target_speed = 16.67  # m/s (60 km/h) - matching traffic speed
        speed_error = abs(ego_velocity - target_speed)
        # Give POSITIVE reward for being close to target, small penalty for deviation
        if speed_error < 2.0:
            # Good speed - positive reward
            r_speed = self.w_speed * (1.0 - speed_error / 2.0)  # [0, 2.0]
        else:
            # Too far from target - small penalty
            r_speed = -self.w_speed * 0.5 * speed_error  # Moderate penalty
        
        # Lane centering reward - IMPROVED: Less penalty for small deviations
        if abs(lane_offset) < 1.0:
            r_lane = self.w_lane * 0.5  # Bonus for good centering
        else:
            r_lane = -self.w_lane * 0.2 * (lane_offset ** 2)  # Reduced penalty
        
        # Progress reward - INCREASED: Always positive for moving forward
        # Scale to [0, w_progress] where max at target_speed
        r_progress = self.w_progress * 2.0 * (ego_velocity / target_speed)  # Doubled
        
        # Comfort reward (penalize large accelerations)
        r_comfort = -self.w_comfort * (action[0] ** 2)
        
        # Lane change rewards (for 2D action space) - STRONG PENALTIES
        r_lane_change = 0.0
        if len(action) > 1:
            lane_cmd = action[1]
            ego_lane_norm = observation[4]
            num_lanes = 3
            ego_lane = int(ego_lane_norm * (num_lanes - 1))
            
            # STRONG penalty for lane changes - DISCOURAGE HEAVILY
            lane_change_magnitude = abs(lane_cmd)
            if lane_change_magnitude > 0.7:  # Strong lane change command
                # Heavy penalty: -2.0 per timestep during lane change
                r_lane_change -= 2.0
            elif lane_change_magnitude > 0.33:  # Moderate lane change
                # Moderate penalty: -1.0 per timestep
                r_lane_change -= 1.0
            
            # LARGE bonus for keeping lane (staying centered)
            if lane_change_magnitude < 0.2:  # Keeping lane stable
                r_lane_change += 0.5  # Reward stability
            
            # Additional bonus for right lane (default/safe lane)
            if ego_lane == num_lanes - 1 and lane_change_magnitude < 0.2:
                r_lane_change += 0.3  # Extra bonus for rightmost lane stability
        
        # === Safety Recovery Bonuses - INCREASED ===
        r_safety = 0.0
        
        # Reward for maintaining safe TTC - DOUBLED bonuses
        tau_min = self.constraint_thresholds['ttc_min']
        tau_comfort = tau_min * 1.5  # Comfortable TTC threshold
        if lead_distance < 100 and lead_distance > 0:
            if ttc > tau_comfort:
                # Bonus for maintaining comfortable TTC
                r_safety += 1.0  # Increased from 0.5
            elif ttc > tau_min:
                # Small bonus for being safe but close
                safety_margin = (ttc - tau_min) / (tau_comfort - tau_min)
                r_safety += 0.5 * safety_margin  # Increased from 0.2
        else:
            # No vehicle ahead - big bonus for clear road
            r_safety += 0.5
        
        # Reward for maintaining safe headway - DOUBLED bonuses
        s_0 = self.constraint_thresholds['headway_s0']
        T_h = self.constraint_thresholds['headway_T']
        if lead_distance < 100 and lead_distance > 0:
            desired_headway = s_0 + T_h * ego_velocity
            if lead_distance > desired_headway * 1.2:
                # Bonus for comfortable following distance
                r_safety += 0.6  # Increased from 0.3
            elif lead_distance > desired_headway:
                # Small bonus for safe following
                safety_margin = (lead_distance - desired_headway) / (desired_headway * 0.2)
                r_safety += 0.3 * safety_margin  # Increased from 0.15
        
        # Reward for smooth control (low jerk) - DOUBLED
        if self.prev_action is not None:
            dt = 0.1
            jerk = abs((action[0] - self.prev_action[0]) / dt)
            j_max = self.constraint_thresholds['jerk_max']
            if jerk < j_max * 0.5:
                # Bonus for very smooth control
                smoothness = 1.0 - (jerk / (j_max * 0.5))
                r_safety += 0.4 * smoothness  # Increased from 0.2
        
        # === SURVIVAL BONUS - Reward for staying alive ===
        # Large positive reward for each timestep without crashing/going off-road
        r_survival = 1.0  # +1.0 per timestep for staying alive and on-road
        
        # Additional bonus for staying in valid lane (not at boundaries)
        if abs(lane_offset) < 2.0:  # Well within lane boundaries
            r_survival += 0.5  # Extra bonus for safe lane position
        
        # Total reward
        reward = r_speed + r_lane + r_progress + r_comfort + r_lane_change + r_safety + r_survival
        
        return reward
    
    def update(self) -> Dict[str, float]:
        """
        Perform one SAC update step with Lagrangian constraints.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        obs, actions, rewards, next_obs, dones, costs = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        costs = costs.to(self.device)
        
        # Compute augmented rewards using Lagrangian multipliers
        lambda_values = torch.FloatTensor([
            self.dual_variables['lambda_ttc'],
            self.dual_variables['lambda_headway'],
            self.dual_variables['lambda_clearance'],
            self.dual_variables['lambda_jerk'],
        ]).to(self.device)
        
        # r_aug = r - Σ λ_k * c_k
        cost_penalty = (costs * lambda_values).sum(dim=1, keepdim=True)
        augmented_rewards = rewards - cost_penalty
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            q_target = augmented_rewards + self.gamma * (1 - dones) * q_next
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.updates += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'avg_q': q_new.mean().item(),
            'avg_cost_penalty': cost_penalty.mean().item(),
        }
    
    def train(self):
        """Set policy to training mode."""
        self.training_mode = True
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        """Set policy to evaluation mode."""
        self.training_mode = False
        self.actor.eval()
        self.critic.eval()
    
    def save(self, path: str):
        """Save policy networks."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'dual_variables': self.dual_variables,
        }, path)
    
    def load(self, path: str):
        """Load policy networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.dual_variables = checkpoint['dual_variables']
    
    def reset(self):
        """Reset episode state."""
        super().reset()
        self.prev_action = None

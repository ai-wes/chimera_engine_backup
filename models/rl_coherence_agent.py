import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random

class CoherenceRLAgent(nn.Module):
    """
    Reinforcement Learning agent that learns to modulate generator outputs
    to maximize biological coherence using PPO (Proximal Policy Optimization).
    """
    
    def __init__(self, latent_dim: int, num_modalities: int = 7, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        
        # Policy network: takes latent representation and outputs modulation parameters
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head: outputs mean and log_std for continuous actions
        self.actor_mean = nn.Linear(hidden_dim, latent_dim)
        self.actor_log_std = nn.Linear(hidden_dim, latent_dim)
        
        # Value head: estimates expected future reward
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Modality-specific modulation networks
        self.modality_modulators = nn.ModuleDict({
            f'mod_{i}': nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.Tanh()  # Output in [-1, 1] for modulation
            ) for i in range(num_modalities)
        })
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # PPO hyperparameters
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Exploration parameters
        self.exploration_noise = 0.1
        self.min_log_std = -5
        self.max_log_std = 2
        
    def forward(self, latent: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RL agent.
        
        Args:
            latent: Latent representation [batch, latent_dim]
            deterministic: If True, use mean action (no sampling)
            
        Returns:
            action: Modulation action [batch, latent_dim]
            log_prob: Log probability of the action
            value: Estimated value of the state
        """
        # Extract features
        features = self.policy_net(latent)
        
        # Get action distribution parameters
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std(features)
        action_log_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std)
        action_std = torch.exp(action_log_std)
        
        # Get value estimate
        value = self.value_head(features).squeeze(-1)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(value)
        else:
            # Sample action from normal distribution
            noise = torch.randn_like(action_mean)
            action = action_mean + noise * action_std
            
            # Compute log probability
            log_prob = -0.5 * ((action - action_mean) / action_std).pow(2) - 0.5 * np.log(2 * np.pi) - action_log_std
            log_prob = log_prob.sum(dim=-1)
        
        # Apply tanh squashing to keep actions bounded
        action = torch.tanh(action)
        
        return action, log_prob, value
    
    def modulate_latent(self, latent: torch.Tensor, modality_idx: int, deterministic: bool = False) -> torch.Tensor:
        """
        Modulate a latent representation for a specific modality.
        
        Args:
            latent: Original latent representation [batch, latent_dim]
            modality_idx: Index of the modality to modulate
            deterministic: If True, use deterministic policy
            
        Returns:
            Modulated latent representation
        """
        # Get modulation action
        action, _, _ = self.forward(latent, deterministic)
        
        # Concatenate latent with action
        combined = torch.cat([latent, action], dim=-1)
        
        # Apply modality-specific modulation
        modulation = self.modality_modulators[f'mod_{modality_idx}'](combined)
        
        # Apply residual connection with learned modulation
        modulated_latent = latent + 0.1 * modulation  # Small modulation to start
        
        return modulated_latent
    
    def compute_advantage(self, rewards: torch.Tensor, values: torch.Tensor, 
                         next_values: torch.Tensor, dones: torch.Tensor, 
                         gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[-1]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def store_transition(self, state: torch.Tensor, action: torch.Tensor, 
                        log_prob: torch.Tensor, value: torch.Tensor, 
                        reward: float, done: bool):
        """Store a transition in the replay buffer."""
        self.memory.append({
            'state': state.cpu(),
            'action': action.cpu(),
            'log_prob': log_prob.cpu(),
            'value': value.cpu(),
            'reward': reward,
            'done': done
        })
    
    def update(self, optimizer: torch.optim.Optimizer, epochs: int = 4, 
              batch_size: int = 32) -> Dict[str, float]:
        """
        Update the agent using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < batch_size:
            return {}
        
        # Convert memory to tensors
        states = torch.stack([m['state'] for m in self.memory]).squeeze(1)
        actions = torch.stack([m['action'] for m in self.memory]).squeeze(1)
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])
        old_values = torch.stack([m['value'] for m in self.memory])
        rewards = torch.tensor([m['reward'] for m in self.memory], dtype=torch.float32)
        dones = torch.tensor([m['done'] for m in self.memory], dtype=torch.float32)
        
        # Move to device
        device = next(self.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        old_values = old_values.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        # Compute advantages
        with torch.no_grad():
            _, _, next_values = self.forward(states, deterministic=True)
            advantages, returns = self.compute_advantage(rewards, old_values, next_values, dones)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):
            # Shuffle indices
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, values = self.forward(batch_states, deterministic=False)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus (encourage exploration)
                entropy = -new_log_probs.mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory after update
        self.memory.clear()
        
        num_updates = epochs * (len(states) // batch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def compute_intrinsic_reward(self, coherence_score: float, 
                                modality_scores: Dict[str, float],
                                previous_coherence: float = 0.73) -> float:
        """
        Compute intrinsic reward based on coherence improvements.
        
        Args:
            coherence_score: Overall biological coherence score
            modality_scores: Individual modality coherence scores
            previous_coherence: Previous epoch's coherence score
            
        Returns:
            Intrinsic reward value
        """
        # Base reward for coherence (scaled to encourage high scores)
        base_reward = coherence_score ** 2  # Quadratic scaling favors higher scores
        
        # Progressive improvement bonus - exponentially rewards larger improvements
        improvement = coherence_score - previous_coherence
        if improvement > 0:
            # Exponential scaling: 1% = 10 points, 2% = 22 points, 4% = 52 points, etc.
            improvement_bonus = (np.exp(improvement * 100) - 1) * 10
            
            # Additional milestone bonuses for breaking key barriers
            if coherence_score >= 0.95 and previous_coherence < 0.95:
                improvement_bonus += 100  # Huge bonus for reaching target
            elif coherence_score >= 0.90 and previous_coherence < 0.90:
                improvement_bonus += 50
            elif coherence_score >= 0.85 and previous_coherence < 0.85:
                improvement_bonus += 25
            elif coherence_score >= 0.80 and previous_coherence < 0.80:
                improvement_bonus += 15
            elif coherence_score >= 0.75 and previous_coherence < 0.75:
                improvement_bonus += 10
        else:
            # Small penalty for regression, but not too harsh to allow exploration
            improvement_bonus = improvement * 5
        
        # Penalty for low modality scores (progressive penalty)
        modality_penalty = 0
        for mod, score in modality_scores.items():
            if score < 0.5:
                # Quadratic penalty - gets much worse as score drops
                modality_penalty += ((0.5 - score) ** 2) * 20
            elif score < 0.7:
                # Linear penalty for moderate scores
                modality_penalty += (0.7 - score) * 5
        
        # Bonus for balanced modality scores (variance penalty)
        scores = list(modality_scores.values())
        if scores:
            variance = np.var(scores)
            # Reward low variance (balanced scores)
            balance_bonus = np.exp(-variance * 10) * 5
        else:
            balance_bonus = 0
        
        # Progressive exploration bonus for high coherence regions
        if coherence_score > 0.9:
            exploration_bonus = ((coherence_score - 0.9) ** 2) * 100
        elif coherence_score > 0.8:
            exploration_bonus = ((coherence_score - 0.8) ** 2) * 50
        elif coherence_score > 0.75:
            exploration_bonus = ((coherence_score - 0.75) ** 2) * 25
        else:
            exploration_bonus = 0
        
        # Momentum bonus - reward consistent improvement
        # (This would need to track history, simplified here)
        momentum_bonus = 0
        if improvement > 0.01:  # If improving by more than 1%
            momentum_bonus = improvement * 50  # Extra reward for momentum
        
        # Total reward with emphasis on improvement
        reward = (base_reward + 
                 improvement_bonus * 2 +  # Double weight on improvement
                 balance_bonus + 
                 exploration_bonus + 
                 momentum_bonus - 
                 modality_penalty)
        
        # Add small random exploration bonus to prevent getting stuck
        exploration_noise = np.random.normal(0, 0.1)
        reward += exploration_noise
        
        return reward
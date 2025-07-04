import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque, defaultdict
import random

class ModalitySpecificRLAgent(nn.Module):
    """
    Individual RL agent for a specific modality that learns to optimize
    its biological coherence independently.
    """
    
    def __init__(self, modality_name: str, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.modality_name = modality_name
        self.latent_dim = latent_dim
        
        # Modality-specific policy network
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head: outputs mean and log_std for continuous actions
        self.actor_mean = nn.Linear(hidden_dim // 2, latent_dim)
        self.actor_log_std = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Value head: estimates expected future reward
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=5000)
        
        # PPO hyperparameters
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Exploration parameters
        self.min_log_std = -5
        self.max_log_std = 2
        
        # Track modality-specific metrics
        self.previous_coherence = 0.0
        self.coherence_history = []
        self.best_coherence = 0.0
        
    def forward(self, latent: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the modality-specific RL agent.
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
    
    def compute_modality_reward(self, current_coherence: float, 
                               modality_specific_metrics: Dict[str, float]) -> float:
        """
        Compute reward specific to this modality's characteristics.
        """
        # Base improvement reward
        improvement = current_coherence - self.previous_coherence
        
        if improvement > 0:
            # Exponential scaling for improvements
            improvement_bonus = (np.exp(improvement * 100) - 1) * 10
        else:
            # Smaller penalty for regression
            improvement_bonus = improvement * 3
        
        # Modality-specific bonuses based on biological constraints
        modality_bonus = 0
        
        if self.modality_name == 'gene':
            # Gene expression specific rewards
            if 'ribosomal_high_expr' in modality_specific_metrics:
                if modality_specific_metrics['ribosomal_high_expr'] > 0.8:
                    modality_bonus += 20
            if 'housekeeping_stable' in modality_specific_metrics:
                if modality_specific_metrics['housekeeping_stable'] > 0.9:
                    modality_bonus += 15
            if 'expression_range' in modality_specific_metrics:
                # Reward appropriate dynamic range
                if 8 < modality_specific_metrics['expression_range'] < 15:
                    modality_bonus += 10
                    
        elif self.modality_name == 'protein':
            # Protein specific rewards
            if 'gene_protein_correlation' in modality_specific_metrics:
                if modality_specific_metrics['gene_protein_correlation'] > 0.7:
                    modality_bonus += 25
            if 'abundance_distribution' in modality_specific_metrics:
                if modality_specific_metrics['abundance_distribution'] > 0.8:
                    modality_bonus += 10
                    
        elif self.modality_name == 'methylation':
            # Methylation specific rewards
            if 'valid_beta_values' in modality_specific_metrics:
                if modality_specific_metrics['valid_beta_values'] > 0.99:
                    modality_bonus += 20
            if 'cpg_island_pattern' in modality_specific_metrics:
                if modality_specific_metrics['cpg_island_pattern'] > 0.7:
                    modality_bonus += 15
            if 'global_methylation_level' in modality_specific_metrics:
                # Reward appropriate global methylation (not too high/low)
                level = modality_specific_metrics['global_methylation_level']
                if 0.2 < level < 0.4:
                    modality_bonus += 10
                    
        elif self.modality_name == 'variant':
            # Variant specific rewards
            if 'sparsity' in modality_specific_metrics:
                # Variants should be sparse
                if modality_specific_metrics['sparsity'] < 0.05:
                    modality_bonus += 20
            if 'hotspot_concentration' in modality_specific_metrics:
                if modality_specific_metrics['hotspot_concentration'] > 0.6:
                    modality_bonus += 15
                    
        elif self.modality_name == 'metabolite':
            # Metabolite specific rewards
            if 'pathway_consistency' in modality_specific_metrics:
                if modality_specific_metrics['pathway_consistency'] > 0.8:
                    modality_bonus += 20
            if 'enzyme_correlation' in modality_specific_metrics:
                if modality_specific_metrics['enzyme_correlation'] > 0.7:
                    modality_bonus += 15
                    
        elif self.modality_name == 'microbiome':
            # Microbiome specific rewards
            if 'diversity_index' in modality_specific_metrics:
                # Reward appropriate diversity
                diversity = modality_specific_metrics['diversity_index']
                if 0.6 < diversity < 0.9:
                    modality_bonus += 15
            if 'abundance_distribution' in modality_specific_metrics:
                if modality_specific_metrics['abundance_distribution'] > 0.7:
                    modality_bonus += 10
                    
        elif self.modality_name == 'clinical':
            # Clinical specific rewards
            if 'categorical_validity' in modality_specific_metrics:
                if modality_specific_metrics['categorical_validity'] > 0.95:
                    modality_bonus += 20
            if 'continuous_ranges' in modality_specific_metrics:
                if modality_specific_metrics['continuous_ranges'] > 0.9:
                    modality_bonus += 15
        
        # Milestone bonuses for crossing coherence thresholds
        milestone_bonus = 0
        if current_coherence >= 0.95 and self.previous_coherence < 0.95:
            milestone_bonus = 100
        elif current_coherence >= 0.90 and self.previous_coherence < 0.90:
            milestone_bonus = 50
        elif current_coherence >= 0.85 and self.previous_coherence < 0.85:
            milestone_bonus = 25
        elif current_coherence >= 0.80 and self.previous_coherence < 0.80:
            milestone_bonus = 15
        
        # Base coherence reward
        base_reward = current_coherence ** 2 * 50
        
        # Total reward
        total_reward = (base_reward + 
                       improvement_bonus * 2 +  # Weight improvement heavily
                       modality_bonus + 
                       milestone_bonus)
        
        # Add small exploration noise
        exploration_noise = np.random.normal(0, 0.05)
        total_reward += exploration_noise
        
        return total_reward
    
    def update_coherence_tracking(self, current_coherence: float):
        """Update coherence tracking for this modality."""
        self.previous_coherence = current_coherence
        self.coherence_history.append(current_coherence)
        if len(self.coherence_history) > 20:
            self.coherence_history.pop(0)
        if current_coherence > self.best_coherence:
            self.best_coherence = current_coherence


class MultiModalityRLSystem(nn.Module):
    """
    System that manages multiple modality-specific RL agents and coordinates
    their learning for optimal multi-modal generation.
    """
    
    def __init__(self, latent_dim: int, modality_names: List[str], hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.modality_names = modality_names
        
        # Create individual RL agents for each modality
        self.modality_agents = nn.ModuleDict({
            mod: ModalitySpecificRLAgent(mod, latent_dim, hidden_dim)
            for mod in modality_names
        })
        
        # Coordination network that learns to balance modality contributions
        self.coordinator = nn.Sequential(
            nn.Linear(latent_dim * len(modality_names), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(modality_names)),
            nn.Softmax(dim=-1)
        )
        
        # Track system-level metrics
        self.system_coherence_history = []
        self.modulation_strengths = {mod: 0.1 for mod in modality_names}
        
    def forward(self, latent_dict: Dict[str, torch.Tensor], 
                deterministic: bool = False) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through all modality-specific agents.
        
        Args:
            latent_dict: Dictionary of latent representations for each modality
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary of (action, log_prob, value) for each modality
        """
        results = {}
        
        for modality, latent in latent_dict.items():
            if modality in self.modality_agents:
                action, log_prob, value = self.modality_agents[modality](latent, deterministic)
                results[modality] = (action, log_prob, value)
        
        return results
    
    def modulate_latents(self, latent_dict: Dict[str, torch.Tensor], 
                        deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Apply modality-specific modulations to latent representations.
        """
        # Get actions from each modality agent
        actions = self.forward(latent_dict, deterministic)
        
        # Apply modulations with modality-specific strengths
        modulated_latents = {}
        for modality, latent in latent_dict.items():
            if modality in actions:
                action, _, _ = actions[modality]
                modulation_strength = self.modulation_strengths[modality]
                modulated_latents[modality] = latent + modulation_strength * action
            else:
                modulated_latents[modality] = latent
        
        # Optional: Apply coordination weights
        if len(latent_dict) > 1:
            # Concatenate all latents
            all_latents = torch.cat(list(latent_dict.values()), dim=-1)
            coord_weights = self.coordinator(all_latents)
            
            # Apply coordination weights to modulations
            for i, modality in enumerate(self.modality_names):
                if modality in modulated_latents:
                    weight = coord_weights[:, i:i+1]
                    modulated_latents[modality] = (latent_dict[modality] * (1 - weight) + 
                                                  modulated_latents[modality] * weight)
        
        return modulated_latents
    
    def compute_rewards(self, coherence_results: Dict[str, float],
                       modality_specific_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute rewards for each modality agent.
        """
        rewards = {}
        
        for modality in self.modality_names:
            if modality in self.modality_agents:
                # Get modality-specific coherence
                modality_coherence = coherence_results.get(f'modality_{modality}', 0.0)
                
                # Get modality-specific metrics
                mod_metrics = modality_specific_metrics.get(modality, {})
                
                # Compute reward
                reward = self.modality_agents[modality].compute_modality_reward(
                    modality_coherence, mod_metrics
                )
                rewards[modality] = reward
                
                # Update tracking
                self.modality_agents[modality].update_coherence_tracking(modality_coherence)
        
        return rewards
    
    def update_modulation_strengths(self, coherence_improvements: Dict[str, float]):
        """
        Dynamically adjust modulation strengths based on performance.
        """
        for modality, improvement in coherence_improvements.items():
            if modality in self.modulation_strengths:
                current_strength = self.modulation_strengths[modality]
                
                if improvement > 0.02:  # Big improvement
                    self.modulation_strengths[modality] = min(0.3, current_strength * 1.2)
                elif improvement > 0.01:  # Good improvement
                    self.modulation_strengths[modality] = min(0.25, current_strength * 1.1)
                elif improvement < -0.01:  # Regression
                    self.modulation_strengths[modality] = max(0.05, current_strength * 0.9)
                else:  # Check for plateau
                    agent = self.modality_agents[modality]
                    if len(agent.coherence_history) >= 5:
                        recent_variance = np.var(agent.coherence_history[-5:])
                        if recent_variance < 0.0001:  # Plateau detected
                            self.modulation_strengths[modality] = min(0.4, current_strength * 1.5)
    
    def store_transitions(self, states: Dict[str, torch.Tensor],
                         actions: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                         rewards: Dict[str, float]):
        """
        Store transitions in each modality agent's replay buffer.
        """
        for modality in self.modality_names:
            if modality in states and modality in actions and modality in rewards:
                action, log_prob, value = actions[modality]
                self.modality_agents[modality].memory.append({
                    'state': states[modality][0:1].cpu(),  # Store first sample
                    'action': action[0:1].cpu(),
                    'log_prob': log_prob[0:1].cpu(),
                    'value': value[0:1].cpu(),
                    'reward': rewards[modality],
                    'done': False
                })
    
    def update_all_agents(self, optimizers: Dict[str, torch.optim.Optimizer],
                         epochs: int = 4, batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        """
        Update all modality-specific agents using PPO.
        """
        all_metrics = {}
        
        for modality, agent in self.modality_agents.items():
            if modality in optimizers and len(agent.memory) >= batch_size:
                # Use the same PPO update logic from the base agent
                metrics = self._ppo_update(agent, optimizers[modality], epochs, batch_size)
                all_metrics[modality] = metrics
        
        return all_metrics
    
    def _ppo_update(self, agent: ModalitySpecificRLAgent, 
                    optimizer: torch.optim.Optimizer,
                    epochs: int, batch_size: int) -> Dict[str, float]:
        """
        Standard PPO update for a single agent.
        """
        # Convert memory to tensors
        states = torch.stack([m['state'] for m in agent.memory]).squeeze(1)
        actions = torch.stack([m['action'] for m in agent.memory]).squeeze(1)
        old_log_probs = torch.stack([m['log_prob'] for m in agent.memory])
        old_values = torch.stack([m['value'] for m in agent.memory])
        rewards = torch.tensor([m['reward'] for m in agent.memory], dtype=torch.float32)
        dones = torch.tensor([m['done'] for m in agent.memory], dtype=torch.float32)
        
        # Move to device
        device = next(agent.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        old_values = old_values.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = old_values[t + 1]
            
            delta = rewards[t] + 0.99 * next_value * (1 - dones[t]) - old_values[t]
            gae = delta + 0.99 * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(epochs):
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
                _, new_log_probs, values = agent(batch_states, deterministic=False)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - agent.clip_param, 1 + agent.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy = -new_log_probs.mean()
                
                # Total loss
                loss = policy_loss + agent.value_loss_coef * value_loss - agent.entropy_coef * entropy
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), agent.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Clear memory after update
        agent.memory.clear()
        
        return {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy': total_entropy / num_updates if num_updates > 0 else 0
        }
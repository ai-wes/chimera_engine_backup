import torch
import torch.nn as nn
from diffusers import UNet1DModel

import torch
from torch import nn
import math
from typing import Optional, Union

class FlowMatcher(nn.Module):
    """
    Enhanced Flow Matching implementation with proper scheduling
    and better noise handling for different modalities.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.num_steps = cfg.diffusion.timesteps
        
        # Enhanced denoising network
        self.denoising_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + 1, 4 * cfg.latent_dim),  # +1 for timestep
            nn.SiLU(),
            nn.LayerNorm(4 * cfg.latent_dim),
            nn.Linear(4 * cfg.latent_dim, 4 * cfg.latent_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * cfg.latent_dim, cfg.latent_dim)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, cfg.latent_dim)
        )
        
        # Per-modality noise schedules (as mentioned in the reference)
        self.register_buffer("per_mod_noise", torch.tensor(cfg.diffusion.per_mod_noise))
        
        # Learnable conditioning projection
        self.condition_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        
    def get_timestep_embedding(self, timesteps, embedding_dim=128):
        """
        Create sinusoidal timestep embeddings (similar to transformer positional encoding)
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
    
    def add_noise(self, z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """
        Add noise according to flow matching schedule
        """
        # Normalize timesteps to [0, 1]
        t_norm = t.float() / self.num_steps
        
        # Flow matching interpolation: z_t = (1-t)*z_0 + t*noise
        alpha_t = (1 - t_norm).view(-1, 1)
        sigma_t = t_norm.view(-1, 1)
        
        z_t = alpha_t * z0 + sigma_t * noise
        return z_t
    
    def forward(self, z0: torch.Tensor, condition: torch.Tensor):
        """
        Forward pass for training
        """
        batch_size = z0.shape[0]
        device = z0.device
        
        # Sample timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(z0)
        
        # Add noise according to schedule
        z_t = self.add_noise(z0, noise, t)
        
        # Get timestep embeddings
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_embed(t_emb)
        
        # Project condition
        cond_proj = self.condition_proj(condition)
        
        # Combine inputs for denoising network
        # Add timestep as an extra dimension
        t_norm = (t.float() / self.num_steps).unsqueeze(1)
        net_input = torch.cat([z_t + cond_proj + t_emb, t_norm], dim=1)
        
        # Predict the flow (or noise)
        predicted_flow = self.denoising_net(net_input)
        
        # Flow matching loss (predict the direction to the noise)
        target = noise - z0  # The flow direction
        loss = nn.functional.mse_loss(predicted_flow, target)
        
        # Return predicted flow ("fake"), loss, and target flow ("real")
        return predicted_flow, loss, target
    
    def sample(self, num_samples: int, condition: torch.Tensor, device: torch.device):
        """
        Generate samples using flow matching
        """
        if len(condition.shape) == 2:
            z = torch.randn(num_samples, condition.shape[-1], device=device)
            condition = condition[:num_samples] if condition.shape[0] >= num_samples else condition.repeat(num_samples // condition.shape[0] + 1, 1)[:num_samples]
        else:
            z = torch.randn(num_samples, condition.shape[-2], device=device)
            condition = condition[:num_samples] if condition.shape[0] >= num_samples else condition.repeat(num_samples // condition.shape[0] + 1, 1, 1)[:num_samples]
        
        # Reverse sampling process
        dt = 1.0 / self.num_steps
        
        for i in reversed(range(self.num_steps)):
            t = torch.full((num_samples,), i, device=device)
            
            # Get timestep embeddings
            t_emb = self.get_timestep_embedding(t)
            t_emb = self.time_embed(t_emb)
            
            # Project condition
            cond_proj = self.condition_proj(condition)
            
            with torch.no_grad():
                # Prepare input
                t_norm = (t.float() / self.num_steps).unsqueeze(1)
                net_input = torch.cat([z + cond_proj + t_emb, t_norm], dim=1)
                
                # Predict flow
                predicted_flow = self.denoising_net(net_input)
                
                # Update z using predicted flow
                z = z - dt * predicted_flow
        
        return z

class KnowledgeConditionedEnhancedDiffusion(nn.Module):
    """
    Enhanced knowledge-conditioned diffusion with better architecture
    """
    
    def __init__(self, latent_dim, cfg):
        super().__init__()
        self.flow_matcher = FlowMatcher(cfg)
        
        # Enhanced conditioning network
        self.condition_encoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        """
        Forward pass with enhanced conditioning
        """
        # Enhance the conditioning signal
        enhanced_condition = self.condition_encoder(condition)
        
        # Flow matcher now returns (predicted_flow, loss, target)
        return self.flow_matcher(z, enhanced_condition)
    
    
        
    def sample(self, num_samples: int, condition: torch.Tensor, device: torch.device):
        """
        Sample with enhanced conditioning
        """
        enhanced_condition = self.condition_encoder(condition.to(device))
        return self.flow_matcher.sample(num_samples, enhanced_condition, device) 

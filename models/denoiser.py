import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: [batch_size] tensor of timesteps
        Returns:
            embeddings: [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalLayerNorm(nn.Module):
    """Layer normalization with conditional scaling"""
    
    def __init__(self, dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.condition_proj = nn.Linear(condition_dim, dim * 2)
        
    def forward(self, x, condition):
        """
        Args:
            x: [batch, ..., dim]
            condition: [batch, condition_dim]
        """
        x = self.norm(x)
        scale, shift = self.condition_proj(condition).chunk(2, dim=-1)
        
        # Reshape for broadcasting
        for _ in range(len(x.shape) - len(scale.shape)):
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            
        return x * (1 + scale) + shift


class DenoisingBlock(nn.Module):
    """Single denoising block with time conditioning"""
    
    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        
        # Time projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Main processing
        self.norm1 = ConditionalLayerNorm(dim, dim)
        self.ff1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        self.norm2 = ConditionalLayerNorm(dim, dim)
        self.ff2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, t_emb):
        """
        Args:
            x: [batch, dim]
            t_emb: [batch, time_dim]
        """
        # Time conditioning
        t_proj = self.time_proj(t_emb)
        
        # First block
        h = self.norm1(x, t_proj)
        h = self.ff1(h)
        x = x + h
        
        # Second block
        h = self.norm2(x, t_proj)
        h = self.ff2(h)
        
        # Gating
        gate = self.gate(x)
        x = x + gate * h
        
        return x


class ModalitySpecificDenoiser(nn.Module):
    """Denoiser specialized for a specific modality"""
    
    def __init__(self, input_dim, hidden_dim, num_blocks=4, time_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 for concatenated noisy + model output
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        # Denoising blocks
        self.blocks = nn.ModuleList([
            DenoisingBlock(hidden_dim, time_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Noise level estimation
        self.noise_estimator = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, model_output, noisy_input, t):
        """
        Args:
            model_output: [batch, input_dim] - output from main model
            noisy_input: [batch, input_dim] - noisy input data
            t: [batch] - timesteps
        Returns:
            denoised: [batch, input_dim]
        """
        batch_size = model_output.shape[0]
        
        # Concatenate inputs
        x = torch.cat([model_output, noisy_input], dim=-1)
        
        # Project to hidden dim
        x = self.input_proj(x)
        
        # Get time embeddings
        t_emb = self.time_embed(t)
        
        # Apply denoising blocks
        for block in self.blocks:
            x = block(x, t_emb)
        
        # Estimate noise level
        noise_level = self.noise_estimator(torch.cat([x, t_emb], dim=-1))
        
        # Output projection
        denoised = self.output_proj(x)
        
        # Adaptive denoising based on estimated noise level
        denoised = noise_level * denoised + (1 - noise_level) * model_output
        
        return denoised


class DiffusionDenoiser(nn.Module):
    """Multi-modal diffusion denoiser"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Modality dimensions (matching main model)
        self.modality_dims = {
            'gene_expression': 20000,
            'dnam': 27000,
            'mirna': 1000,
            'cnv': 20000,
            'protein': 200,
            'clinical': 50,
            'mutation': 100
        }
        
        # Create modality-specific denoisers
        self.denoisers = nn.ModuleList([
            ModalitySpecificDenoiser(
                input_dim=dim,
                hidden_dim=cfg.latent_dim,
                num_blocks=4,
                time_dim=128,
                dropout=cfg.mask_dropout
            )
            for dim in self.modality_dims.values()
        ])
        
        # Cross-modal conditioning
        self.cross_modal_attn = nn.MultiheadAttention(
            cfg.latent_dim,
            num_heads=8,
            dropout=cfg.mask_dropout,
            batch_first=True
        )
        
        # Global time embedding
        self.global_time_embed = nn.Sequential(
            TimeEmbedding(128),
            nn.Linear(128, cfg.latent_dim),
            nn.GELU()
        )
        
    def forward(self, model_output, noisy_input, t, modality_idx=None):
        """
        Args:
            model_output: Output from main model (single modality or list)
            noisy_input: Noisy input (single modality or list)
            t: Timesteps [batch]
            modality_idx: If provided, denoise only this modality
        Returns:
            denoised: Denoised output(s)
        """
        # Single modality case
        if modality_idx is not None:
            return self.denoisers[modality_idx](model_output, noisy_input, t)
        
        # Multi-modal case
        if not isinstance(model_output, list):
            raise ValueError("For multi-modal denoising, inputs must be lists")
        
        batch_size = model_output[0].shape[0]
        denoised_outputs = []
        
        # Get global time embedding
        global_t_emb = self.global_time_embed(t)
        
        # First pass: initial denoising
        hidden_states = []
        for i, (out, noisy, denoiser) in enumerate(zip(model_output, noisy_input, self.denoisers)):
            # Get hidden state before final projection
            x = torch.cat([out, noisy], dim=-1)
            x = denoiser.input_proj(x)
            
            t_emb = denoiser.time_embed(t)
            for block in denoiser.blocks:
                x = block(x, t_emb)
                
            hidden_states.append(x)
        
        # Cross-modal refinement
        if len(hidden_states) > 1:
            # Stack hidden states
            hidden_stack = torch.stack(hidden_states, dim=1)  # [batch, num_modalities, hidden_dim]
            
            # Add global time embedding
            hidden_stack = hidden_stack + global_t_emb.unsqueeze(1)
            
            # Cross-modal attention
            refined, _ = self.cross_modal_attn(hidden_stack, hidden_stack, hidden_stack)
            
            # Final denoising with refined features
            for i, (out, noisy, denoiser) in enumerate(zip(model_output, noisy_input, self.denoisers)):
                refined_hidden = refined[:, i, :]
                
                # Estimate noise level
                t_emb = denoiser.time_embed(t)
                noise_level = denoiser.noise_estimator(torch.cat([refined_hidden, t_emb], dim=-1))
                
                # Output projection
                denoised = denoiser.output_proj(refined_hidden)
                
                # Adaptive denoising
                denoised = noise_level * denoised + (1 - noise_level) * out
                denoised_outputs.append(denoised)
        else:
            # Single modality, use standard denoising
            for i, (out, noisy, denoiser) in enumerate(zip(model_output, noisy_input, self.denoisers)):
                denoised = denoiser(out, noisy, t)
                denoised_outputs.append(denoised)
        
        return denoised_outputs

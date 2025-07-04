import torch
import torch.nn as nn
import numpy as np

class StatisticalHead(nn.Module):
    """
    A decoder head that generates outputs by learning the parameters
    of a target distribution, rather than the values directly.
    This is critical for achieving high statistical coherence.
    """
    def __init__(self, input_dim: int, output_dim: int, dist_type: str = 'normal'):
        super().__init__()
        self.output_dim = output_dim
        self.dist_type = dist_type

        # The network learns to predict the MEAN and the LOG-STANDARD-DEVIATION
        self.mean_predictor = nn.Linear(input_dim, output_dim)
        self.std_predictor = nn.Linear(input_dim, output_dim)
        
        # This is a new, crucial component. It learns a "diversity factor" per sample.
        self.diversity_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output a value between 0 and 1
        )
        
        # Initialize std predictor biases to a less conservative value
        # This helps prevent variance collapse
        nn.init.constant_(self.std_predictor.bias, -1.0)

    def forward(self, x: torch.Tensor, target_stats: dict, diversity_scale: float = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input from the previous decoder layer.
            target_stats (dict): A dict with 'mean' and 'std' of the real data.
            diversity_scale (float): Optional override for diversity scaling. If None, uses default.
        """
        base_mean = target_stats['mean'].to(x.device)
        base_std = target_stats['std'].to(x.device)
        
        # --- The Core Logic ---
        # 1. Learn the *shape* of the distribution (deviations from the base)
        # The network learns to predict offsets from the real data's mean, scaled by its std
        mean_offsets = torch.tanh(self.mean_predictor(x)) * base_std 
        # The network learns to predict multipliers for the real data's std
        # Tightened clamp range to keep variance closer to target
        std_multipliers = torch.exp(torch.clamp(self.std_predictor(x), -0.5, 0.5))
        
        final_mean = base_mean + mean_offsets
        final_std = base_std * std_multipliers
        
        # Variance preservation: ensure variance stays close to target stats
        # Tighten the range to [0.7, 1.5] of target std for better statistical matching
        min_std = base_std * 0.7  # At least 70% of target std
        max_std = base_std * 1.5  # At most 150% of target std
        final_std = torch.clamp(final_std, min=min_std, max=max_std)

        # --- IMPROVED DIVERSITY MECHANISM ---
        # 2. Generate correlated noise to preserve pathway structure
        # Instead of independent random noise per gene, we generate structured noise
        diversity_factor = self.diversity_predictor(x) # Shape: [batch_size, 1]
        
        if diversity_scale is None:
            diversity_scale = 0.3 if self.training else 0.1  # Less diversity during evaluation
        
        # Generate low-dimensional noise and project to preserve correlations
        batch_size, feature_dim = final_mean.shape
        # Use much lower dimension for noise (e.g., 50 instead of 5000/15000)
        noise_dim = min(50, feature_dim // 100)  
        low_dim_noise = torch.randn(batch_size, noise_dim, device=final_mean.device)
        
        # Project to full dimension - this creates correlated noise patterns
        if not hasattr(self, 'noise_projection'):
            # Create a fixed random projection matrix (preserves structure across batches)
            self.register_buffer('noise_projection', 
                               torch.randn(noise_dim, feature_dim, device=final_mean.device) / np.sqrt(noise_dim))
        
        structured_noise = torch.matmul(low_dim_noise, self.noise_projection)
        batch_diversity_offset = structured_noise * base_std * diversity_scale * diversity_factor
        final_mean_with_diversity = final_mean + batch_diversity_offset

        # 3. Sample from the final, diversity-aware distribution
        if self.training:
            epsilon = torch.randn_like(final_std)
            output = final_mean_with_diversity + epsilon * final_std
        else:
            output = final_mean
        
        # Apply distribution-specific final activation with numerical stability
        if self.dist_type == 'beta':
            # Methylation beta-values are in [0, 1].
            # Clamp input to sigmoid to prevent extreme values causing NaNs
            output_clamped = torch.clamp(output, min=-15.0, max=15.0)
            # Apply sigmoid
            beta_values = torch.sigmoid(output_clamped)
            # FIX: Add a small epsilon to prevent exact 0 or 1, and clamp to ensure strict [0, 1]
            epsilon = 1e-6
            return torch.clamp(beta_values, min=epsilon, max=1.0 - epsilon)
        else:
            # For normal distribution, return the output directly
            return output
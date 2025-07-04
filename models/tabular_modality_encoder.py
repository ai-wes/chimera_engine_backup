import torch
import torch.nn as nn

class TabularModalityEncoder(nn.Module):
    """Simple MLP encoder for tabular modality data"""
    def __init__(self, in_dim, latent_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim + latent_dim) // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, in_dim]
        Returns:
            Tensor of shape [batch_size, latent_dim]
        """
        return self.encoder(x)

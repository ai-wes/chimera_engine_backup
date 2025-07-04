import torch
import math
from torch import nn

def gradient_centralization(weight):
    """Apply gradient centralization to MHA weights to prevent gradient explosion"""
    if weight.grad is not None and len(weight.grad.shape) > 1:
        # Center gradients by subtracting mean across output dimension
        weight.grad = weight.grad - weight.grad.mean(dim=tuple(range(1, len(weight.grad.shape))), keepdim=True)

class HierarchicalAttention(nn.Module):
    """
    Four-level mask-aware multi-head attention:
    molecular → pathway → systemic → phenotype.
    
    This is an enhanced version based on the improved implementation
    that provides level-specific attention blocks and better mask management.
    """
    def __init__(self, d_model: int, num_heads: int = 8, 
                 hierarchy_levels: list = None):
        super().__init__()
        
        if hierarchy_levels is None:
            hierarchy_levels = ["molecular", "pathway", "systemic", "phenotype"]
        
        self.hierarchy_levels = hierarchy_levels
        self.d_model = d_model
        
        # Create separate attention blocks for each hierarchy level
        self.blocks = nn.ModuleDict({
            lvl: nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1
            ) for lvl in hierarchy_levels
        })
        
        # Layer normalization for each level
        self.layer_norms = nn.ModuleDict({
            lvl: nn.LayerNorm(d_model) for lvl in hierarchy_levels
        })
        
        # Feed-forward networks for each level
        self.feed_forwards = nn.ModuleDict({
            lvl: nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(0.1)
            ) for lvl in hierarchy_levels
        })
        
        # Final output normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Register buffers for attention masks
        self.register_buffer("attn_masks", 
                           torch.zeros(len(hierarchy_levels), 1, 1))
        
        # Register gradient centralization hook for all MHA weights
        for block in self.blocks.values():
            for name, param in block.named_parameters():
                if 'weight' in name and len(param.shape) > 1:
                    param.register_hook(lambda grad, p=param: self._apply_gradient_centralization(grad, p))
    
    def _apply_gradient_centralization(self, grad, param):
        """Apply gradient centralization to prevent gradient explosion"""
        if grad is not None and len(grad.shape) > 1:
            return grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
        return grad

    def set_mask(self, level: str, mask: torch.Tensor) -> None:
        """
        Set attention mask for a specific hierarchy level.
        
        Args:
            level: Hierarchy level name
            mask: Attention mask with shape (B, N, N) 
                  where 0 = allow attention, -inf = disallow
        """
        if level not in self.hierarchy_levels:
            raise ValueError(f"Unknown level: {level}. Available: {self.hierarchy_levels}")
        
        idx = self.hierarchy_levels.index(level)
        self.attn_masks[idx] = mask

    def get_default_masks(self, batch_size: int, seq_len: int, device: torch.device):
        """Create default hierarchy-specific masks"""
        masks = {}
        
        # Molecular level - allow intra-modality attention
        molecular_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            molecular_mask[:, i, i] = 0.0  # self-attention allowed
        masks["molecular"] = molecular_mask
        
        # Pathway level - gene-protein and gene-methylation interactions
        pathway_mask = torch.full((batch_size, seq_len, seq_len), -float('inf'), device=device)
        if seq_len >= 3:  # Assuming gene=0, protein=1, methylation=2
            pathway_mask[:, 0, 1] = 0.0  # gene -> protein
            pathway_mask[:, 1, 0] = 0.0  # protein -> gene
            pathway_mask[:, 0, 2] = 0.0  # gene -> methylation
            pathway_mask[:, 2, 0] = 0.0  # methylation -> gene
        masks["pathway"] = pathway_mask
        
        # Systemic level - broader biological connections
        systemic_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
        masks["systemic"] = systemic_mask
        
        # Phenotype level - all modalities to clinical (assuming clinical is last)
        phenotype_mask = torch.full((batch_size, seq_len, seq_len), -float('inf'), device=device)
        if seq_len >= 2:
            phenotype_mask[:, :, -1] = 0.0  # all -> clinical
            phenotype_mask[:, -1, :] = 0.0  # clinical -> all
        masks["phenotype"] = phenotype_mask
        
        return masks

    def forward(self, x: torch.Tensor, masks: dict = None) -> torch.Tensor:
        """
        Forward pass through hierarchical attention layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            masks: Dictionary of masks for each level, or None for defaults
            
        Returns:
            Enhanced representations after hierarchical attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Get masks (use provided or create defaults)
        if masks is None:
            masks = self.get_default_masks(batch_size, seq_len, x.device)
        
        # Process through each hierarchy level sequentially
        for level in self.hierarchy_levels:
            # Get mask for this level
            mask = masks.get(level, None)
            
            # Reshape mask for MultiheadAttention if needed
            if mask is not None and mask.dim() == 3:
                # PyTorch expects mask shape (N*num_heads, L, S) for batch_first=True
                # But for 2D masks, it can handle (L, S) 
                # So we'll use the first sample's mask as representative
                mask = mask[0]  # Take first batch element: (seq_len, seq_len)
            
            # Layer normalization before attention
            x_norm = self.layer_norms[level](x)
            
            # Multi-head attention with residual connection
            attn_output, attn_weights = self.blocks[level](
                query=x_norm, key=x_norm, value=x_norm,
                attn_mask=mask
            )
            x = x + attn_output
            
            # Feed-forward with residual connection
            ff_output = self.feed_forwards[level](self.layer_norms[level](x))
            x = x + ff_output
        
        return self.final_norm(x)

    def forward_single_level(self, x: torch.Tensor, level: str, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through a single hierarchy level.
        Useful for ablation studies or level-specific analysis.
        """
        if level not in self.hierarchy_levels:
            raise ValueError(f"Unknown level: {level}")
        
        # Reshape mask for MultiheadAttention if needed
        if mask is not None and mask.dim() == 3:
            mask = mask[0]  # Take first batch element: (seq_len, seq_len)
        
        # Layer normalization before attention
        x_norm = self.layer_norms[level](x)
        
        # Multi-head attention with residual connection
        attn_output, _ = self.blocks[level](
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=mask
        )
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forwards[level](self.layer_norms[level](x))
        x = x + ff_output
        
        return x

    def get_attention_weights(self, x: torch.Tensor, level: str, mask: torch.Tensor = None):
        """Get attention weights for analysis and visualization"""
        if level not in self.hierarchy_levels:
            raise ValueError(f"Unknown level: {level}")
        
        # Reshape mask for MultiheadAttention if needed
        if mask is not None and mask.dim() == 3:
            mask = mask[0]  # Take first batch element: (seq_len, seq_len)
        
        x_norm = self.layer_norms[level](x)
        _, attn_weights = self.blocks[level](
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=mask
        )
        return attn_weights

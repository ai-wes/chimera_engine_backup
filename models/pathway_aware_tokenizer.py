import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

class PathwayAwareTokenizer(nn.Module):
    """
    Pathway-Aware Tokenization that structures the latent space according to biological pathways.
    
    Instead of a flat 256-dim latent space, we create structured tokens:
    - One token per KEGG/Reactome pathway
    - Residual "misc" tokens for unmapped features
    - Learned adapter that bins each gene/protein into its dominant pathway token
    
    This forces the model's capacity to align with biological hierarchy,
    preventing memorization of idiosyncratic correlations.
    """
    
    def __init__(self, latent_dim=256, pathway_config_path=None, num_misc_tokens=32):
        super().__init__()
        
        # Load pathway mappings if available
        self.pathway_mappings = self._load_pathway_mappings(pathway_config_path)
        self.num_pathways = len(self.pathway_mappings) if self.pathway_mappings else 50  # Default
        self.num_misc_tokens = num_misc_tokens
        self.total_tokens = self.num_pathways + self.num_misc_tokens
        
        # Token dimension (divide latent space among tokens)
        self.token_dim = latent_dim // self.total_tokens
        
        # Ensure token_dim is divisible by number of attention heads (4)
        # Round down to nearest multiple of 4
        self.token_dim = (self.token_dim // 4) * 4
        if self.token_dim == 0:
            self.token_dim = 4  # Minimum viable dimension
            
        self.latent_dim = self.token_dim * self.total_tokens  # Adjusted to be divisible
        
        # Learnable pathway tokens
        self.pathway_tokens = nn.Parameter(torch.randn(self.num_pathways, self.token_dim))
        self.misc_tokens = nn.Parameter(torch.randn(self.num_misc_tokens, self.token_dim))
        
        # Gene-to-pathway assignment network
        self.gene_pathway_adapter = nn.ModuleDict()
        self.protein_pathway_adapter = nn.ModuleDict()
        
        # Create adapters for gene and protein modalities
        for modality, in_dim in [('gene', 5000), ('protein', 226)]:
            adapter = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, self.num_pathways + self.num_misc_tokens)  # Soft assignment to pathways
            )
            if modality == 'gene':
                self.gene_pathway_adapter = adapter
            else:
                self.protein_pathway_adapter = adapter
        
        # Token-to-feature decoders
        self.token_decoders = nn.ModuleDict({
            'gene': nn.Linear(self.token_dim, 5000),
            'protein': nn.Linear(self.token_dim, 226),
            'methylation': nn.Linear(self.token_dim, 100),
            'variant': nn.Linear(self.token_dim, 100),
            'metabolite': nn.Linear(self.token_dim, 3),
            'microbiome': nn.Linear(self.token_dim, 1000),
            'clinical': nn.Linear(self.token_dim, 100)
        })
        
        # Modality-to-token projections for non-gene/protein modalities
        self.modality_projections = nn.ModuleDict({
            'methylation': nn.Linear(100, self.token_dim),
            'variant': nn.Linear(100, self.token_dim),
            'metabolite': nn.Linear(3, self.token_dim),
            'microbiome': nn.Linear(1000, self.token_dim),
            'clinical': nn.Linear(100, self.token_dim)
        })
        
        # Token mixer for cross-pathway communication
        # Adjust number of heads if token_dim is too small
        num_heads = min(4, self.token_dim)
        self.token_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=num_heads,
                dim_feedforward=self.token_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
    def _load_pathway_mappings(self, config_path):
        """Load pathway mappings from config file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Create synthetic pathway names for demonstration
            return {f"pathway_{i}": i for i in range(50)}
    
    def encode(self, features_dict):
        """
        Encode features into pathway-aware tokens.
        
        Args:
            features_dict: Dict of modality -> features
            
        Returns:
            tokens: (batch_size, num_tokens, token_dim)
        """
        batch_size = next(iter(features_dict.values())).shape[0]
        device = next(iter(features_dict.values())).device
        
        # Initialize token activations
        token_activations = torch.zeros(batch_size, self.total_tokens, self.token_dim).to(device)
        
        # Process gene features
        if 'gene' in features_dict:
            gene_features = features_dict['gene']
            # Get pathway assignments (soft)
            pathway_logits = self.gene_pathway_adapter(gene_features)
            pathway_weights = F.softmax(pathway_logits, dim=-1)  # (batch, num_tokens)
            
            # Weighted combination of tokens based on pathway assignment
            all_tokens = torch.cat([self.pathway_tokens, self.misc_tokens], dim=0)  # (total_tokens, token_dim)
            
            # Compute weighted token activations
            for b in range(batch_size):
                weighted_tokens = pathway_weights[b:b+1].T @ gene_features[b:b+1]  # (total_tokens, gene_dim)
                # Project to token dimension
                token_activations[b] += all_tokens * pathway_weights[b:b+1].T
        
        # Process protein features similarly
        if 'protein' in features_dict:
            protein_features = features_dict['protein']
            pathway_logits = self.protein_pathway_adapter(protein_features)
            pathway_weights = F.softmax(pathway_logits, dim=-1)
            
            all_tokens = torch.cat([self.pathway_tokens, self.misc_tokens], dim=0)
            for b in range(batch_size):
                token_activations[b] += all_tokens * pathway_weights[b:b+1].T
        
        # For other modalities, distribute across misc tokens
        misc_start_idx = self.num_pathways
        for modality in ['methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
            if modality in features_dict and modality in self.modality_projections:
                # Use learned projection to misc tokens
                mod_features = features_dict[modality]
                mod_tokens = self.modality_projections[modality](mod_features)
                
                # Assign to rotating misc tokens
                token_idx = misc_start_idx + (hash(modality) % self.num_misc_tokens)
                token_activations[:, token_idx] += F.gelu(mod_tokens)
        
        # Mix tokens for cross-pathway communication
        tokens = self.token_mixer(token_activations)
        
        return tokens
    
    def decode(self, tokens, modality):
        """
        Decode tokens back to feature space for a specific modality.
        
        Args:
            tokens: (batch_size, num_tokens, token_dim)
            modality: Which modality to decode
            
        Returns:
            features: (batch_size, modality_dim)
        """
        # Pool tokens (weighted by learned importance)
        token_importance = F.softmax(tokens.sum(dim=-1), dim=-1)  # (batch, num_tokens)
        pooled = torch.bmm(token_importance.unsqueeze(1), tokens).squeeze(1)  # (batch, token_dim)
        
        # Decode to specific modality
        if modality in self.token_decoders:
            return self.token_decoders[modality](pooled)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def get_pathway_attention_scores(self, features_dict):
        """
        Get attention scores over pathways for interpretability.
        
        Returns:
            Dict of modality -> pathway attention scores
        """
        scores = {}
        
        if 'gene' in features_dict:
            gene_logits = self.gene_pathway_adapter(features_dict['gene'])
            scores['gene'] = F.softmax(gene_logits[:, :self.num_pathways], dim=-1)
        
        if 'protein' in features_dict:
            protein_logits = self.protein_pathway_adapter(features_dict['protein'])
            scores['protein'] = F.softmax(protein_logits[:, :self.num_pathways], dim=-1)
        
        return scores
    
    def reshape_for_attention(self, tokens):
        """
        Reshape tokens to flat latent representation for compatibility with existing models.
        
        Args:
            tokens: (batch_size, num_tokens, token_dim)
            
        Returns:
            latent: (batch_size, latent_dim)
        """
        batch_size = tokens.shape[0]
        return tokens.reshape(batch_size, -1)
    
    def reshape_from_attention(self, latent):
        """
        Reshape flat latent back to token structure.
        
        Args:
            latent: (batch_size, latent_dim)
            
        Returns:
            tokens: (batch_size, num_tokens, token_dim)
        """
        batch_size = latent.shape[0]
        return latent.reshape(batch_size, self.total_tokens, self.token_dim)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, HeteroData
import math
import os
from .hier_attention import HierarchicalAttention
from .diffusion_decoder import KnowledgeConditionedEnhancedDiffusion
from models.knowledge_guided_edn import KGEDN
from .tabular_modality_encoder import TabularModalityEncoder
from .pathway_aware_tokenizer import PathwayAwareTokenizer
from .modality_decoders import ModalityDecoders
import json


# GeneExpressionDecoder is now in modality_decoders.py


class ModalityTokenizer(nn.Module):
    """Convert raw modality data into tokens"""
    
    def __init__(self, input_dim, hidden_dim, num_tokens, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        
        # Projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        
        # Cross-attention for tokenization
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            tokens: [batch, num_tokens, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Project input
        x_proj = self.input_proj(x)  # [batch, hidden_dim]
        x_proj = x_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Expand query tokens
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [batch, num_tokens, hidden_dim]
        
        # Cross-attention
        tokens, _ = self.cross_attn(queries, x_proj, x_proj)
        
        # Output projection
        tokens = self.output_proj(tokens)
        
        return tokens


class PathwayTokenizer(nn.Module):
    """Tokenize based on biological pathways"""
    
    def __init__(self, gene_dim, hidden_dim, num_pathways=100, dropout=0.1):
        super().__init__()
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.num_pathways = num_pathways
        
        # Gene to pathway attention
        self.pathway_queries = nn.Parameter(torch.randn(num_pathways, hidden_dim))
        self.gene_proj = nn.Linear(gene_dim, hidden_dim)
        
        self.pathway_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Pathway token refinement
        self.pathway_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, gene_expr, pathway_mask=None):
        """
        Args:
            gene_expr: [batch, gene_dim]
            pathway_mask: Optional mask for pathways
        Returns:
            pathway_tokens: [batch, num_pathways, hidden_dim]
        """
        batch_size = gene_expr.shape[0]
        
        # Project genes
        gene_proj = self.gene_proj(gene_expr)  # [batch, hidden_dim]
        gene_proj = gene_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Expand pathway queries
        queries = self.pathway_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention
        pathway_tokens, attn_weights = self.pathway_attn(
            queries, gene_proj, gene_proj,
            key_padding_mask=pathway_mask
        )
        
        # Refine
        pathway_tokens = self.pathway_refine(pathway_tokens)
        
        return pathway_tokens, attn_weights


class HierarchicalTransformer(nn.Module):
    """Hierarchical transformer with biological structure"""
    
    def __init__(self, hidden_dim, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Hierarchical levels: molecular -> pathway -> system
        self.molecular_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim * 4, dropout, 
                activation='gelu', batch_first=True
            ) for _ in range(num_layers // 3)
        ])
        
        self.pathway_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim * 4, dropout,
                activation='gelu', batch_first=True
            ) for _ in range(num_layers // 3)
        ])
        
        self.system_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim * 4, dropout,
                activation='gelu', batch_first=True
            ) for _ in range(num_layers // 3)
        ])
        
        # Level transitions
        self.molecular_to_pathway = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.pathway_to_system = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: [batch, seq_len, hidden_dim]
            mask: Optional attention mask
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # Molecular level
        molecular_out = tokens
        for layer in self.molecular_layers:
            molecular_out = layer(molecular_out, src_key_padding_mask=mask)
        
        # Transition to pathway level
        pathway_out = self.molecular_to_pathway(molecular_out)
        for layer in self.pathway_layers:
            pathway_out = layer(pathway_out, src_key_padding_mask=mask)
        
        # Transition to system level
        system_out = self.pathway_to_system(pathway_out)
        for layer in self.system_layers:
            system_out = layer(system_out, src_key_padding_mask=mask)
        
        return system_out


class ModalityDecoder(nn.Module):
    """Decode tokens back to modality space"""
    
    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Token aggregation
        self.token_pool = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, output_dim)
        )
        
        # Learnable aggregation token
        self.agg_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, tokens):
        """
        Args:
            tokens: [batch, num_tokens, hidden_dim]
        Returns:
            output: [batch, output_dim]
        """
        batch_size = tokens.shape[0]
        
        # Use attention to aggregate tokens
        agg_query = self.agg_token.expand(batch_size, -1, -1)
        aggregated, _ = self.token_pool(agg_query, tokens, tokens)
        aggregated = aggregated.squeeze(1)  # [batch, hidden_dim]
        
        # Decode
        output = self.decoder(aggregated)
        
        return output


class HGTDModel(nn.Module):
    """Hierarchical Graph-Transformer Diffusion Model"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Modality dimensions
        self.modality_dims = {
            'gene_expression': 20000,
            'dnam': 27000,
            'mirna': 1000,
            'cnv': 20000,
            'protein': 200,
            'clinical': 50,
            'mutation': 100
        }
        
        hidden_dim = cfg.latent_dim
        
        # Modality tokenizers
        self.tokenizers = nn.ModuleDict({
            name: ModalityTokenizer(
                dim, hidden_dim, 
                num_tokens=min(64, dim // 100),
                dropout=cfg.mask_dropout
            )
            for name, dim in self.modality_dims.items()
        })
        
        # Pathway tokenizer for gene expression
        if cfg.use_pathway_tokenizer:
            self.pathway_tokenizer = PathwayTokenizer(
                self.modality_dims['gene_expression'],
                hidden_dim,
                num_pathways=100,
                dropout=cfg.mask_dropout
            )
        
        # Misc tokens for global context
        self.misc_tokens = nn.Parameter(
            torch.randn(1, cfg.num_misc_tokens, hidden_dim)
        )
        
        # Hierarchical transformer
        self.transformer = HierarchicalTransformer(
            hidden_dim,
            num_heads=8,
            num_layers=cfg.num_hier_blocks * 3,  # 3 levels per block
            dropout=cfg.mask_dropout
        )
        
        # Modality decoders
        self.decoders = nn.ModuleDict({
            name: ModalityDecoder(hidden_dim, dim, dropout=cfg.mask_dropout)
            for name, dim in self.modality_dims.items()
        })
        
        # Modality embeddings
        self.modality_embeds = nn.Parameter(
            torch.randn(len(self.modality_dims), hidden_dim)
        )
        
        # Position embeddings
        max_tokens = sum(min(64, d // 100) for d in self.modality_dims.values())
        max_tokens += cfg.num_misc_tokens
        if cfg.use_pathway_tokenizer:
            max_tokens += 100  # pathway tokens
            
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, hidden_dim))
        
        # Masking
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        logging.info(f"Initialized HGTD model with {max_tokens} total tokens")
        
    def forward(self, modality_data, mask_prob=0.0, return_attn=False):
        """
        Args:
            modality_data: List of tensors [batch, dim] for each modality
            mask_prob: Probability of masking tokens
            return_attn: Whether to return attention weights
        Returns:
            outputs: List of reconstructed modalities
        """
        batch_size = modality_data[0].shape[0]
        all_tokens = []
        token_to_modality = []
        
        # Tokenize each modality
        for i, (name, data) in enumerate(zip(self.modality_dims.keys(), modality_data)):
            tokens = self.tokenizers[name](data)
            
            # Add modality embedding
            mod_embed = self.modality_embeds[i].unsqueeze(0).unsqueeze(0)
            mod_embed = mod_embed.expand(batch_size, tokens.shape[1], -1)
            tokens = tokens + mod_embed
            
            all_tokens.append(tokens)
            token_to_modality.extend([i] * tokens.shape[1])
        
        # Add pathway tokens if using
        if hasattr(self, 'pathway_tokenizer') and self.cfg.use_pathway_tokenizer:
            pathway_tokens, _ = self.pathway_tokenizer(modality_data[0])  # gene expression
            all_tokens.append(pathway_tokens)
            token_to_modality.extend([-1] * pathway_tokens.shape[1])  # -1 for pathway
        
        # Add misc tokens
        misc_tokens = self.misc_tokens.expand(batch_size, -1, -1)
        all_tokens.append(misc_tokens)
        token_to_modality.extend([-2] * misc_tokens.shape[1])  # -2 for misc
        
        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens, dim=1)  # [batch, total_tokens, hidden_dim]
        
        # Add position embeddings
        pos_embed = self.pos_embed[:, :all_tokens.shape[1], :]
        all_tokens = all_tokens + pos_embed
        
        # Apply masking
        if mask_prob > 0 and self.training:
            mask = torch.rand(batch_size, all_tokens.shape[1], device=all_tokens.device) < mask_prob
            mask_embed = self.mask_token.expand(batch_size, all_tokens.shape[1], -1)
            all_tokens = torch.where(mask.unsqueeze(-1), mask_embed, all_tokens)
        else:
            mask = None
        
        # Apply hierarchical transformer
        transformed = self.transformer(all_tokens, mask=mask)
        
        # Decode each modality
        outputs = []
        start_idx = 0
        
        for i, (name, dim) in enumerate(self.modality_dims.items()):
            # Find tokens for this modality
            num_tokens = min(64, dim // 100)
            modality_tokens = transformed[:, start_idx:start_idx + num_tokens]
            start_idx += num_tokens
            
            # Decode
            output = self.decoders[name](modality_tokens)
            outputs.append(output)
        
        if return_attn:
            return outputs, None  # Could return attention weights if needed
        
        return outputs
    
    
    
    


# ClinicalDecoder is now in modality_decoders.py


class EnhancedHGTD(nn.Module):
    """
    Enhanced Hierarchical Graph-Transformer Diffusion model
    incorporating the best practices from multiple implementations.
    
    Key improvements:
    - Enhanced hierarchical attention with level-specific blocks
    - Improved diffusion with proper schedulers
    - Optional knowledge-guided energy constraints
    - Better biological coherence evaluation
    """
    
    def __init__(self, cfg, mod_in_dims):
        super().__init__()
        self.cfg = cfg
        
        # Load feature mappings if available
        self.feature_mappings = None
        try:
            import os
            feature_mappings_path = os.path.join(cfg.data_root, 'feature_mappings.json')
            if os.path.exists(feature_mappings_path):
                with open(feature_mappings_path, 'r') as f:
                    self.feature_mappings = json.load(f)
        except:
            pass
        
        # 1. Enhanced modality-wise encoders (keep existing GNN encoders)
        self.encoders = nn.ModuleDict({
            mod: TabularModalityEncoder(in_dim, cfg.latent_dim)
            for mod, in_dim in mod_in_dims.items()
        })
        
        # 2. Enhanced hierarchical attention (major improvement)
        self.hier_attn = HierarchicalAttention(
            d_model=cfg.latent_dim,
            num_heads=8,
            hierarchy_levels=["molecular", "pathway", "systemic", "phenotype"]
        )
        
        # 3. Choose diffusion approach based on config
        if getattr(cfg, 'use_energy_diffusion', False):
            # Novel KG-EDN approach with biological constraints
            self.diffusion = KGEDN(cfg)
        else:
            # Enhanced standard diffusion
            self.diffusion = KnowledgeConditionedEnhancedDiffusion(cfg.latent_dim, cfg)
        
        # 4. Enhanced critics (keep existing but add more sophisticated ones)
        # TODO: Implement LocalCritic and GlobalCritic
        # self.local_critic = LocalCritic(cfg.latent_dim)
        # self.global_critic = GlobalCritic(cfg.latent_dim)
        
        # 5. Additional components for better biological coherence
        self.modality_projector = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(cfg.latent_dim, cfg.latent_dim),
                nn.LayerNorm(cfg.latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for mod in mod_in_dims.keys()
        })
        
        # Cross-modal alignment heads
        self.alignment_heads = nn.ModuleDict({
            f"{mod1}_{mod2}": nn.Sequential(
                nn.Linear(2 * cfg.latent_dim, cfg.latent_dim),
                nn.ReLU(),
                nn.Linear(cfg.latent_dim, 1),
                nn.Sigmoid()
            ) for i, mod1 in enumerate(mod_in_dims.keys()) 
            for mod2 in list(mod_in_dims.keys())[i+1:]
        })
        
        # Add pathway-aware tokenizer if enabled
        self.use_pathway_tokenizer = getattr(cfg, 'use_pathway_tokenizer', False)
        if self.use_pathway_tokenizer:
            pathway_config_path = os.path.join(cfg.data_root, 'pathway_config.json') if hasattr(cfg, 'data_root') else None
            self.pathway_tokenizer = PathwayAwareTokenizer(
                latent_dim=cfg.latent_dim,
                pathway_config_path=pathway_config_path,
                num_misc_tokens=32
            )
            
            # Add projection layer to match diffusion model dimensions
            if self.pathway_tokenizer.latent_dim != cfg.latent_dim:
                self.pathway_projection = nn.Linear(self.pathway_tokenizer.latent_dim, cfg.latent_dim)
        
        # Configure decoders for each modality
        decoder_config = {
            'gene': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['gene'],
                'hidden_layers': [cfg.latent_dim * 2, cfg.latent_dim * 4],
                'dropout': 0.1
            },
            'protein': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['protein'],
                'hidden_layers': [cfg.latent_dim * 2, cfg.latent_dim * 2],
                'coupled_with': ['gene'],
                'coupling_dim': 256,
                'dropout': 0.1
            },
            'methylation': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['methylation'],
                'hidden_layers': [cfg.latent_dim * 2, cfg.latent_dim * 2],
                'dropout': 0.1
            },
            'variant': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['variant'],
                'hidden_layers': [cfg.latent_dim * 2, cfg.latent_dim],
                'dropout': 0.1
            },
            'metabolite': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['metabolite'],
                'hidden_layers': [cfg.latent_dim, cfg.latent_dim],
                'coupled_with': ['protein'],
                'coupling_dim': 256,
                'dropout': 0.1
            },
            'microbiome': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['microbiome'],
                'hidden_layers': [cfg.latent_dim * 2, cfg.latent_dim],
                'dropout': 0.1
            },
            'clinical': {
                'input_dim': cfg.latent_dim,
                'output_dim': mod_in_dims['clinical'],
                'hidden_layers': [cfg.latent_dim * 2, cfg.latent_dim],
                'continuous_features': mod_in_dims['clinical'] - 9,  # First 9 are categorical
                'categorical_features': 9,
                'dropout': 0.1
            }
        }
        
        # Initialize the new modality decoders with statistical heads
        stats_path = os.path.join(cfg.data_root, 'train', 'data_stats.pkl')
        self.modality_decoders = ModalityDecoders(decoder_config, stats_path=stats_path)
    
    def forward(self, batch, masks=None, kg_embed=None):
        """
        Enhanced forward pass with better biological modeling
        """
        # 1. Use pathway-aware tokenizer if enabled
        if self.use_pathway_tokenizer:
            # Prepare features dict for pathway tokenizer
            features_dict = {}
            for mod in ['gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
                if mod in batch:
                    features_dict[mod] = batch[mod]
            
            # Encode with pathway tokenizer
            pathway_tokens = self.pathway_tokenizer.encode(features_dict)  # [batch, num_tokens, token_dim]
            
            # Convert to flat representation for compatibility
            z_global = self.pathway_tokenizer.reshape_for_attention(pathway_tokens)  # [batch, latent_dim]
            
            # Project to match diffusion model dimensions if needed
            if hasattr(self, 'pathway_projection'):
                z_global = self.pathway_projection(z_global)
            
            # Store tokens for later decoding
            self._pathway_tokens = pathway_tokens
        else:
            # Original encoding path
            modality_embeddings = {}
            for mod, enc in self.encoders.items():
                if mod in batch:
                    # Enhanced projection after encoding
                    raw_embed = enc(batch[mod])
                    modality_embeddings[mod] = self.modality_projector[mod](raw_embed)
            
            # 2. Stack embeddings for hierarchical attention
            # Handle variable number of modalities gracefully
            embed_list = [modality_embeddings[mod] for mod in sorted(modality_embeddings.keys())]
            if len(embed_list) == 0:
                raise ValueError("No valid modalities found in batch")
            
            # Ensure all embeddings have same sequence length by global pooling if needed
            processed_embeds = []
            for embed in embed_list:
                if len(embed.shape) == 3:  # [batch, seq, dim]
                    processed_embeds.append(embed.mean(dim=1))  # Global average pooling
                else:  # [batch, dim]
                    processed_embeds.append(embed)
            
            z_stack = torch.stack(processed_embeds, dim=1)  # [batch, n_modalities, dim]
            
            # 3. Enhanced hierarchical attention
            z_enhanced = self.hier_attn(z_stack, masks)
            
            # 4. Global representation (average across modalities)
            z_global = z_enhanced.mean(dim=1)  # [batch, dim]
        
        # 5. Enhanced diffusion decoding
        if kg_embed is None:
            # Use learned knowledge embedding if not provided
            kg_embed = torch.zeros_like(z_global)
        
        if hasattr(self.diffusion, 'forward') and len(torch.nn.utils.parameters_to_vector(self.diffusion.parameters())) > 0:
            # KG-EDN or enhanced diffusion
            if isinstance(self.diffusion, KGEDN):
                z_decoded, diff_loss, energy_components = self.diffusion(z_global, kg_embed)
                return z_decoded, diff_loss, energy_components
            else:
                result = self.diffusion(z_global, kg_embed)
                if isinstance(result, tuple) and len(result) == 3:
                    z_decoded, diff_loss, _ = result
                else:
                    z_decoded, diff_loss = result
        else:
            # Fallback
            z_decoded = z_global
            diff_loss = torch.tensor(0.0, device=z_global.device)
        
        return z_decoded, diff_loss
    
    def compute_alignment_loss(self, modality_embeddings):
        """
        Compute cross-modal alignment loss for biological coherence
        """
        alignment_loss = 0.0
        count = 0
        
        mods = list(modality_embeddings.keys())
        for i, mod1 in enumerate(mods):
            for mod2 in mods[i+1:]:
                key = f"{mod1}_{mod2}"
                if key in self.alignment_heads:
                    # Concatenate embeddings
                    concat_embed = torch.cat([
                        modality_embeddings[mod1], 
                        modality_embeddings[mod2]
                    ], dim=-1)
                    
                    # Predict alignment score
                    alignment_score = self.alignment_heads[key](concat_embed)
                    
                    # We want high alignment (score close to 1)
                    target_alignment = torch.ones_like(alignment_score)
                    alignment_loss += nn.functional.mse_loss(alignment_score, target_alignment)
                    count += 1
        
        return alignment_loss / max(count, 1)
    
    def critic_losses(self, z_real, z_fake, graph_real=None, graph_fake=None):
        """
        Enhanced critic losses
        """
        # TODO: Implement when LocalCritic and GlobalCritic are available
        # local_loss = self.local_critic(z_real, z_fake)
        # 
        # if graph_real is not None and graph_fake is not None:
        #     global_loss = self.global_critic(graph_real, graph_fake)
        # else:
        #     # Use latent representations as proxy for graph-level features
        #     global_loss = self.global_critic(z_real.mean(dim=0, keepdim=True), 
        #                                    z_fake.mean(dim=0, keepdim=True))
        # 
        # return local_loss, global_loss
        
        # Return dummy losses for now
        device = z_real.device
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    def sample(self, num_samples: int, condition: torch.Tensor, device: torch.device):
        """
        Generate samples using the enhanced diffusion model
        """
        with torch.no_grad():
            if hasattr(self.diffusion, 'sample'):
                return self.diffusion.sample(num_samples, condition, device)
            else:
                # Fallback sampling
                return torch.randn(num_samples, self.cfg.latent_dim, device=device)
    
    def evaluate_biological_coherence(self, generated_data, real_data):
        """
        Enhanced biological coherence evaluation
        """
        coherence_scores = {}
        
        # 1. Cross-modal alignment coherence
        if hasattr(self, 'alignment_heads'):
            alignment_scores = []
            for key, head in self.alignment_heads.items():
                # Assuming generated_data and real_data are dictionaries of modalities
                mod1, mod2 = key.split('_')
                if mod1 in generated_data and mod2 in generated_data:
                    gen_concat = torch.cat([generated_data[mod1], generated_data[mod2]], dim=-1)
                    gen_alignment = head(gen_concat).mean().item()
                    alignment_scores.append(gen_alignment)
            
            if alignment_scores:
                coherence_scores['alignment_coherence'] = sum(alignment_scores) / len(alignment_scores)
        
        # 2. Energy-based coherence (if using KG-EDN)
        if isinstance(self.diffusion, KGEDN):
            # Stack all generated modality data
            gen_stack = torch.stack([generated_data[mod] for mod in generated_data.keys()], dim=1)
            gen_global = gen_stack.mean(dim=1)
            feasibility = self.diffusion.evaluate_biological_feasibility(gen_global)
            coherence_scores.update(feasibility)
        
        # 3. Distribution similarity (simple version)
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            common_mods = set(real_data.keys()) & set(generated_data.keys())
            dist_similarities = []
            
            for mod in common_mods:
                # Simple distribution similarity using mean and std
                real_mean, real_std = real_data[mod].mean(), real_data[mod].std()
                gen_mean, gen_std = generated_data[mod].mean(), generated_data[mod].std()
                
                mean_sim = 1.0 - torch.abs(real_mean - gen_mean) / (torch.abs(real_mean) + 1e-8)
                std_sim = 1.0 - torch.abs(real_std - gen_std) / (torch.abs(real_std) + 1e-8)
                
                dist_similarities.append((mean_sim + std_sim) / 2)
            
            if dist_similarities:
                coherence_scores['distribution_similarity'] = sum(dist_similarities) / len(dist_similarities)
        
        return coherence_scores
    
    def critic_losses(self, z_real, z_fake, local_real=None, global_real=None):
        """
        Compute local and global critic losses for adversarial training with gradient penalty.
        
        Args:
            z_real: Real latent representations
            z_fake: Generated latent representations
            local_real: Optional local real features
            global_real: Optional global real features
            
        Returns:
            local_critic_loss: Local critic loss
            global_critic_loss: Global critic loss
        """
        # Initialize critic networks if not present
        if not hasattr(self, 'local_critic'):
            self.local_critic = nn.Sequential(
                nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim),
                nn.LayerNorm(self.cfg.latent_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.cfg.latent_dim, 1)
            ).to(z_real.device)
            
        if not hasattr(self, 'global_critic'):
            self.global_critic = nn.Sequential(
                nn.Linear(self.cfg.latent_dim * 2, self.cfg.latent_dim),
                nn.LayerNorm(self.cfg.latent_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.cfg.latent_dim, 1)
            ).to(z_real.device)
        
        # Local critic loss (WGAN-GP style)
        local_real_score = self.local_critic(z_real.detach())
        local_fake_score = self.local_critic(z_fake.detach())  # Detach fake for critic training
        
        # Wasserstein distance
        local_critic_loss = -torch.mean(local_real_score) + torch.mean(local_fake_score)
        
        # Gradient penalty for local critic
        local_gp = self.compute_gradient_penalty(self.local_critic, z_real, z_fake)
        local_critic_loss = local_critic_loss + 10.0 * local_gp  # Lambda=10 is typical for WGAN-GP
        
        # Global critic loss
        # Concatenate real and fake for global context
        global_real_input = torch.cat([z_real.detach(), z_real.detach()], dim=1)
        global_fake_input = torch.cat([z_fake.detach(), z_fake.detach()], dim=1)
        
        global_real_score = self.global_critic(global_real_input)
        global_fake_score = self.global_critic(global_fake_input)
        
        global_critic_loss = -torch.mean(global_real_score) + torch.mean(global_fake_score)
        
        # Gradient penalty for global critic
        global_gp = self.compute_gradient_penalty(self.global_critic, global_real_input, global_fake_input)
        global_critic_loss = global_critic_loss + 10.0 * global_gp
        
        # Ensure losses are valid
        if torch.isnan(local_critic_loss):
            local_critic_loss = torch.zeros(1, device=z_real.device, requires_grad=True)
        if torch.isnan(global_critic_loss):
            global_critic_loss = torch.zeros(1, device=z_real.device, requires_grad=True)
            
        return local_critic_loss, global_critic_loss
    
    def compute_gradient_penalty(self, critic, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, device=device)
        if real_data.dim() > 2:
            for _ in range(real_data.dim() - 2):
                alpha = alpha.unsqueeze(-1)
        
        # Interpolate between real and fake
        interpolated = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
        interpolated.requires_grad_(True)
        
        # Compute critic output
        critic_output = critic(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
        
        return gradient_penalty
    
    def generator_losses(self, z_fake):
        """
        Compute generator losses for WGAN-GP training.
        
        Args:
            z_fake: Generated latent representations
            
        Returns:
            local_gen_loss: Local generator loss
            global_gen_loss: Global generator loss
        """
        # Generator wants to maximize critic scores (minimize negative scores)
        local_fake_score = self.local_critic(z_fake)
        local_gen_loss = -torch.mean(local_fake_score)
        
        # Global generator loss
        global_fake_input = torch.cat([z_fake, z_fake], dim=1)
        global_fake_score = self.global_critic(global_fake_input)
        global_gen_loss = -torch.mean(global_fake_score)
        
        return local_gen_loss, global_gen_loss
    
    def get_hierarchy_attention_maps(self, batch, masks=None):
        """
        Get attention maps for each hierarchy level for analysis
        """
        # Encode modalities
        modality_embeddings = {}
        for mod, enc in self.encoders.items():
            if mod in batch:
                raw_embed = enc(batch[mod])
                modality_embeddings[mod] = self.modality_projector[mod](raw_embed)
        
        embed_list = [modality_embeddings[mod] for mod in sorted(modality_embeddings.keys())]
        processed_embeds = []
        for embed in embed_list:
            if len(embed.shape) == 3:
                processed_embeds.append(embed.mean(dim=1))
            else:
                processed_embeds.append(embed)
        
        z_stack = torch.stack(processed_embeds, dim=1)
        
        # Get attention weights for each level
        attention_maps = {}
        for level in self.hier_attn.hierarchy_levels:
            mask = masks.get(level, None) if masks else None
            attention_weights = self.hier_attn.get_attention_weights(z_stack, level, mask)
            attention_maps[level] = attention_weights
        
        return attention_maps
    
    def decode_modality(self, z, modality):
        """Decode latent representation back to specific modality"""
        # Use pathway tokenizer if enabled
        if self.use_pathway_tokenizer and hasattr(self, '_pathway_tokens'):
            # If we projected the dimensions, we need to unproject first
            if hasattr(self, 'pathway_projection'):
                # Create inverse projection if not exists
                if not hasattr(self, 'pathway_unprojection'):
                    self.pathway_unprojection = nn.Linear(self.cfg.latent_dim, self.pathway_tokenizer.latent_dim).to(z.device)
                z = self.pathway_unprojection(z)
            
            # Convert flat latent back to tokens
            tokens = self.pathway_tokenizer.reshape_from_attention(z)
            # Decode specific modality from tokens
            return self.pathway_tokenizer.decode(tokens, modality)
        else:
            # Create modality embeddings dict with single latent for all modalities
            modality_embeddings = {mod: z for mod in self.modality_decoders.decoders.keys()}
            
            # Decode all modalities (we'll pick the one we need)
            decoded_outputs = self.modality_decoders(modality_embeddings, enforce_coupling=True)
            
            # Return the requested modality
            if modality in decoded_outputs:
                # Handle clinical output which is a dict
                if modality == 'clinical' and isinstance(decoded_outputs[modality], dict):
                    # Concatenate continuous and categorical
                    continuous = decoded_outputs[modality]['continuous']
                    categorical = decoded_outputs[modality]['categorical']
                    return torch.cat([categorical, continuous], dim=-1)
                else:
                    return decoded_outputs[modality]
            else:
                raise ValueError(f"No decoder output for modality: {modality}") 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import pickle
from torch.utils.checkpoint import checkpoint

# Correctly import the new module we just created
from .statistical_head import StatisticalHead

#==============================================================================
#  NEW, MORE POWERFUL BASE BLOCKS
#==============================================================================

class ResBlock(nn.Module):
    """A residual block with LayerNorm for stable, deep networks."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class BaseDecoder(nn.Module):
    """An upgraded BaseDecoder using ResBlocks. It no longer contains the final projection layer."""
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim

        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU(inplace=True)]
        for i in range(len(hidden_layers)):
            layers.append(ResBlock(hidden_layers[i], dropout=dropout))
            if i < len(hidden_layers) - 1:
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

#==============================================================================
#  STATISTICALLY-AWARE DECODERS (THE CORE FIX)
#==============================================================================

class GeneDecoder(BaseDecoder):
    # --- THE CHANGE: Add stats_path to __init__ ---
    def __init__(self, config: Dict, stats_path: str):
        super().__init__(
            input_dim=config.get('input_dim', 1024),
            hidden_layers=config['hidden_layers'],
            dropout=config.get('dropout', 0.1)
        )
        final_hidden_dim = config['hidden_layers'][-1]
        # Gene expression data is already log-transformed, so use normal distribution
        self.statistical_head = StatisticalHead(
            input_dim=final_hidden_dim, output_dim=config['output_dim'], dist_type='normal'
        )
        
        # --- Use the provided path ---
        try:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"data_stats.pkl not found at the specified path: {stats_path}. Please run the `calculate_stats.py` script first.")
        
        if 'gene' in stats: gene_stats_key = 'gene'
        elif 'gene' in stats: gene_stats_key = 'gene'
        else: raise KeyError(f"Could not find 'gene' or 'gene' statistics in {stats_path}.")

        self.register_buffer('target_stats_mean', torch.tensor(stats[gene_stats_key]['mean'], dtype=torch.float32))
        self.register_buffer('target_stats_std', torch.tensor(stats[gene_stats_key]['std'], dtype=torch.float32))

    def forward(self, x: torch.Tensor, diversity_scale: float = None) -> torch.Tensor:
        # Add input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: NaN/Inf in GeneDecoder input, replacing with zeros")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        features = self.decoder(x)
        
        # Check intermediate features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"WARNING: NaN/Inf in GeneDecoder features, replacing with zeros")
            features = torch.where(torch.isnan(features) | torch.isinf(features), torch.zeros_like(features), features)
        
        target_stats = {'mean': self.target_stats_mean, 'std': self.target_stats_std}
        output = self.statistical_head(features, target_stats, diversity_scale=diversity_scale)
        
        # Final safety clamp for log-space gene expression
        output = torch.clamp(output, -10.0, 20.0)
        
        # Last check
        if torch.isnan(output).any():
            print(f"WARNING: NaN in GeneDecoder output despite clamping, replacing with mean")
            output = torch.where(torch.isnan(output), self.target_stats_mean.expand_as(output), output)
        
        return output

class MethylationDecoder(BaseDecoder):
    # --- THE CHANGE: Add stats_path to __init__ ---
    def __init__(self, config: Dict, stats_path: str):
        super().__init__(
            input_dim=config.get('input_dim', 1024),
            hidden_layers=config['hidden_layers'],
            dropout=config.get('dropout', 0.1)
        )
        final_hidden_dim = config['hidden_layers'][-1]
        self.statistical_head = StatisticalHead(
            input_dim=final_hidden_dim, output_dim=config['output_dim'], dist_type='beta'
        )
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        if 'methylation' not in stats: raise KeyError(f"'methylation' statistics not found in {stats_path}.")
        self.register_buffer('target_stats_mean', torch.tensor(stats['methylation']['mean'], dtype=torch.float32))
        self.register_buffer('target_stats_std', torch.tensor(stats['methylation']['std'], dtype=torch.float32))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: NaN/Inf in MethylationDecoder input, replacing with zeros")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        features = self.decoder(x)
        
        # Check intermediate features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"WARNING: NaN/Inf in MethylationDecoder features, replacing with zeros")
            features = torch.where(torch.isnan(features) | torch.isinf(features), torch.zeros_like(features), features)
        
        target_stats = {'mean': self.target_stats_mean, 'std': self.target_stats_std}
        output = self.statistical_head(features, target_stats)
        
        # Last check - methylation should be in [0,1]
        if torch.isnan(output).any():
            print(f"WARNING: NaN in MethylationDecoder output, replacing with mean")
            output = torch.where(torch.isnan(output), self.target_stats_mean.expand_as(output), output)
        
        return output

#==============================================================================
#  ORIGINAL DECODERS & COUPLERS (FOR OTHER MODALITIES)
#==============================================================================

class NonStatBaseDecoder(nn.Module):
    """The original BaseDecoder for modalities where we don't apply the statistical head."""
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], activation: str = "relu", dropout: float = 0.1, use_batch_norm: bool = True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm: layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == "relu": layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu": layers.append(nn.GELU())
            elif activation == "leaky_relu": layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class ProteinDecoder(NonStatBaseDecoder):
    def __init__(self, config: Dict):
        input_dim = config.get('input_dim', 1024); coupling_dim = 0
        if config.get('coupled_with') and 'gene' in config.get('coupled_with', []): coupling_dim = config.get('coupling_dim', 256)
        super().__init__(input_dim=input_dim + coupling_dim, output_dim=config['output_dim'], hidden_layers=config['hidden_layers'], activation=config.get('activation', 'relu'), dropout=config.get('dropout', 0.1))
        
        # Add direct gene-to-protein projection for stronger coupling
        self.gene_projection = nn.Sequential(
            nn.Linear(5000, 512),  # Assuming 5000 gene dims
            nn.ReLU(),
            nn.Linear(512, config['output_dim']),
            nn.Sigmoid()  # Protein levels are positive
        )
        
    def forward(self, x: torch.Tensor, gene_expr: torch.Tensor = None) -> torch.Tensor:
        base_output = F.softplus(super().forward(x))
        
        # If gene expression provided, mix it in with realistic coupling
        if gene_expr is not None:
            gene_proj = self.gene_projection(gene_expr)
            
            # Further reduced coupling: 25% from gene projection, 75% from independent factors
            # This reflects post-transcriptional regulation, protein stability, etc.
            output = 0.25 * gene_proj + 0.75 * base_output
            
            # Add biological variability (always, not just training)
            # Post-transcriptional noise (translation efficiency, protein degradation)
            noise = torch.randn_like(output) * 0.2  # Increased noise
            output = output + noise
            
            # Ensure positive values with small epsilon
            return F.relu(output) + 1e-6
        
        return base_output

class VariantDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__(); self.input_dim = config.get('input_dim', 1024); self.output_dim = config['output_dim']; self.variant_generator = nn.Sequential(nn.Linear(self.input_dim, config['hidden_layers'][0]), nn.ReLU(inplace=True), nn.Dropout(config.get('dropout', 0.1)), nn.Linear(config['hidden_layers'][0], config['hidden_layers'][1]), nn.ReLU(inplace=True), nn.Dropout(config.get('dropout', 0.1))); self.position_decoder = nn.Linear(config['hidden_layers'][1], self.output_dim); self.allele_decoder = nn.Linear(config['hidden_layers'][1], 4); self.sparsity_controller = nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.variant_generator(x)
        positions = torch.sigmoid(self.position_decoder(features))
        allele_logits = self.allele_decoder(features)
        alleles = F.gumbel_softmax(allele_logits, tau=0.5, hard=True)
        
        # Ensure sparsity factor is non-negative for Poisson distribution
        sparsity_raw = self.sparsity_controller(x)  # Should be in [0,1] from sigmoid
        
        # Scale to reasonable range for variant counts
        # Remove the 0.01 multiplication that was making lambda too small
        poisson_lambda = sparsity_raw * self.output_dim
        
        # Clamp to reasonable range: 1-20 variants per sample
        poisson_lambda = torch.clamp(poisson_lambda, min=1.0, max=20.0)
        
        # Additional safety: ensure no NaN/Inf values
        if torch.isnan(poisson_lambda).any() or torch.isinf(poisson_lambda).any():
            print(f"WARNING: NaN/Inf in poisson_lambda, replacing with safe value")
            poisson_lambda = torch.ones_like(poisson_lambda) * 1.0
        
        # Detach from computation graph to avoid gradient issues
        with torch.no_grad():
            num_variants = torch.poisson(poisson_lambda.detach()).int()
            # Ensure at least 1 variant
            num_variants = torch.clamp(num_variants, min=1)
        
        return {'positions': positions, 'alleles': alleles, 'num_variants': num_variants}

class MetaboliteDecoder(NonStatBaseDecoder):
    def __init__(self, config: Dict):
        input_dim = config.get('input_dim', 1024); coupling_dim = 0
        if config.get('coupled_with') and 'protein' in config.get('coupled_with', []): coupling_dim = config.get('coupling_dim', 256)
        super().__init__(input_dim=input_dim + coupling_dim, output_dim=config['output_dim'], hidden_layers=config['hidden_layers'], activation=config.get('activation', 'relu'), dropout=config.get('dropout', 0.1))
        
        # Add metabolic pathway constraints
        self.pathway_modulator = nn.Sequential(
            nn.Linear(config['output_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, config['output_dim']),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base metabolite levels
        base_output = super().forward(x)
        
        # Apply softplus instead of exp to avoid explosion
        metabolites = F.softplus(base_output)
        
        # Apply pathway constraints
        pathway_factors = self.pathway_modulator(metabolites)
        metabolites = metabolites * pathway_factors
        
        # Add metabolic noise for realism (always, not just training)
        noise = torch.randn_like(metabolites) * 0.15
        metabolites = metabolites + noise
        
        # Don't force positive - let the model learn realistic distributions
        # Some metabolites can have negative log-fold changes
            
        return metabolites

class MicrobiomeDecoder(NonStatBaseDecoder):
    def __init__(self, config: Dict):
        super().__init__(input_dim=config.get('input_dim', 1024), output_dim=config['output_dim'], hidden_layers=config['hidden_layers'], activation=config.get('activation', 'relu'), dropout=config.get('dropout', 0.1)); self.abundance_normalizer = nn.Softmax(dim=-1); self.diversity_controller = nn.Sequential(nn.Linear(config.get('input_dim', 1024), 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_output = super().forward(x); diversity_factor = self.diversity_controller(x); temperature = 0.1 + diversity_factor * 0.9
        return self.abundance_normalizer(raw_output / temperature)

class ClinicalDecoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.base_decoder = NonStatBaseDecoder(
            input_dim=config.get('input_dim', 1024),
            output_dim=config['hidden_layers'][-1],
            hidden_layers=config['hidden_layers'],
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.1)
        )
        self.continuous_features = config.get('continuous_features', 50)
        self.categorical_features = config.get('categorical_features', 50)
        self.continuous_decoder = nn.Linear(config['hidden_layers'][-1], self.continuous_features)
        self.categorical_decoder = nn.Linear(config['hidden_layers'][-1], self.categorical_features)
        
        # Add clinical relationship modulator
        self.clinical_relationships = nn.Sequential(
            nn.Linear(self.continuous_features, 32),
            nn.ReLU(),
            nn.Linear(32, self.continuous_features),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.base_decoder(x)
        
        # Generate base clinical values
        continuous_raw = self.continuous_decoder(features)
        
        # Apply different activations for different value ranges
        # First 10 features: vital signs (bounded)
        vitals = torch.tanh(continuous_raw[:, :10])
        # Next 20 features: lab values (semi-bounded)
        labs = F.softplus(continuous_raw[:, 10:30])
        # Remaining: other measurements
        others = continuous_raw[:, 30:]
        
        continuous = torch.cat([vitals, labs, others], dim=1)
        
        # Apply clinical relationships to ensure consistency
        relationship_factors = self.clinical_relationships(continuous)
        continuous = continuous * (0.5 + 0.5 * relationship_factors)
        
        # Add clinical variability (always, not just training)
        noise = torch.randn_like(continuous) * 0.1  # Increased noise
        continuous = continuous + noise
        
        # Categorical features
        categorical_logits = self.categorical_decoder(features)
        categorical = F.gumbel_softmax(categorical_logits, tau=0.5, hard=True, dim=-1)
        
        return {'continuous': continuous, 'categorical': categorical}

class CentralDogmaCoupler(nn.Module):
    def __init__(self, coupling_dim: int = 256, gene_dim: int = 5000):
        super().__init__()
        self.coupling_dim = coupling_dim
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, coupling_dim)
        )
        self.gene_projection = nn.Linear(coupling_dim, 1024)
        self.time_delay = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        
        # Make coupling strength learnable and add biological noise
        self.coupling_strength = nn.Parameter(torch.tensor(0.4), requires_grad=True)  # Start at 40%
        self.regulation_noise = nn.Sequential(
            nn.Linear(coupling_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, protein_embedding: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        gene_features = self.gene_encoder(gene)
        gene_features_projected = self.gene_projection(gene_features)
        
        # Learnable coupling strength clamped to lower range
        coupling_strength = torch.sigmoid(self.coupling_strength) * 0.3 + 0.1  # 10-40% range
        
        # Add post-transcriptional regulation noise
        regulation_factor = self.regulation_noise(gene_features)
        regulation_factor = 0.7 + 0.3 * regulation_factor  # 70-100% expression efficiency
        
        # Mix embeddings with regulation
        mixed_embedding = protein_embedding * (1 - coupling_strength) + gene_features_projected * coupling_strength
        mixed_embedding = mixed_embedding * regulation_factor
        
        # Time modulation for translation delay
        time_modulation = torch.sigmoid(self.time_delay) * 0.8 + 0.2
        
        # Add biological noise during training AND evaluation to prevent gaming
        # Always add noise, not just during training
        noise = torch.randn_like(mixed_embedding) * 0.15
        mixed_embedding = mixed_embedding + noise
            
        return mixed_embedding * time_modulation

class MetabolicCoupler(nn.Module):
    def __init__(self, coupling_dim: int = 256, protein_dim: int = 226):
        super().__init__(); self.coupling_dim = coupling_dim; self.protein_encoder = nn.Sequential(nn.Linear(protein_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(1024, coupling_dim)); self.metabolite_proj = nn.Linear(1024, coupling_dim); self.enzyme_attention = nn.MultiheadAttention(embed_dim=coupling_dim, num_heads=8, dropout=0.1, batch_first=True); self.fusion_layer = nn.Sequential(nn.Linear(1024 + coupling_dim, 1024), nn.LayerNorm(1024), nn.ReLU(inplace=True), nn.Dropout(0.1))
    def forward(self, metabolite_embedding: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        protein_features = self.protein_encoder(protein); metabolite_features = self.metabolite_proj(metabolite_embedding); protein_features_expanded = protein_features.unsqueeze(1); metabolite_features_expanded = metabolite_features.unsqueeze(1); attended_features, _ = self.enzyme_attention(query=metabolite_features_expanded, key=protein_features_expanded, value=protein_features_expanded); attended_features = attended_features.squeeze(1); coupled_features = torch.cat([metabolite_embedding, attended_features], dim=-1)
        return self.fusion_layer(coupled_features)

#==============================================================================
#  TOP-LEVEL DECODER MODULE
#==============================================================================

class ModalityDecoders(nn.Module):
    # --- THE CHANGE: Add stats_path to __init__ ---
    def __init__(self, config: Dict, stats_path: str):
        super().__init__()
        self.config = config
        self.use_checkpointing = False
        
        # --- Pass the path down to the relevant decoders ---
        self.decoders = nn.ModuleDict({
            'gene': GeneDecoder(config['gene'], stats_path=stats_path),
            'protein': ProteinDecoder(config['protein']),
            'methylation': MethylationDecoder(config['methylation'], stats_path=stats_path),
            'variant': VariantDecoder(config['variant']),
            'metabolite': MetaboliteDecoder(config['metabolite']),
            'microbiome': MicrobiomeDecoder(config['microbiome']),
            'clinical': ClinicalDecoder(config['clinical'])
        })
        
        # Get gene and protein dimensions from config
        gene_dim = config.get('gene', {}).get('output_dim', 5000)
        protein_dim = config.get('protein', {}).get('output_dim', 226)
        self.central_dogma_coupler = CentralDogmaCoupler(gene_dim=gene_dim)
        self.metabolic_coupler = MetabolicCoupler(protein_dim=protein_dim)
    
    def forward(self, modality_embeddings: Dict[str, torch.Tensor], enforce_coupling: bool = True) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        def run_decoder(decoder, *args):
            return decoder(*args)

        if self.use_checkpointing and self.training:
            gene_output = checkpoint(run_decoder, self.decoders['gene'], modality_embeddings['gene'], use_reentrant=False)
        else:
            gene_output = self.decoders['gene'](modality_embeddings['gene'])
        outputs['gene'] = gene_output
        
        protein_input = self.central_dogma_coupler(modality_embeddings['protein'], gene_output) if enforce_coupling else modality_embeddings['protein']
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        if enforce_coupling:
            logger.debug(f"Central dogma coupling ENABLED - protein input modified")
        else:
            logger.debug(f"Central dogma coupling DISABLED - using raw embeddings")
        if self.use_checkpointing and self.training:
            outputs['protein'] = checkpoint(run_decoder, self.decoders['protein'], protein_input, gene_output, use_reentrant=False)
        else:
            outputs['protein'] = self.decoders['protein'](protein_input, gene_output)

        if self.use_checkpointing and self.training:
            outputs['methylation'] = checkpoint(run_decoder, self.decoders['methylation'], modality_embeddings['methylation'], use_reentrant=False)
        else:
            outputs['methylation'] = self.decoders['methylation'](modality_embeddings['methylation'])

        if self.use_checkpointing and self.training:
            outputs['variant'] = checkpoint(run_decoder, self.decoders['variant'], modality_embeddings['variant'], use_reentrant=False)
        else:
            outputs['variant'] = self.decoders['variant'](modality_embeddings['variant'])
        
        metabolite_input = self.metabolic_coupler(modality_embeddings['metabolite'], outputs['protein']) if enforce_coupling else modality_embeddings['metabolite']
        if self.use_checkpointing and self.training:
            outputs['metabolite'] = checkpoint(run_decoder, self.decoders['metabolite'], metabolite_input, use_reentrant=False)
        else:
            outputs['metabolite'] = self.decoders['metabolite'](metabolite_input)
        
        if self.use_checkpointing and self.training:
            outputs['microbiome'] = checkpoint(run_decoder, self.decoders['microbiome'], modality_embeddings['microbiome'], use_reentrant=False)
        else:
            outputs['microbiome'] = self.decoders['microbiome'](modality_embeddings['microbiome'])

        if self.use_checkpointing and self.training:
            outputs['clinical'] = checkpoint(run_decoder, self.decoders['clinical'], modality_embeddings['clinical'], use_reentrant=False)
        else:
            outputs['clinical'] = self.decoders['clinical'](modality_embeddings['clinical'])
        
        return outputs
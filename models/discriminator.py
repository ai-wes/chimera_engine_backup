import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class ModalityCritic(nn.Module):
    """Single modality critic for adversarial training"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        
        # Feature extraction
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i == 0 else hidden_dim // (2 ** (i - 1))
            out_dim = max(out_dim, 64)  # Minimum 64 dims
            
            layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            current_dim = out_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.final_dim = current_dim
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.final_dim // 2, 1)
        )
        
        # Statistical feature extraction
        self.stat_extractor = nn.Sequential(
            nn.Linear(5, hidden_dim // 4),  # mean, std, min, max, median
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            critic_score: [batch, 1]
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Extract statistical features
        stats = torch.stack([
            x.mean(dim=-1),
            x.std(dim=-1),
            x.min(dim=-1)[0],
            x.max(dim=-1)[0],
            x.median(dim=-1)[0]
        ], dim=-1)
        
        stat_features = self.stat_extractor(stats)
        
        # Combine features
        combined = torch.cat([features, stat_features], dim=-1)
        
        # Critic score
        score = self.critic_head(combined)
        
        return score


class CrossModalCritic(nn.Module):
    """Critic for cross-modal relationships"""
    
    def __init__(self, modality_dims, hidden_dim, dropout=0.1):
        super().__init__()
        self.modality_dims = modality_dims
        
        # Modality projections
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2)
            )
            for name, dim in modality_dims.items()
        })
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Relationship modeling
        self.relationship_net = nn.Sequential(
            nn.Linear(hidden_dim * len(modality_dims), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, modality_data):
        """
        Args:
            modality_data: List of tensors [batch, dim] for each modality
        Returns:
            critic_score: [batch, 1]
        """
        batch_size = modality_data[0].shape[0]
        
        # Project each modality
        projected = []
        for i, (name, data) in enumerate(zip(self.modality_dims.keys(), modality_data)):
            proj = self.projections[name](data)
            projected.append(proj)
        
        # Stack for attention
        proj_stack = torch.stack(projected, dim=1)  # [batch, num_modalities, hidden_dim]
        
        # Cross-modal attention
        attended, _ = self.cross_attn(proj_stack, proj_stack, proj_stack)
        
        # Flatten for relationship modeling
        flattened = attended.view(batch_size, -1)
        
        # Compute relationship score
        score = self.relationship_net(flattened)
        
        return score


class GlobalCoherenceCritic(nn.Module):
    """Critic for global biological coherence"""
    
    def __init__(self, modality_dims, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Modality encoders with different architectures
        self.encoders = nn.ModuleDict()
        
        # Gene expression encoder (largest)
        self.encoders['gene_expression'] = nn.Sequential(
            nn.Linear(modality_dims['gene_expression'], 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim)
        )
        
        # DNA methylation encoder
        self.encoders['dnam'] = nn.Sequential(
            nn.Linear(modality_dims['dnam'], 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim)
        )
        
        # Other modality encoders
        for name, dim in modality_dims.items():
            if name not in ['gene_expression', 'dnam']:
                self.encoders[name] = nn.Sequential(
                    nn.Linear(dim, min(dim, 256)),
                    nn.LayerNorm(min(dim, 256)),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(min(dim, 256), hidden_dim)
                )
        
        # Global coherence network
        self.coherence_net = nn.Sequential(
            nn.Linear(hidden_dim * len(modality_dims), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Pairwise coherence scores
        self.pairwise_coherence = nn.ModuleList([
            nn.Bilinear(hidden_dim, hidden_dim, 1)
            for _ in range(len(modality_dims) * (len(modality_dims) - 1) // 2)
        ])
        
    def forward(self, modality_data):
        """
        Args:
            modality_data: List of tensors [batch, dim] for each modality
        Returns:
            critic_score: [batch, 1]
        """
        batch_size = modality_data[0].shape[0]
        
        # Encode each modality
        encoded = []
        for i, (name, data) in enumerate(zip(self.encoders.keys(), modality_data)):
            enc = self.encoders[name](data)
            encoded.append(enc)
        
        # Compute pairwise coherence
        pairwise_scores = []
        pair_idx = 0
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                score = self.pairwise_coherence[pair_idx](encoded[i], encoded[j])
                pairwise_scores.append(score)
                pair_idx += 1
        
        # Concatenate all encodings
        all_encoded = torch.cat(encoded, dim=-1)
        
        # Global coherence score
        global_score = self.coherence_net(all_encoded)
        
        # Combine with pairwise scores
        if pairwise_scores:
            pairwise_avg = torch.stack(pairwise_scores, dim=-1).mean(dim=-1, keepdim=True)
            final_score = 0.7 * global_score + 0.3 * pairwise_avg
        else:
            final_score = global_score
        
        return final_score


class MultiModalDiscriminator(nn.Module):
    """Multi-agent discriminator with specialized critics"""
    
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
        
        # Individual modality critics
        self.modality_critics = nn.ModuleDict({
            name: ModalityCritic(dim, hidden_dim, num_layers=3, dropout=cfg.mask_dropout)
            for name, dim in self.modality_dims.items()
        })
        
        # Cross-modal critic
        self.cross_modal_critic = CrossModalCritic(
            self.modality_dims, hidden_dim, dropout=cfg.mask_dropout
        )
        
        # Global coherence critic
        self.global_critic = GlobalCoherenceCritic(
            self.modality_dims, hidden_dim, dropout=cfg.mask_dropout
        )
        
        # Critic aggregation
        num_critics = len(self.modality_dims) + 2  # modalities + cross + global
        self.critic_aggregator = nn.Sequential(
            nn.Linear(num_critics, num_critics * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(cfg.mask_dropout),
            nn.Linear(num_critics * 2, num_critics),
            nn.LeakyReLU(0.2),
            nn.Linear(num_critics, 1)
        )
        
        # Learnable critic weights
        self.critic_weights = nn.Parameter(torch.ones(num_critics) / num_critics)
        
        logging.info(f"Initialized multi-modal discriminator with {num_critics} critics")
        
    def forward(self, modality_data, return_all_scores=False):
        """
        Args:
            modality_data: List of tensors [batch, dim] for each modality
            return_all_scores: If True, return individual critic scores
        Returns:
            final_score: [batch, 1] aggregated discriminator score
            all_scores: Dict of individual scores (if return_all_scores=True)
        """
        batch_size = modality_data[0].shape[0]
        all_scores = {}
        score_list = []
        
        # Individual modality critics
        for i, (name, data) in enumerate(zip(self.modality_dims.keys(), modality_data)):
            score = self.modality_critics[name](data)
            all_scores[f'{name}_critic'] = score
            score_list.append(score)
        
        # Cross-modal critic
        cross_score = self.cross_modal_critic(modality_data)
        all_scores['cross_modal_critic'] = cross_score
        score_list.append(cross_score)
        
        # Global coherence critic
        global_score = self.global_critic(modality_data)
        all_scores['global_critic'] = global_score
        score_list.append(global_score)
        
        # Stack all scores
        scores_tensor = torch.cat(score_list, dim=-1)  # [batch, num_critics]
        
        # Apply learned weights
        weights = F.softmax(self.critic_weights, dim=0)
        weighted_scores = scores_tensor * weights.unsqueeze(0)
        
        # Aggregate
        if self.cfg.get('critic_coefs', None):
            # Use config-specified coefficients if available
            coef_local = self.cfg.critic_coefs.get('local', 0.7)
            coef_global = self.cfg.critic_coefs.get('global', 0.3)
            
            local_scores = weighted_scores[:, :-2].mean(dim=-1, keepdim=True)
            global_scores = weighted_scores[:, -2:].mean(dim=-1, keepdim=True)
            
            final_score = coef_local * local_scores + coef_global * global_scores
        else:
            # Use learned aggregation
            final_score = self.critic_aggregator(weighted_scores)
        
        all_scores['final_score'] = final_score
        
        if return_all_scores:
            return final_score, all_scores
        
        return final_score

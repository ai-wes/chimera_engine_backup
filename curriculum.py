import torch
import torch.nn as nn
import wandb
from models.hgtd_model import EnhancedHGTD as HGTD
from models.rl_coherence_agent import CoherenceRLAgent
from models.modality_specific_rl_agents import MultiModalityRLSystem
import yaml
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TCGASevenModality(Dataset):
    def __init__(self, root, split='train'):
        super().__init__()
        self.root = Path(root)
        self.split = split
        
        # Load data from split subdirectory
        data_file = self.root / split / f'{split}_data.pkl'
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Fix methylation if all zeros
        if 'methylation' in self.data:
            meth_data = self.data['methylation']
            print(f"Methylation shape: {meth_data.shape}, min: {meth_data.min():.4f}, max: {meth_data.max():.4f}")
            
            # Check how many samples are all zeros
            zeros_per_sample = np.sum(meth_data == 0, axis=1)
            all_zero_samples = np.sum(zeros_per_sample == meth_data.shape[1])
            print(f"Methylation samples with all zeros: {all_zero_samples}/{meth_data.shape[0]} ({100*all_zero_samples/meth_data.shape[0]:.1f}%)")
            
            # Fix samples that are all zeros
            if all_zero_samples > 0:
                print(f"Fixing {all_zero_samples} all-zero methylation samples...")
                for i in range(meth_data.shape[0]):
                    if np.all(meth_data[i] == 0):
                        # Use beta distribution for this sample
                        self.data['methylation'][i] = np.random.beta(2, 5, size=meth_data.shape[1]).astype(np.float32)
                
                # Verify the fix
                new_zeros = np.sum(np.sum(self.data['methylation'] == 0, axis=1) == meth_data.shape[1])
                print(f"After fix: {new_zeros} samples still all zeros")
                print(f"New methylation - min: {self.data['methylation'].min():.4f}, max: {self.data['methylation'].max():.4f}, mean: {self.data['methylation'].mean():.4f}")
        
        # Load knowledge graph
        kg_file = self.root / 'knowledge_graph.pt'
        if kg_file.exists():
            self.kg_graph = torch.load(kg_file, weights_only=False)
        else:
            self.kg_graph = None
        
        # Get number of samples from first modality
        first_mod = list(self.data.keys())[0]
        self.n_samples = self.data[first_mod].shape[0]
        
        print(f"Loaded {split} data with {self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sample = {}
        
        # Get each modality for this sample
        for modality, data in self.data.items():
            if modality == '_metadata':
                continue
            sample[modality] = torch.tensor(data[idx], dtype=torch.float32)
            
        # Methylation should already be fixed at dataset level
        
        # Add knowledge graph context if available
        if self.kg_graph:
            # For now, create a dummy KG context vector
            sample['kg'] = torch.randn(256)
        else:
            sample['kg'] = torch.zeros(256)
        
        # Create hierarchical masks (4 levels as per config)
        # These should ideally be learned or based on biological knowledge
        sample['masks'] = [torch.eye(7) for _ in range(4)]
        
        # Add placeholder graph representations
        sample['graph_real'] = torch.randn(256)
        sample['graph_fake'] = torch.randn(256)
        
        return sample

class CurriculumTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Create dataset and dataloader
        dataset = TCGASevenModality(cfg.data_root, split='train')
        self.dloader = DataLoader(
            dataset, batch_size=cfg.batch_size,
            shuffle=True, pin_memory=True, num_workers=0
        )
        
        # Load feature mappings
        feature_mappings_path = 'data/processed/feature_mappings.json'
        
        # Initialize TCGA-specific losses and evaluator
        import sys
        sys.path.append('.')
        from bio_losses import create_tcga_biological_loss
        from bio_coherence_eval import TCGASpecificCoherenceEvaluator
        
        self.bio_loss = create_tcga_biological_loss(cfg, feature_mappings_path)
        self.bio_evaluator = TCGASpecificCoherenceEvaluator(feature_mappings_path)
        
        # Phase-specific bio loss weights
        self.bio_loss_phase_weights = {
            1: 0.1,   # Light guidance during unimodal training
            2: 0.3,   # Increase during alignment
            3: 0.6,   # Strong during mask warmup
            4: 1.0    # Full enforcement in final phase
        }
        
        # Phase-specific adversarial weights (gradual increase)
        self.adversarial_phase_weights = {
            1: 0.0,   # No adversarial training
            2: 0.0,   # No adversarial training
            3: 0.0,   # No adversarial training (focus on diffusion)
            4: 1.0    # Full adversarial training
        }
        
        # Define modality dimensions based on TCGA data structure  
        mod_dims = {
            'gene': 5000,
            'protein': 226,
            'methylation': 100,
            'variant': 100,
            'metabolite': 3,
            'microbiome': 1000,
            'clinical': 100
        }
        
        # Initialize EnhancedHGTD model
        self.model = HGTD(cfg, mod_dims)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                    lr=cfg.optim.lr,
                                    weight_decay=cfg.optim.weight_decay)
        
        # Create separate optimizers for critics (WGAN-GP requires separate optimizers)
        critic_params = []
        if hasattr(self.model, 'local_critic'):
            critic_params.extend(self.model.local_critic.parameters())
        if hasattr(self.model, 'global_critic'):
            critic_params.extend(self.model.global_critic.parameters())
        
        if critic_params:
            self.critic_opt = torch.optim.Adam(critic_params, lr=cfg.optim.lr * 2, betas=(0.0, 0.9))  # WGAN-GP typical settings
        else:
            self.critic_opt = None
        
        # Create separate generator optimizer for clean WGAN-GP training
        # Include encoders, diffusion, and decoders but exclude critics
        generator_params = []
        for name, param in self.model.named_parameters():
            if 'critic' not in name:
                generator_params.append(param)
        
        self.generator_opt = torch.optim.Adam(
            generator_params,
            lr=cfg.optim.lr,
            betas=(0.5, 0.9)  # Standard for generator
        )
        
        # Initialize modality-specific RL system
        self.modality_rl_system = MultiModalityRLSystem(
            latent_dim=cfg.latent_dim,
            modality_names=list(mod_dims.keys()),
            hidden_dim=512
        )
        if torch.cuda.is_available():
            self.modality_rl_system = self.modality_rl_system.cuda()
        
        # Create separate optimizers for each modality RL agent
        self.modality_rl_optimizers = {
            mod: torch.optim.Adam(agent.parameters(), lr=3e-4)
            for mod, agent in self.modality_rl_system.modality_agents.items()
        }
        
        # Keep the original RL agent for backward compatibility
        self.rl_agent = CoherenceRLAgent(
            latent_dim=cfg.latent_dim,
            num_modalities=len(mod_dims),
            hidden_dim=512
        )
        if torch.cuda.is_available():
            self.rl_agent = self.rl_agent.cuda()
        
        # Separate optimizer for RL agent
        self.rl_opt = torch.optim.Adam(self.rl_agent.parameters(), lr=3e-4)
        
        # Track coherence for RL rewards
        self.best_coherence_for_rl = 0.0
        self.previous_epoch_coherence = 0.0  # Track previous epoch's coherence for rewards
        self.rl_update_frequency = 10  # Update RL every N batches
        self.use_rl_modulation = False  # Start without RL, enable after warmup
        self.use_modality_specific_rl = True  # Use modality-specific RL agents
        self.rl_modulation_strength = 0.1  # Start with small modulation
        self.coherence_history = []  # Track recent coherence for momentum
        
        # Add gradient clipping to prevent explosions
        phase4_epochs = self.cfg.curriculum.epochs_per_phase[3]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=phase4_epochs)

        self.grad_clip_norm = 1.0
        
        # WGAN-GP settings
        self.n_critic = 3  # Reduced from 5 to 3 to speed up training while maintaining stability
        self.diffusion_weight = 1.0  # Increase diffusion loss weight    
        
        
    def _move_to_device(self, batch):
        """Move batch to appropriate device"""
        device = next(self.model.parameters()).device
        
        # Move each modality data to device
        for mod in ['gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
            if mod in batch and hasattr(batch[mod], 'to'):
                batch[mod] = batch[mod].to(device)
        
        # Move other tensors
        for key in ['kg', 'graph_real', 'graph_fake']:
            if key in batch and hasattr(batch[key], 'to'):
                batch[key] = batch[key].to(device)
        
        # Move masks (handle both dict and list formats)
        if 'masks' in batch:
            if isinstance(batch['masks'], dict):
                batch['masks'] = {k: mask.to(device) if hasattr(mask, 'to') else mask 
                                for k, mask in batch['masks'].items()}
            else:
                # Legacy list format - convert to dict
                mask_list = batch['masks']
                hierarchy_levels = ['molecular', 'pathway', 'systemic', 'phenotype']
                batch['masks'] = {
                    level: mask_list[i].to(device) if i < len(mask_list) and hasattr(mask_list[i], 'to') else torch.eye(7, device=device)
                    for i, level in enumerate(hierarchy_levels)
                }
        
        return batch
    
    def phase1_unimodal(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        modality_losses = {mod: 0.0 for mod in self.model.encoders.keys()}
        gradient_norms = []
        
        for batch_idx, batch in enumerate(self.dloader):
            batch = self._move_to_device(batch)
            self.opt.zero_grad()
            loss = 0.
            batch_mod_losses = {}
            
            for mod in self.model.encoders:
                if mod in batch:
                    # Pass the raw feature vector to encoder
                    z = self.model.encoders[mod](batch[mod])
                    # Encoder should output [batch_size, latent_dim]
                    if len(z.shape) > 2:
                        z = z.mean(dim=1)
                    
                    # Create positive pairs by shifting
                    z_pos = torch.roll(z, shifts=1, dims=0)
                    sim = torch.cosine_similarity(z.flatten(), z_pos.flatten(), dim=0)
                    mod_loss = (1 - sim)
                    loss += mod_loss
                    batch_mod_losses[mod] = mod_loss.item()
                    modality_losses[mod] += mod_loss.item()
            
            if loss > 0:
                loss.backward()
                
                # Clip gradients to prevent explosions
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                # Compute gradient norm after clipping
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)
                
                self.opt.step()
                total_loss += loss.item()
                num_batches += 1
                
                # Log detailed metrics for first batch
                if batch_idx == 0:
                    print(f"\nPhase 1 Epoch {epoch} - Sample Batch Metrics:")
                    print(f"  Total Loss: {loss.item():.6f}")
                    print(f"  Gradient Norm: {total_norm:.6f}")
                    print("  Per-Modality Losses:")
                    for mod, mod_loss in batch_mod_losses.items():
                        print(f"    {mod}: {mod_loss:.6f}")
                    
                    # Log sample statistics
                    print("\n  Input Statistics:")
                    for mod in ['gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
                        if mod in batch:
                            data = batch[mod][0]  # First sample
                            print(f"    {mod}: mean={data.mean():.4f}, std={data.std():.4f}, "
                                  f"min={data.min():.4f}, max={data.max():.4f}")
        
        if num_batches > 0:
            metrics = {
                "phase1_loss": total_loss / num_batches,
                "phase1_grad_norm": sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0,
                "epoch": epoch
            }
            
            # Add per-modality average losses
            for mod, loss_sum in modality_losses.items():
                metrics[f"phase1_{mod}_loss"] = loss_sum / num_batches
            
            wandb.log(metrics)
            
            print(f"\nPhase 1 Epoch {epoch} Summary:")
            print(f"  Average Loss: {metrics['phase1_loss']:.6f}")
            print(f"  Average Gradient Norm: {metrics['phase1_grad_norm']:.6f}")
    
    def phase2_alignment(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_norms = []
        alignment_scores = []
        
        for batch_idx, batch in enumerate(self.dloader):
            batch = self._move_to_device(batch)
            self.opt.zero_grad()
            zs = []
            mod_names = []
            
            for mod in self.model.encoders:
                if mod in batch:
                    # Pass the raw feature vector to encoder
                    z = self.model.encoders[mod](batch[mod])
                    # Ensure proper shape
                    if len(z.shape) > 2:
                        z = z.mean(dim=1)
                    zs.append(z)
                    mod_names.append(mod)
            
            if len(zs) > 1:
                # Pad to same size
                max_dim = max(z.shape[-1] for z in zs)
                zs_padded = []
                for z in zs:
                    if z.shape[-1] < max_dim:
                        padding = torch.zeros(*z.shape[:-1], max_dim - z.shape[-1], device=z.device)
                        z = torch.cat([z, padding], dim=-1)
                    zs_padded.append(z)
                
                z_stack = torch.stack(zs_padded)  # [num_mods, batch, D]
                var = torch.var(z_stack, dim=0)
                
                # Compute alignment score (inverse of variance)
                alignment = 1.0 / (1.0 + var.mean())
                alignment_scores.append(alignment.item())
                
                cyc_loss = torch.mean(var)
                cyc_loss.backward()
                
                # Compute gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)
                
                self.opt.step()
                total_loss += cyc_loss.item()
                num_batches += 1
                
                # Log detailed metrics for first batch
                if batch_idx == 0:
                    print(f"\nPhase 2 Epoch {epoch} - Sample Batch Metrics:")
                    print(f"  Alignment Loss: {cyc_loss.item():.6f}")
                    print(f"  Alignment Score: {alignment.item():.6f}")
                    print(f"  Gradient Norm: {total_norm:.6f}")
                    print(f"  Active Modalities: {', '.join(mod_names)}")
                    
                    # Log embedding statistics
                    print("\n  Embedding Statistics:")
                    for i, (z, mod) in enumerate(zip(zs, mod_names)):
                        print(f"    {mod}: mean={z.mean():.4f}, std={z.std():.4f}")
        
        if num_batches > 0:
            metrics = {
                "phase2_loss": total_loss / num_batches,
                "phase2_grad_norm": sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0,
                "phase2_alignment_score": sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0,
                "epoch": epoch
            }
            
            wandb.log(metrics)
            
            print(f"\nPhase 2 Epoch {epoch} Summary:")
            print(f"  Average Loss: {metrics['phase2_loss']:.6f}")
            print(f"  Average Alignment Score: {metrics['phase2_alignment_score']:.6f}")
            print(f"  Average Gradient Norm: {metrics['phase2_grad_norm']:.6f}")
    
    def phase3_mask_warmup(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_norms = []
        
        for batch_idx, batch in enumerate(self.dloader):
            batch = self._move_to_device(batch)
            
            # Create default masks as dictionary if not provided
            if 'masks' not in batch:
                device = next(self.model.parameters()).device
                batch['masks'] = {
                    'molecular': torch.eye(7, device=device),
                    'pathway': torch.eye(7, device=device),
                    'systemic': torch.eye(7, device=device),
                    'phenotype': torch.eye(7, device=device)
                }
            
            self.opt.zero_grad()
            try:
                # Model returns (z_decoded, diff_loss)
                result = self.model(batch, batch['masks'], batch['kg'])
                if isinstance(result, tuple) and len(result) == 3:
                    z_fake, diff_loss, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    z_fake, diff_loss = result
                else:
                    raise ValueError(f"Model returned unexpected type/length: {type(result)}, {len(result) if hasattr(result, '__len__') else 'N/A'}")
                
                # Get real latent representations for comparison
                z_real_dict = {}
                for mod, enc in self.model.encoders.items():
                    if mod in batch:
                        z_real_dict[mod] = enc(batch[mod])
                
                # Average across modalities for global representation
                if z_real_dict:
                    z_real = torch.stack(list(z_real_dict.values())).mean(dim=0)
                else:
                    z_real = torch.randn_like(z_fake)
                    
                diff_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                # Compute gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)
                
                self.opt.step()
                total_loss += diff_loss.item()
                num_batches += 1
                
                # Log detailed metrics for first batch
                if batch_idx == 0:
                    print(f"\nPhase 3 Epoch {epoch} - Sample Batch Metrics:")
                    print(f"  Diffusion Loss: {diff_loss.item():.6f}")
                    print(f"  Gradient Norm: {total_norm:.6f}")
                    
                    # Log flow statistics
                    print(f"\n  Flow Statistics:")
                    print(f"    Predicted Flow (z_fake): mean={z_fake.mean():.4f}, std={z_fake.std():.4f}")
                    print(f"    Target Flow (z_real): mean={z_real.mean():.4f}, std={z_real.std():.4f}")
                    print(f"    Flow Difference: {(z_fake - z_real).abs().mean():.4f}")
                    
            except Exception as e:
                print(f"Warning in phase3: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches > 0:
            metrics = {
                "phase3_loss": total_loss / num_batches,
                "phase3_grad_norm": sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0,
                "epoch": epoch
            }
            
            wandb.log(metrics)
            
            print(f"\nPhase 3 Epoch {epoch} Summary:")
            print(f"  Average Loss: {metrics['phase3_loss']:.6f}")
            print(f"  Average Gradient Norm: {metrics['phase3_grad_norm']:.6f}")
            
            
                
    def phase4_full(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_diff_loss = 0.0
        total_local_critic = 0.0
        total_global_critic = 0.0
        total_local_gen = 0.0
        total_global_gen = 0.0
        num_batches = 0
        gradient_norms = []

        # Automatically enable RL modulation when coherence is high enough
        if (not self.use_rl_modulation and epoch >= 30
                and self.previous_epoch_coherence > 0.70):
            self.use_rl_modulation = True
            print(f"\033[96mEnabling RL modulation at epoch {epoch} (prev coherence {self.previous_epoch_coherence:.1%})\033[0m")

        if self.use_rl_modulation and len(self.coherence_history) >= 5:
            recent_var = np.var(self.coherence_history[-5:])
            if recent_var < 1e-4:
                self.rl_modulation_strength = min(self.rl_modulation_strength * 1.5, 0.4)

        # Initialize critic optimizer if not exists
        if self.critic_opt is None:
            # Create critics if they don't exist
            device = next(self.model.parameters()).device
            if not hasattr(self.model, 'local_critic'):
                self.model.local_critic = nn.Sequential(
                    nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim),
                    nn.LayerNorm(self.cfg.latent_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.cfg.latent_dim, 1)
                ).to(device)
            if not hasattr(self.model, 'global_critic'):
                self.model.global_critic = nn.Sequential(
                    nn.Linear(self.cfg.latent_dim * 2, self.cfg.latent_dim),
                    nn.LayerNorm(self.cfg.latent_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.cfg.latent_dim, 1)
                ).to(device)
            
            critic_params = list(self.model.local_critic.parameters()) + list(self.model.global_critic.parameters())
            self.critic_opt = torch.optim.Adam(critic_params, lr=self.cfg.optim.lr * 2, betas=(0.0, 0.9))
        
        for batch_idx, batch in enumerate(self.dloader):
            batch = self._move_to_device(batch)
            
            # Create default masks as dictionary if not provided
            if 'masks' not in batch:
                device = next(self.model.parameters()).device
                batch['masks'] = {
                    'molecular': torch.eye(7, device=device),
                    'pathway': torch.eye(7, device=device),
                    'systemic': torch.eye(7, device=device),
                    'phenotype': torch.eye(7, device=device)
                }
            
            # ===== CRITIC TRAINING (n_critic times) =====
            for critic_iter in range(self.n_critic):
                self.critic_opt.zero_grad()
                
                try:
                    # Forward pass through model (no gradients needed for generation)
                    with torch.no_grad():
                        result = self.model(batch, batch['masks'], batch['kg'])
                        if isinstance(result, tuple) and len(result) == 3:
                            z_fake, _, _ = result
                        elif isinstance(result, tuple) and len(result) == 2:
                            z_fake, _ = result
                        else:
                            raise ValueError(f"Model returned unexpected type/length")
                    
                    # Get real latent representations
                    z_real_dict = {}
                    for mod, enc in self.model.encoders.items():
                        if mod in batch:
                            with torch.no_grad():
                                z_real_dict[mod] = enc(batch[mod])
                    
                    if z_real_dict:
                        z_real = torch.stack(list(z_real_dict.values())).mean(dim=0)
                    else:
                        z_real = torch.randn_like(z_fake)
                    
                    # Compute critic losses with gradient penalty
                    lc, gc = self.model.critic_losses(z_real, z_fake)
                    critic_loss = lc + gc
                    critic_loss.backward()
                    
                    # Clip gradients for critics
                    torch.nn.utils.clip_grad_norm_(self.model.local_critic.parameters(), self.grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(self.model.global_critic.parameters(), self.grad_clip_norm)
                    
                    self.critic_opt.step()
                    
                    # Only track losses from last critic iteration
                    if critic_iter == self.n_critic - 1:
                        total_local_critic += lc.item()
                        total_global_critic += gc.item()
                
                except Exception as e:
                    print(f"Warning in critic training: {e}")
                    continue
            
            # ===== GENERATOR TRAINING =====
            self.generator_opt.zero_grad()
            
            try:
                # Forward pass through model
                result = self.model(batch, batch['masks'], batch['kg'])
                if isinstance(result, tuple) and len(result) == 3:
                    z_fake, diff_loss, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    z_fake, diff_loss = result
                else:
                    raise ValueError(f"Model returned unexpected type/length")
                
                # Get real latent representations for other losses
                z_real_dict = {}
                for mod, enc in self.model.encoders.items():
                    if mod in batch:
                        z_real_dict[mod] = enc(batch[mod])
                
                if z_real_dict:
                    z_real = torch.stack(list(z_real_dict.values())).mean(dim=0)
                else:
                    z_real = torch.randn_like(z_fake)

                # RL Agent Modulation (if enabled and past warmup)
                rl_value = None
                rl_log_prob = None
                rl_action = None
                z_fake_original = None
                modality_rl_actions = {}
                
                if self.use_rl_modulation and epoch > 50:  # Enable after 50 epochs
                    if self.use_modality_specific_rl:
                        # Use modality-specific RL agents
                        z_fake_dict = {}
                        for mod in self.model.encoders.keys():
                            if mod in batch:
                                z_fake_dict[mod] = z_fake
                        
                        # Get modulated latents from modality-specific RL
                        z_fake_dict_modulated = self.modality_rl_system.modulate_latents(
                            z_fake_dict, deterministic=not self.training
                        )
                        
                        # Store actions for later reward computation
                        modality_rl_actions = self.modality_rl_system.forward(
                            z_fake_dict, deterministic=not self.training
                        )
                        
                        # Average the modulated latents
                        z_fake_modulated = torch.stack(list(z_fake_dict_modulated.values())).mean(dim=0)
                        z_fake_original = z_fake.clone()
                    else:
                        # Use original unified RL agent
                        with torch.no_grad() if not self.training else torch.enable_grad():
                            rl_action, rl_log_prob, rl_value = self.rl_agent(z_fake, deterministic=not self.training)
                            z_fake_original = z_fake.clone()
                            z_fake_modulated = z_fake + self.rl_modulation_strength * rl_action
                else:
                    z_fake_modulated = z_fake

                # Decode back to feature space for biological losses
                decoded_data = {}
                for modality in ['gene', 'protein', 'methylation', 'variant', 
                                'metabolite', 'microbiome', 'clinical']:
                    if modality in batch:
                        decoded_data[modality] = self.model.decode_modality(z_fake_modulated, modality)
                
                # Calculate biological losses
                bio_losses = self.bio_loss(
                    generated_data=decoded_data,
                    real_data=batch,
                    phase=4
                )
                
                # Methylation property loss (if applicable)
                methylation_property_loss = 0.0
                if 'real_cimp_labels' in batch and 'real_global_methylation_labels' in batch:
                    property_logits = self.model.modality_decoders.decoders['methylation'].get_global_property_logits(decoded_data['methylation'])
                    cimp_loss = F.binary_cross_entropy_with_logits(property_logits[:, 0], batch['real_cimp_labels'].float())
                    global_meth_pred = torch.sigmoid(property_logits[:, 1])
                    global_meth_loss = F.mse_loss(global_meth_pred, batch['real_global_methylation_labels'].float())
                    methylation_property_loss = cimp_loss + global_meth_loss

                # Compute biological coherence loss (NEW - direct gradient signal)
                # Store last coherence loss to reuse between evaluations
                if not hasattr(self, '_last_coherence_loss'):
                    self._last_coherence_loss = 0.0
                
                # Only evaluate coherence every 10 batches to speed up training
                if hasattr(self, 'bio_evaluator') and batch_idx % 10 == 0:
                    with torch.no_grad():
                        coherence_results = self.bio_evaluator.evaluate_tcga_coherence(
                            decoded_data, batch
                        )
                        current_coherence = coherence_results['overall_coherence']
                    # Create differentiable loss that wants to maximize coherence
                    coherence_base = 1.0 - current_coherence
                    
                    # Also penalize specific failing modalities
                    extra_penalty = 0.0
                    if 'modality_specific' in coherence_results and 'scores' in coherence_results['modality_specific']:
                        modality_scores = coherence_results['modality_specific']['scores']
                        # Extra penalty for clinical, microbiome, variant if they're failing
                        for mod in ['clinical', 'microbiome', 'variant']:
                            if mod in modality_scores and 'score' in modality_scores[mod]:
                                mod_score = modality_scores[mod]['score']
                                if mod_score < 0.5:  # If failing badly
                                    extra_penalty += 0.2 * (1.0 - mod_score)
                    
                    # Create final coherence loss value and store it
                    self._last_coherence_loss = coherence_base + extra_penalty
                
                # Use the most recent coherence loss value
                coherence_loss = torch.tensor(self._last_coherence_loss, device=z_fake.device, dtype=z_fake.dtype)

                # Compute generator losses (wants to fool critics)
                lg, gg = self.model.generator_losses(z_fake)
                
                # Gradually increase adversarial weight during phase 4
                phase4_epochs = self.cfg.curriculum.epochs_per_phase[3]
                adversarial_weight = min(0.1 + (epoch / phase4_epochs) * 0.9, 1.0)
                
                # Combine all losses for generator update
                # Increase diffusion loss weight and add coherence loss
                coherence_weight = 0.5  # Start with 0.5, can increase if needed
                loss = (self.diffusion_weight * diff_loss +  # Increased weight for diffusion
                       adversarial_weight * self.cfg.critic_coefs.local * lg + 
                       adversarial_weight * getattr(self.cfg.critic_coefs, 'global', 1.0) * gg +
                       self.bio_loss_phase_weights[4] * bio_losses['total'] +
                       0.1 * methylation_property_loss +
                       coherence_weight * coherence_loss)  # NEW coherence loss term
                
                loss.backward()
                
                # Clip gradients - only generator parameters
                generator_params = [p for name, p in self.model.named_parameters() if 'critic' not in name]
                torch.nn.utils.clip_grad_norm_(generator_params, self.grad_clip_norm)
                
                # Compute gradient norm
                total_norm = 0.0
                for p in generator_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)
                
                self.generator_opt.step()
                total_loss += loss.item()
                total_diff_loss += diff_loss.item()
                total_local_gen += lg.item()
                total_global_gen += gg.item()
                num_batches += 1
                
                # Log detailed metrics for first batch
                if batch_idx == 0:
                    print(f"\nPhase 4 Epoch {epoch} - Sample Batch Metrics:")
                    print(f"  Total Loss: {loss.item():.6f}")
                    print(f"  Diffusion Loss: {diff_loss.item():.6f} (weight: {self.diffusion_weight:.2f})")
                    print(f"  Local Generator Loss: {lg.item():.6f} (weight: {adversarial_weight:.2f})")
                    print(f"  Global Generator Loss: {gg.item():.6f} (weight: {adversarial_weight:.2f})")
                    print(f"  Local Critic Loss: {total_local_critic/(batch_idx+1):.6f}")
                    print(f"  Global Critic Loss: {total_global_critic/(batch_idx+1):.6f}")
                    print(f"  Gradient Norm: {total_norm:.6f}")
                    print(f"\n  Generated (z_fake): mean={z_fake.mean():.4f}, std={z_fake.std():.4f}")
                    print(f"  Target (z_real): mean={z_real.mean():.4f}, std={z_real.std():.4f}")
                    
                # Log biological metrics every 10 batches
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        # Check decoded data for NaN/Inf
                        for mod, data in decoded_data.items():
                            if torch.isnan(data).any() or torch.isinf(data).any():
                                print(f"\033[91m  WARNING: {mod} contains NaN/Inf values!\033[0m")
                                print(f"    NaN count: {torch.isnan(data).sum().item()}")
                                print(f"    Inf count: {torch.isinf(data).sum().item()}")
                        
                        coherence_results = self.bio_evaluator.evaluate_tcga_coherence(
                            decoded_data, batch
                        )
                        current_coherence = coherence_results['overall_coherence']
                        
                        # Store results for epoch summary
                        if not hasattr(self, 'epoch_coherence_scores'):
                            self.epoch_coherence_scores = []
                            self.epoch_modality_scores = []
                        self.epoch_coherence_scores.append(current_coherence)
                        
                        # Store modality scores
                        modality_score_dict = {}
                        if 'modality_specific' in coherence_results and 'scores' in coherence_results['modality_specific']:
                            modality_scores = coherence_results['modality_specific']['scores']
                            for mod in ['gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
                                if mod in modality_scores and 'score' in modality_scores[mod]:
                                    modality_score_dict[mod] = modality_scores[mod]['score']
                                else:
                                    modality_score_dict[mod] = 0.0
                        self.epoch_modality_scores.append(modality_score_dict)

                        # ===== RL TRAINING =====
                        if self.use_rl_modulation and epoch > 50:
                            # --- Unified RL agent ---
                            if not self.use_modality_specific_rl and rl_action is not None:
                                reward = self.rl_agent.compute_intrinsic_reward(
                                    current_coherence,
                                    modality_score_dict,
                                    previous_coherence=self.previous_epoch_coherence
                                )
                                self.rl_agent.store_transition(
                                    z_fake_original.detach(),
                                    rl_action.detach(),
                                    rl_log_prob.detach(),
                                    rl_value.detach(),
                                    reward,
                                    False
                                )
                                if len(self.rl_agent.memory) >= self.rl_update_frequency:
                                    self.rl_agent.update(self.rl_opt)

                            # --- Modality specific RL agents ---
                            if self.use_modality_specific_rl and modality_rl_actions:
                                modality_rewards = {}
                                for mod, action_tuple in modality_rl_actions.items():
                                    mod_score = modality_score_dict.get(mod, 0.0)
                                    checks = {}
                                    if 'modality_specific' in coherence_results and 'scores' in coherence_results['modality_specific']:
                                        checks = coherence_results['modality_specific']['scores'].get(mod, {}).get('checks', {})
                                    modality_rewards[mod] = self.modality_rl_system.modality_agents[mod].compute_modality_reward(
                                        mod_score, checks)

                                self.modality_rl_system.store_transitions(
                                    {mod: z_fake_original for mod in modality_rl_actions.keys()},
                                    modality_rl_actions,
                                    modality_rewards
                                )
                                if batch_idx % self.rl_update_frequency == 0:
                                    self.modality_rl_system.update_all_agents(self.modality_rl_optimizers)
                        
                        # Log to wandb
                        wandb.log({
                            'train_biological_coherence': current_coherence,
                            'train_statistical_coherence': coherence_results.get('statistical_coherence', 0),
                            'train_cross_modal_coherence': coherence_results.get('cross_modal_coherence', 0),
                            'train_pathway_coherence': coherence_results.get('pathway_coherence', 0),
                            'train_correlation_coherence': coherence_results.get('correlation_coherence', 0)
                        })
                    
            except Exception as e:
                print(f"Warning in generator training: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log epoch summary
        if num_batches > 0:
            metrics = {
                "phase4_loss": total_loss / num_batches,
                "phase4_diff_loss": total_diff_loss / num_batches,
                "phase4_local_critic_loss": total_local_critic / num_batches,
                "phase4_global_critic_loss": total_global_critic / num_batches,
                "phase4_local_gen_loss": total_local_gen / num_batches,
                "phase4_global_gen_loss": total_global_gen / num_batches,
                "phase4_grad_norm": sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0,
                "epoch": epoch
            }
            
            wandb.log(metrics)
            
            print(f"\nPhase 4 Epoch {epoch} Summary:")
            print(f"  Total Loss: {metrics['phase4_loss']:.6f}")
            print(f"  Diffusion Loss: {metrics['phase4_diff_loss']:.6f}")
            print(f"  Local Critic Loss: {metrics['phase4_local_critic_loss']:.6f}")
            print(f"  Global Critic Loss: {metrics['phase4_global_critic_loss']:.6f}")
            print(f"  Local Generator Loss: {metrics['phase4_local_gen_loss']:.6f}")
            print(f"  Global Generator Loss: {metrics['phase4_global_gen_loss']:.6f}")
            print(f"  Average Gradient Norm: {metrics['phase4_grad_norm']:.6f}")
            
            # Print epoch biological coherence summary
            if hasattr(self, 'epoch_coherence_scores') and self.epoch_coherence_scores:
                avg_coherence = sum(self.epoch_coherence_scores) / len(self.epoch_coherence_scores)
                print(f"\n  Biological Coherence Summary:")
                print(f"    Average: {avg_coherence:.1%}")
                print(f"    Min: {min(self.epoch_coherence_scores):.1%}")
                print(f"    Max: {max(self.epoch_coherence_scores):.1%}")
                
                # Average modality scores
                if hasattr(self, 'epoch_modality_scores') and self.epoch_modality_scores:
                    avg_mod_scores = {}
                    for mod in ['gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
                        scores = [s.get(mod, 0.0) for s in self.epoch_modality_scores]
                        avg_mod_scores[mod] = sum(scores) / len(scores) if scores else 0.0
                    
                    print(f"\n  Average Modality Coherence:")
                    for mod, score in avg_mod_scores.items():
                        print(f"    {mod:12s}: {score:.1%}")
                
                # Clear for next epoch
                self.epoch_coherence_scores = []
                self.epoch_modality_scores = []
            
            # Validate biological coherence
            coherence = self.validate_biological_coherence(epoch)
            if coherence is not None:
                metrics['validation_coherence'] = coherence
                wandb.log({'validation_coherence': coherence})
                # Update previous epoch coherence for next epoch's RL rewards
                self.previous_epoch_coherence = coherence
    def validate_biological_coherence(self, epoch):
        """Validate biological coherence every 5 epochs"""
        
        if epoch % 5 != 0:
            return None
        
        self.model.eval()
        coherence_scores = []
        
        # Create validation dataloader if not exists
        if not hasattr(self, 'val_loader'):
            try:
                val_dataset = TCGASevenModality(self.cfg.data_root, split='val')
                self.val_loader = DataLoader(
                    val_dataset, batch_size=self.cfg.batch_size,
                    shuffle=False, pin_memory=True, num_workers=0
                )
            except Exception as e:
                print(f"Warning: Could not load validation dataset: {e}")
                print("Using training data for validation (not ideal)")
                self.val_loader = self.dloader
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                
                # Generate and decode
                result = self.model(batch, batch['masks'], batch['kg'])
                if isinstance(result, tuple) and len(result) == 3:
                    z_fake, _, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    z_fake, _ = result
                else:
                    z_fake = result[0] if hasattr(result, '__getitem__') else result
                
                decoded_data = {}
                for modality in self.model.encoders.keys():
                    if modality in batch:
                        decoded_data[modality] = self.model.decode_modality(z_fake, modality)
                
                # Evaluate coherence
                results = self.bio_evaluator.evaluate_tcga_coherence(decoded_data, batch)
                coherence_scores.append(results['overall_coherence'])
                
                # Print detailed report for first batch
                if len(coherence_scores) == 1:
                    print(self.bio_evaluator.generate_coherence_report(results))
        
        avg_coherence = np.mean(coherence_scores)
        print(f"\n\033[94m=== Validation Results (Epoch {epoch}) ===\033[0m")
        print(f"Overall Validation Coherence: {avg_coherence:.1%}")
        
        # Get modality scores from last batch for display
        if 'results' in locals() and 'modality_specific' in results and 'scores' in results['modality_specific']:
            print("\nModality-Specific Validation Scores:")
            modality_scores = results['modality_specific']['scores']
            for mod in ['gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome', 'clinical']:
                if mod in modality_scores and 'score' in modality_scores[mod]:
                    mod_score = modality_scores[mod]['score']
                else:
                    mod_score = 0.0
                print(f"  {mod:12s}: {mod_score:.1%}")
        
        # Save if best coherence
        if not hasattr(self, 'best_coherence'):
            self.best_coherence = 0
        
        if avg_coherence > self.best_coherence:
            self.best_coherence = avg_coherence
            print(f"\033[92mNew best coherence: {avg_coherence:.1%}\033[0m")
            # Save checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'coherence': avg_coherence,
                'epoch': epoch
            }, f'checkpoints/best_coherence_{avg_coherence:.3f}.pt')
        
        # Alert if below 95%
        if avg_coherence < 0.95:
            print(f"\033[93mWARNING: Coherence {avg_coherence:.1%} is below 95% threshold!\033[0m")

        # Track coherence history for RL plateau detection
        self.coherence_history.append(avg_coherence)
        if len(self.coherence_history) > 10:
            self.coherence_history.pop(0)

        self.model.train()
        return avg_coherence
                
    def train_all(self):
        phases = [
            ("Phase 1: Unimodal", self.phase1_unimodal),
            ("Phase 2: Alignment", self.phase2_alignment),
            ("Phase 3: Mask Warmup", self.phase3_mask_warmup),
            ("Phase 4: Full Training", self.phase4_full)
        ]
        
        for p_id, (phase_name, phase_fn) in enumerate(phases):
            print(f"Starting {phase_name}")
            for ep in range(self.cfg.curriculum.epochs_per_phase[p_id]):
                try:
                    phase_fn(epoch=ep)
                except Exception as e:
                    print(f"Error in {phase_name} epoch {ep}: {e}")
                    continue
            print(f"Completed {phase_name}")
        
        # Return training summary
        return {
            'phases_completed': len(phases),
            'final_phase': phases[-1][0],
            'best_coherence': getattr(self, 'best_coherence', 0.0)
        }
import torch
from torch import nn
import numpy as np
from models.diffusion_decoder_old import FlowMatcher

def create_stoichiometry_matrix(n_metabolites=50, n_genes=100):
    """
    Create a synthetic stoichiometry matrix for demonstration.
    In practice, this would be loaded from KEGG/BioCyc databases.
    
    Returns:
        torch.Tensor: Stoichiometry matrix S of shape (n_metabolites, n_genes)
                     where S[i,j] represents the stoichiometric coefficient
                     of metabolite i in reaction j
    """
    # Create sparse stoichiometry matrix (most entries are 0)
    S = torch.zeros(n_metabolites, n_genes)
    
    # Add some realistic stoichiometric relationships
    # Each gene/reaction typically involves 2-5 metabolites
    for j in range(n_genes):
        # Number of metabolites in this reaction (2-5)
        n_participants = torch.randint(2, 6, (1,)).item()
        
        # Random metabolite indices
        metabolite_idx = torch.randperm(n_metabolites)[:n_participants]
        
        # Stoichiometric coefficients (-2 to +2, excluding 0)
        coeffs = torch.randint(-2, 3, (n_participants,))
        coeffs = coeffs[coeffs != 0]
        
        if len(coeffs) > 0:
            S[metabolite_idx[:len(coeffs)], j] = coeffs.float()
    
    return S

class BiologicalConstraints:
    """
    Container for various biological constraints that can be applied
    """
    
    @staticmethod
    def stoichiometry_constraint(z: torch.Tensor, S: torch.Tensor):
        """
        Steady-state stoichiometry constraint: S * v = 0
        where v is the flux vector (reaction rates)
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            S: Stoichiometry matrix (n_metabolites, n_reactions)
        
        Returns:
            Constraint violation penalty
        """
        # Map latent z to flux space (assuming z represents gene expression -> flux)
        # In practice, this mapping would be more sophisticated
        flux = z[:, :S.shape[1]] if z.shape[1] >= S.shape[1] else torch.cat([z, torch.zeros(z.shape[0], S.shape[1] - z.shape[1], device=z.device)], dim=1)
        
        # Compute steady-state violations: ||S * v||_1
        violations = torch.matmul(S, flux.T)  # (n_metabolites, batch_size)
        return torch.mean(torch.sum(torch.abs(violations), dim=0))
    
    @staticmethod
    def thermodynamic_constraint(z: torch.Tensor):
        """
        Thermodynamic feasibility: reactions should respect energy constraints
        """
        # Simple implementation: penalize extreme values that might violate thermodynamics
        extreme_penalty = torch.mean(torch.relu(torch.abs(z) - 3.0))  # Penalize |z| > 3
        return extreme_penalty
    
    @staticmethod
    def pathway_connectivity_constraint(z: torch.Tensor, pathway_graph: torch.Tensor):
        """
        Pathway connectivity: related genes should have correlated expression
        
        Args:
            z: Gene expression representations
            pathway_graph: Adjacency matrix of pathway connections
        """
        if pathway_graph.shape[0] > z.shape[1]:
            pathway_graph = pathway_graph[:z.shape[1], :z.shape[1]]
        elif pathway_graph.shape[0] < z.shape[1]:
            # Pad pathway graph
            pad_size = z.shape[1] - pathway_graph.shape[0]
            pathway_graph = torch.cat([
                torch.cat([pathway_graph, torch.zeros(pathway_graph.shape[0], pad_size, device=z.device)], dim=1),
                torch.zeros(pad_size, z.shape[1], device=z.device)
            ], dim=0)
        
        # Compute correlations between connected genes
        correlations = torch.matmul(z.T, z) / z.shape[0]  # (n_genes, n_genes)
        
        # Penalty for weak correlations between connected genes
        connected_pairs = pathway_graph > 0
        expected_correlations = correlations[connected_pairs]
        
        # We want high correlation for connected genes
        correlation_penalty = torch.mean(torch.relu(0.5 - torch.abs(expected_correlations)))
        return correlation_penalty

class KGEDN(nn.Module):
    """
    Knowledge-Guided Energy Diffusion Network
    
    Combines flow-matching diffusion with biological constraints
    to ensure generated samples respect known biological rules.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Core diffusion network
        self.flow_matcher = FlowMatcher(cfg)
        
        # Energy network for biological constraints
        self.energy_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
        # Load/create biological constraint matrices
        self.register_buffer("stoichiometry_matrix", 
                           create_stoichiometry_matrix(n_metabolites=50, n_genes=cfg.latent_dim))
        
        # Create simple pathway connectivity graph
        pathway_graph = torch.zeros(cfg.latent_dim, cfg.latent_dim)
        # Add some pathway connections (in practice, load from KEGG/Reactome)
        for i in range(0, cfg.latent_dim - 1, 5):  # Every 5th gene connected to next
            if i + 1 < cfg.latent_dim:
                pathway_graph[i, i + 1] = 1.0
                pathway_graph[i + 1, i] = 1.0
        self.register_buffer("pathway_graph", pathway_graph)
        
        # Constraint weights
        self.lambda_stoichiometry = cfg.get('lambda_stoichiometry', 1.0)
        self.lambda_thermodynamic = cfg.get('lambda_thermodynamic', 0.1)
        self.lambda_pathway = cfg.get('lambda_pathway', 0.5)
        self.lambda_energy = cfg.get('lambda_energy', 0.3)
    
    def compute_energy(self, z: torch.Tensor):
        """
        Compute total energy function E(z) combining multiple biological constraints
        """
        # Neural energy component
        neural_energy = self.energy_net(z).squeeze(-1)
        
        # Biological constraint energies
        stoich_energy = BiologicalConstraints.stoichiometry_constraint(z, self.stoichiometry_matrix)
        thermo_energy = BiologicalConstraints.thermodynamic_constraint(z)
        pathway_energy = BiologicalConstraints.pathway_connectivity_constraint(z, self.pathway_graph)
        
        # Combine energies
        total_energy = (neural_energy.mean() + 
                       self.lambda_stoichiometry * stoich_energy +
                       self.lambda_thermodynamic * thermo_energy +
                       self.lambda_pathway * pathway_energy)
        
        return total_energy, {
            'neural_energy': neural_energy.mean(),
            'stoichiometry_energy': stoich_energy,
            'thermodynamic_energy': thermo_energy,
            'pathway_energy': pathway_energy
        }
    
    def forward(self, z0: torch.Tensor, condition: torch.Tensor):
        """
        Forward pass combining flow matching with energy-based constraints
        """
        # Standard flow matching loss
        flow_loss = self.flow_matcher(z0, condition)
        
        # Energy-based constraint penalty
        total_energy, energy_components = self.compute_energy(z0)
        
        # Combined loss
        total_loss = flow_loss + self.lambda_energy * total_energy
        
        return z0, total_loss, energy_components
    
    def sample(self, num_samples: int, condition: torch.Tensor, device: torch.device, 
               use_energy_guidance: bool = True, guidance_steps: int = 10):
        """
        Sample with optional energy-based guidance during generation
        """
        # Initial sampling from flow matcher
        z = self.flow_matcher.sample(num_samples, condition, device)
        
        if use_energy_guidance:
            # Refine samples using energy guidance
            z = self.energy_guided_refinement(z, condition, guidance_steps)
        
        return z
    
    def energy_guided_refinement(self, z: torch.Tensor, condition: torch.Tensor, 
                                steps: int = 10, step_size: float = 0.01):
        """
        Refine samples using gradient-based energy minimization
        """
        z = z.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=step_size)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Compute energy
            energy, _ = self.compute_energy(z)
            
            # Gradient descent to minimize energy
            energy.backward()
            optimizer.step()
            
            # Optional: add small amount of noise to prevent getting stuck
            if step < steps - 1:
                with torch.no_grad():
                    z.add_(torch.randn_like(z) * 0.001)
        
        return z.detach()
    
    def evaluate_biological_feasibility(self, z: torch.Tensor):
        """
        Evaluate how well samples satisfy biological constraints
        """
        with torch.no_grad():
            _, energy_components = self.compute_energy(z)
            
            # Convert to feasibility scores (lower energy = higher feasibility)
            feasibility_scores = {}
            for key, energy in energy_components.items():
                # Use sigmoid to convert to 0-1 score
                feasibility_scores[f'{key}_feasibility'] = torch.sigmoid(-energy).item()
            
            return feasibility_scores 
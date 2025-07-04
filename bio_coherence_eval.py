import torch
import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path

class TCGASpecificCoherenceEvaluator:
    """Biological coherence evaluation specifically for your TCGA dataset"""
    
    def __init__(self, feature_mappings_path: str):
        # Load the actual feature mappings
        with open(feature_mappings_path, 'r') as f:
            self.feature_mappings = json.load(f)
        
        # Cancer type mapping (from your documentation)
        self.cancer_types = {
            'AA': 'Adrenocortical',
            'BH': 'Breast', 
            'BP': 'Unknown_BP',
            'BR': 'Brain',
            'CV': 'Cervical',
            'DD': 'Unknown_DD'
        }
        
        # Clinical feature indices (first 9 are real)
        self.clinical_features = {
            0: 'submitter_id',
            1: 'primary_site',
            2: 'disease_type',
            3: 'gender',
            4: 'race',
            5: 'ethnicity',
            6: 'vital_status',
            7: 'age_at_diagnosis',
            8: 'days_to_death'
        }
        
        self._initialize_biological_knowledge()
        
    def _initialize_biological_knowledge(self):
        """Initialize TCGA-specific biological constraints"""
        
        # Cancer-type specific markers based on TCGA cancer types
        self.cancer_markers = {
            'BH': {  # Breast
                'genes': ['ESR1', 'PGR', 'ERBB2', 'MKI67'],
                'proteins': ['ER', 'PR', 'HER2', 'Ki67']
            },
            'BR': {  # Brain
                'genes': ['IDH1', 'IDH2', 'MGMT', 'EGFR'],
                'proteins': ['EGFR', 'EGFR_pY1173']
            },
            'CV': {  # Cervical
                'genes': ['CDKN2A', 'TP53', 'PIK3CA'],
                'proteins': ['p16', 'p53']
            },
            'AA': {  # Adrenocortical
                'genes': ['TP53', 'CTNNB1', 'CDKN2A'],
                'proteins': ['p53', 'Beta-catenin']
            }
        }
        
        # Key pathways relevant to your protein list
        self.pathway_proteins = {
            'PI3K_AKT': ['PI3K-p110', 'PI3K-p85', 'Akt', 'Akt_pS473', 'Akt_pT308', 
                         'PTEN', 'mTOR', 'mTOR_pS2448', 'p70S6K', 'p70S6K_pT389'],
            'MAPK': ['MEK1', 'MEK1_pS217_S221', 'p38_MAPK', 'p38_pT180_Y182',
                     'MAPK_pT202_Y204', 'JNK_pT183_Y185'],
            'Cell_Cycle': ['Rb', 'Rb_pS807_S811', 'Cyclin_D1', 'Cyclin_E1', 
                          'Cyclin_B1', 'CDK1', 'p27', 'p27_pT157', 'p27_pT198'],
            'Apoptosis': ['Bcl-2', 'Bcl-xL', 'Bax', 'Bad_pS112', 'Bid', 
                         'Caspase-3', 'Caspase-7_cleavedD198', 'PARP_cleaved'],
            'DNA_Damage': ['p53', 'Chk1', 'Chk1_pS345', 'Chk2', 'Chk2_pT68',
                          'ATM', 'ATM_pS1981', 'H2AX_pS139']
        }
        
        # Mutation patterns from your variant list
        self.mutation_patterns = {
            'hypermutated': ['TTN', 'MUC16', 'DST'],  # Often mutated due to size
            'driver_mutations': ['TP53', 'KRAS', 'PIK3CA', 'PTEN', 'BRAF'],
            'mutual_exclusive': [
                ['KRAS_Missense', 'NRAS_Missense', 'BRAF_Missense'],
                ['IDH1_Missense', 'IDH2_Missense']
            ]
        }
        
        # Methylation patterns (we'll use probe indices as proxy)
        # In reality, you'd want CpG island/shore/shelf annotations
        self.methylation_regions = {
            'hypermethylated_cancer': list(range(0, 20)),  # First 20 probes
            'hypomethylated_cancer': list(range(80, 100))  # Last 20 probes
        }
        
    def evaluate_tcga_coherence(self, generated_data: Dict[str, torch.Tensor],
                               real_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
        """Main evaluation function for TCGA data coherence"""
        
        results = {
            'overall_coherence': 0.0,
            'modality_specific': {},
            'cross_modal': {},
            'cancer_type_specific': {},
            'pathway_coherence': {},
            'mutation_patterns': {},
            'passes_95_threshold': False
        }
        
        # 1. Check modality-specific patterns
        results['modality_specific'] = self._check_modality_patterns(generated_data)
        
        # 2. Check cross-modal relationships
        results['cross_modal'] = self._check_cross_modal_tcga(generated_data)
        
        # 3. Check cancer-type specific patterns (if clinical data available)
        if 'clinical' in generated_data:
            results['cancer_type_specific'] = self._check_cancer_patterns(generated_data)
        
        # 4. Check pathway coherence with actual proteins
        if 'protein' in generated_data:
            results['pathway_coherence'] = self._check_pathway_coherence_tcga(generated_data['protein'])
        
        # 5. Check mutation patterns
        if 'variant' in generated_data:
            results['mutation_patterns'] = self._check_mutation_patterns_tcga(generated_data['variant'])
        
        # Calculate overall score
        scores = []
        weights = {
            'modality_specific': 0.25,
            'cross_modal': 0.30,
            'cancer_type_specific': 0.15,
            'pathway_coherence': 0.20,
            'mutation_patterns': 0.10
        }
        
        for component, weight in weights.items():
            if component in results and 'score' in results[component]:
                scores.append(results[component]['score'] * weight)
        
        results['overall_coherence'] = sum(scores) / sum(weights.values())
        results['passes_95_threshold'] = results['overall_coherence'] >= 0.95
        
        return results
    
    def _check_modality_patterns(self, data: Dict) -> Dict:
        """Check TCGA-specific modality patterns"""
        
        scores = {}
        
        # Gene expression patterns
        if 'gene' in data:
            gene_scores = self._check_gene_tcga(data['gene'])
            scores['gene'] = gene_scores
        
        # Protein/RPPA patterns
        if 'protein' in data:
            protein_scores = self._check_protein_rppa(data['protein'])
            scores['protein'] = protein_scores
        
        # Methylation beta values
        if 'methylation' in data:
            meth_scores = self._check_methylation_tcga(data['methylation'])
            scores['methylation'] = meth_scores
        
        # Clinical validity
        if 'clinical' in data:
            clinical_scores = self._check_clinical_tcga(data['clinical'])
            scores['clinical'] = clinical_scores
        
        # Variant patterns (using existing method)
        if 'variant' in data:
            variant_scores = self._check_variant_tcga(data['variant'])
            scores['variant'] = variant_scores
            
        # Metabolite patterns
        if 'metabolite' in data:
            metabolite_scores = self._check_metabolite_tcga(data['metabolite'])
            scores['metabolite'] = metabolite_scores
            
        # Microbiome patterns
        if 'microbiome' in data:
            microbiome_scores = self._check_microbiome_tcga(data['microbiome'])
            scores['microbiome'] = microbiome_scores
        
        # Weight modalities by biological importance
        # Core molecular modalities get higher weight
        modality_weights = {
            'gene': 1.0,        # Core molecular data
            'protein': 1.0,     # Core molecular data
            'methylation': 0.8, # Epigenetic regulation
            'variant': 0.8,     # Genetic mutations
            'metabolite': 0.7,  # Downstream molecular
            'microbiome': 0.5,  # Associated but not core
            'clinical': 0.3     # Demographic/phenotypic
        }
        
        weighted_scores = []
        total_weight = 0.0
        
        for modality, score_dict in scores.items():
            if 'score' in score_dict:
                weight = modality_weights.get(modality, 1.0)
                weighted_scores.append(score_dict['score'] * weight)
                total_weight += weight
        
        avg_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        
        return {
            'scores': scores,
            'score': avg_score
        }
    
    def _check_gene_tcga(self, gene_data: torch.Tensor) -> Dict:
        """Check gene expression patterns specific to TCGA"""
        
        gene_np = gene_data.detach().cpu().numpy()
        checks = {}
        
        # Debug: Check gene data statistics
        
        # 1. Check key oncogenes and tumor suppressors
        key_genes = {
            'oncogenes': ['MYC', 'KRAS', 'EGFR', 'ERBB2', 'MET', 'CCND1'],
            'tumor_suppressors': ['TP53', 'RB1', 'PTEN', 'CDKN2A', 'VHL', 'APC']
        }
        
        for gene_type, gene_list in key_genes.items():
            # Filter out None indices
            indices = [self.feature_mappings['gene_to_idx'].get(g) 
                      for g in gene_list 
                      if g in self.feature_mappings['gene_to_idx']]
            indices = [idx for idx in indices if idx is not None]
            
            if indices:
                expr_values = gene_np[:, indices]
                # These genes should show variation across samples
                cv = np.std(expr_values, axis=0) / (np.mean(expr_values, axis=0) + 1e-8)
                checks[f'{gene_type}_variation'] = float(np.mean(cv > 0.3))
                # print(f"{gene_type} CV: {cv}, variation check: {checks[f'{gene_type}_variation']}")
        
        # 2. Check ribosomal genes (should be highly expressed)
        ribo_genes = [g for g in self.feature_mappings['gene_list'] 
                     if g.startswith('RPS') or g.startswith('RPL') or 'rRNA' in g][:10]
        ribo_indices = [self.feature_mappings['gene_to_idx'][g] for g in ribo_genes 
                       if g in self.feature_mappings['gene_to_idx']]
        
        if ribo_indices:
            ribo_expr = gene_np[:, ribo_indices]
            # Ribosomal genes should be in top 25% of expression
            expr_percentile = np.percentile(gene_np, 75)
            checks['ribosomal_high_expr'] = float(np.mean(ribo_expr > expr_percentile))
        
        # 3. Check sex-specific genes
        sex_genes = {'XIST': 'female', 'RPS4Y1': 'male', 'UTY': 'male'}
        sex_indices = {g: self.feature_mappings['gene_to_idx'].get(g) 
                      for g in sex_genes if g in self.feature_mappings['gene_to_idx']}
        
        if sex_indices:
            # These should show bimodal distribution
            bimodal_scores = []
            for gene, idx in sex_indices.items():
                if idx is not None:
                    expr = gene_np[:, idx]
                    # Simple bimodality check: high variance, two distinct groups
                    bimodal = np.std(expr) > np.mean(expr) * 0.5
                    bimodal_scores.append(bimodal)
            
            if bimodal_scores:
                checks['sex_genes_bimodal'] = float(np.mean(bimodal_scores))
        
        # print(f"Gene checks: {checks}")
        score = np.mean(list(checks.values())) if checks else 0.5
        
        return {
            'score': score,
            'checks': checks
        }
    
    def _check_protein_rppa(self, protein_data: torch.Tensor) -> Dict:
        """Check RPPA-specific protein patterns"""
        
        protein_np = protein_data.detach().cpu().numpy()
        protein_list = self.feature_mappings['protein_list']
        checks = {}
        
        # 1. Check phosphorylation ratios for key proteins
        phospho_pairs = [
            ('Akt', ['Akt_pS473', 'Akt_pT308']),
            ('mTOR', ['mTOR_pS2448']),
            ('Rb', ['Rb_pS807_S811']),
            ('p70S6K', ['p70S6K_pT389']),
            ('MEK1', ['MEK1_pS217_S221'])
        ]
        
        phospho_ratios = []
        for base_protein, phospho_forms in phospho_pairs:
            if base_protein in protein_list:
                base_idx = protein_list.index(base_protein)
                for phospho in phospho_forms:
                    if phospho in protein_list:
                        phospho_idx = protein_list.index(phospho)
                        
                        # Check ratio is reasonable (0-100%)
                        ratio = protein_np[:, phospho_idx] / (protein_np[:, base_idx] + 1e-8)
                        valid_ratio = np.mean((ratio >= 0) & (ratio <= 1))
                        phospho_ratios.append(valid_ratio)
        
        if phospho_ratios:
            checks['phospho_ratio_valid'] = float(np.mean(phospho_ratios))
        
        # 2. Check cleaved/activated proteins
        cleaved_proteins = ['Caspase-3', 'Caspase-7_cleavedD198', 'PARP_cleaved']
        cleaved_indices = [protein_list.index(p) for p in cleaved_proteins if p in protein_list]
        
        if cleaved_indices:
            # Cleaved proteins should be lower than uncleaved on average
            cleaved_expr = protein_np[:, cleaved_indices]
            # Should be sparse (mostly inactive)
            checks['cleaved_sparse'] = float(np.mean(cleaved_expr < np.median(protein_np)))
        
        # 3. Check pathway coordination
        # Proteins in same pathway should correlate
        for pathway_name, pathway_proteins in self.pathway_proteins.items():
            indices = [protein_list.index(p) for p in pathway_proteins if p in protein_list]
            
            if len(indices) > 3:
                pathway_data = protein_np[:, indices]
                # Calculate average pairwise correlation
                corr_matrix = np.corrcoef(pathway_data.T)
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                avg_corr = np.mean(np.abs(corr_matrix[mask]))
                checks[f'{pathway_name}_coherence'] = float(avg_corr > 0.3)
        
        score = np.mean(list(checks.values())) if checks else 0.5
        
        return {
            'score': score,
            'checks': checks
        }
    
    def _check_methylation_tcga(self, meth_data: torch.Tensor) -> Dict:
        """Check methylation patterns for TCGA data"""
        
        meth_np = meth_data.detach().cpu().numpy()
        checks = {}
        
        # 1. All values should be valid beta values [0,1]
        checks['valid_beta_values'] = float(np.all((meth_np >= 0) & (meth_np <= 1)))
        
        # 2. Check for expected methylation distribution
        # Most CpGs are either highly methylated or unmethylated
        mean_meth_per_probe = np.mean(meth_np, axis=0)
        bimodal_probes = (mean_meth_per_probe < 0.3) | (mean_meth_per_probe > 0.7)
        checks['bimodal_distribution'] = float(np.mean(bimodal_probes) > 0.5)
        
        # 3. Check for CpG island methylator phenotype (CIMP)
        # Some samples should show hypermethylation
        hypermethylated_samples = np.mean(meth_np > 0.7, axis=1) > 0.3
        checks['cimp_present'] = float(np.any(hypermethylated_samples))
        
        # 4. Global hypomethylation in cancer
        # Average methylation should be slightly lower than normal (0.5)
        global_methylation = np.mean(meth_np)
        checks['global_hypomethylation'] = float(0.3 < global_methylation < 0.6)
        
        score = np.mean(list(checks.values()))
        
        return {
            'score': score,
            'checks': checks
        }
    
    def _check_clinical_tcga(self, clinical_data: torch.Tensor) -> Dict:
        """Check clinical data validity for TCGA"""
        
        clinical_np = clinical_data.detach().cpu().numpy()
        checks = {}
        
        # Only first 9 features are interpretable
        if clinical_np.shape[1] >= 9:
            # Age at diagnosis (index 7) should be reasonable
            age = clinical_np[:, 7]
            checks['age_valid'] = float(np.all((age >= 0) & (age <= 100)))
            
            # Days to death (index 8) should be non-negative
            if clinical_np.shape[1] > 8:
                survival = clinical_np[:, 8]
                checks['survival_valid'] = float(np.all(survival >= 0))
            
            # Gender (index 3) should be binary or small categorical
            if clinical_np.shape[1] > 3:
                gender = clinical_np[:, 3]
                unique_genders = np.unique(gender)
                checks['gender_valid'] = float(len(unique_genders) <= 3)
        
        score = np.mean(list(checks.values())) if checks else 0.7
        
        return {
            'score': score,
            'checks': checks
        }
    
    def _check_cross_modal_tcga(self, data: Dict) -> Dict:
        """Check cross-modal relationships specific to TCGA"""
        
        cross_scores = {}
        
        # 1. Gene-Protein correlation for matched pairs
        if 'gene' in data and 'protein' in data:
            gp_score = self._check_gene_protein_tcga(data['gene'], data['protein'])
            cross_scores['gene_protein'] = gp_score
        
        # 2. Mutation-Expression effects
        if 'variant' in data and 'gene' in data:
            mut_expr_score = self._check_mutation_expression_tcga(data['variant'], data['gene'])
            cross_scores['mutation_expression'] = mut_expr_score
        
        # 3. Clinical-Molecular associations
        if 'clinical' in data and 'gene' in data:
            clinical_mol_score = self._check_clinical_molecular_tcga(
                data['clinical'], data['gene'], data.get('protein')
            )
            cross_scores['clinical_molecular'] = clinical_mol_score
        
        avg_score = np.mean(list(cross_scores.values())) if cross_scores else 0.5
        
        return {
            'scores': cross_scores,
            'score': avg_score
        }
    
    def _check_gene_protein_tcga(self, gene_data: torch.Tensor, 
                                 protein_data: torch.Tensor) -> float:
        """Check gene-protein correlations for TCGA matched pairs"""
        
        gene_np = gene_data.detach().cpu().numpy()
        protein_np = protein_data.detach().cpu().numpy()
        
        # Define known gene-protein pairs from RPPA
        matched_pairs = [
            ('AKT1', 'Akt'), ('TP53', 'p53'), ('RB1', 'Rb'),
            ('EGFR', 'EGFR'), ('ERBB2', 'HER2'), ('MYC', 'c-Myc'),
            ('PTEN', 'PTEN'), ('BCL2', 'Bcl-2'), ('CCND1', 'Cyclin_D1'),
            ('CDH1', 'E-Cadherin'), ('CTNNB1', 'Beta-catenin')
        ]
        
        correlations = []
        for gene_name, protein_name in matched_pairs:
            if (gene_name in self.feature_mappings['gene_to_idx'] and 
                protein_name in self.feature_mappings['protein_list']):
                
                gene_idx = self.feature_mappings['gene_to_idx'][gene_name]
                protein_idx = self.feature_mappings['protein_list'].index(protein_name)
                
                # Calculate correlation
                corr = np.corrcoef(gene_np[:, gene_idx], protein_np[:, protein_idx])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            # Most correlations should be positive and moderate
            positive_corr = np.mean(np.array(correlations) > 0.3)
            return float(positive_corr)
        
        return 0.5
    
    def _check_mutation_expression_tcga(self, variant_data: torch.Tensor,
                                       gene_data: torch.Tensor) -> float:
        """Check mutation effects on expression"""
        
        variant_np = variant_data.detach().cpu().numpy()
        gene_np = gene_data.detach().cpu().numpy()
        variant_list = self.feature_mappings['variant_list']
        
        effect_scores = []
        
        # Check specific mutation-expression relationships
        mutation_effects = {
            'TP53': {'effect': 'variable', 'variants': ['TP53_Missense', 'TP53_Nonsense']},
            'PTEN': {'effect': 'decrease', 'variants': ['PTEN_Missense', 'PTEN_Nonsense']},
            'PIK3CA': {'effect': 'increase', 'variants': ['PIK3CA_Missense']}
        }
        
        for gene_name, info in mutation_effects.items():
            if gene_name in self.feature_mappings['gene_to_idx']:
                gene_idx = self.feature_mappings['gene_to_idx'][gene_name]
                
                for variant_name in info['variants']:
                    if variant_name in variant_list:
                        var_idx = variant_list.index(variant_name)
                        
                        # Check if mutation affects expression
                        has_mutation = variant_np[:, var_idx] > 0
                        
                        if np.sum(has_mutation) > 3 and np.sum(~has_mutation) > 3:
                            expr_with = gene_np[has_mutation, gene_idx]
                            expr_without = gene_np[~has_mutation, gene_idx]
                            
                            if info['effect'] == 'decrease':
                                # Mutation should decrease expression
                                effect_correct = np.mean(expr_with) < np.mean(expr_without)
                            elif info['effect'] == 'increase':
                                # Mutation should increase expression
                                effect_correct = np.mean(expr_with) > np.mean(expr_without)
                            else:  # variable
                                # Just check for difference
                                effect_correct = np.abs(np.mean(expr_with) - np.mean(expr_without)) > 0.1
                            
                            effect_scores.append(float(effect_correct))
        
        return np.mean(effect_scores) if effect_scores else 0.6
    
    def _check_clinical_molecular_tcga(self, clinical_data: torch.Tensor,
                                      gene_data: torch.Tensor,
                                      protein_data: Optional[torch.Tensor]) -> float:
        """Check clinical-molecular associations"""
        
        clinical_np = clinical_data.detach().cpu().numpy()
        gene_np = gene_data.detach().cpu().numpy()
        
        association_scores = []
        
        # Check cancer-type specific markers (using primary_site at index 1)
        if clinical_np.shape[1] > 1:
            primary_sites = clinical_np[:, 1]
            unique_sites = np.unique(primary_sites)
            
            # For each cancer type, check if known markers are elevated
            for site_code in unique_sites:
                site_mask = primary_sites == site_code
                
                if np.sum(site_mask) > 5:  # Need enough samples
                    # Map to our cancer types (simplified)
                    for cancer_code, markers in self.cancer_markers.items():
                        marker_indices = [self.feature_mappings['gene_to_idx'].get(g)
                                        for g in markers['genes']
                                        if g in self.feature_mappings['gene_to_idx']]
                        
                        if marker_indices:
                            # Check if markers are differentially expressed
                            marker_expr_cancer = gene_np[site_mask][:, marker_indices]
                            marker_expr_other = gene_np[~site_mask][:, marker_indices]
                            
                            # Simple test: markers should be different between groups
                            diff_expr = np.abs(np.mean(marker_expr_cancer) - 
                                             np.mean(marker_expr_other)) > 0.5
                            association_scores.append(float(diff_expr))
        
        # Check age associations (older patients might have different profiles)
        if clinical_np.shape[1] > 7:
            age = clinical_np[:, 7]
            # Split by median age
            median_age = np.median(age)
            young = age < median_age
            old = age >= median_age
            
            # Check if cell cycle genes differ by age
            cell_cycle_genes = ['CCND1', 'CDK4', 'MKI67', 'PCNA']
            cc_indices = [self.feature_mappings['gene_to_idx'].get(g)
                         for g in cell_cycle_genes
                         if g in self.feature_mappings['gene_to_idx']]
            
            if cc_indices and np.sum(young) > 5 and np.sum(old) > 5:
                cc_young = gene_np[young][:, cc_indices].mean()
                cc_old = gene_np[old][:, cc_indices].mean()
                # Cell cycle genes often higher in younger patients
                association_scores.append(float(cc_young > cc_old))
        
        return np.mean(association_scores) if association_scores else 0.6
    
    def _check_cancer_patterns(self, data: Dict) -> Dict:
        """Check cancer-type specific patterns"""
        
        clinical_np = data['clinical'].detach().cpu().numpy()
        scores = {}
        
        # Only check if we have primary_site information
        if clinical_np.shape[1] > 1:
            primary_sites = clinical_np[:, 1]
            unique_sites = np.unique(primary_sites)
            
            for site in unique_sites:
                site_mask = primary_sites == site
                if np.sum(site_mask) < 3:  # Need minimum samples
                    continue
                
                site_scores = {}
                
                # Check molecular signatures for this cancer type
                if 'gene' in data:
                    # Simplified check - real implementation would use specific signatures
                    gene_subset = data['gene'][site_mask]
                    # Check if expression patterns are consistent within cancer type
                    gene_var = torch.var(gene_subset, dim=0).mean().item()
                    site_scores['expression_consistency'] = float(gene_var < 2.0)
                
                if 'protein' in data and site_scores:
                    scores[f'site_{int(site)}'] = np.mean(list(site_scores.values()))
        
        avg_score = np.mean(list(scores.values())) if scores else 0.7
        
        return {
            'site_scores': scores,
            'score': avg_score
        }
    
    def _check_pathway_coherence_tcga(self, protein_data: torch.Tensor) -> Dict:
        """Check pathway coherence using actual RPPA proteins"""
        
        protein_np = protein_data.detach().cpu().numpy()
        protein_list = self.feature_mappings['protein_list']
        pathway_scores = {}
        
        # Debug (commented out for cleaner output)
        # print(f"Checking pathway coherence for {len(self.pathway_proteins)} pathways")
        
        for pathway_name, pathway_proteins in self.pathway_proteins.items():
            # Get indices for proteins in this pathway
            indices = [protein_list.index(p) for p in pathway_proteins 
                      if p in protein_list]
            
            if len(indices) < 3:  # Need at least 3 proteins
                # print(f"  {pathway_name}: Only {len(indices)} proteins found, skipping")
                continue
            
            pathway_data = protein_np[:, indices]
            
            # Check coherence metrics
            checks = {}
            
            # 1. Proteins should correlate within pathway
            corr_matrix = np.corrcoef(pathway_data.T)
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            avg_corr = np.mean(corr_matrix[mask])
            checks['correlation'] = float(avg_corr > 0.2)
            
            # Debug (commented out for cleaner output)
            # print(f"  {pathway_name}: avg_corr={avg_corr:.3f}, check={checks['correlation']}")
            
            # 2. Phosphorylated and total forms should correlate
            phospho_pairs = []
            proteins_in_pathway = [p for p in pathway_proteins if p in protein_list]
            for i, protein1 in enumerate(proteins_in_pathway[:len(indices)]):
                if '_p' in protein1:
                    base = protein1.split('_p')[0]
                    for j, protein2 in enumerate(proteins_in_pathway[:len(indices)]):
                        if protein2 == base:
                            corr = np.corrcoef(pathway_data[:, i], pathway_data[:, j])[0, 1]
                            phospho_pairs.append(corr > 0.5)
            
            if phospho_pairs:
                checks['phospho_correlation'] = float(np.mean(phospho_pairs))
            
            pathway_scores[pathway_name] = np.mean(list(checks.values()))
            # print(f"  {pathway_name}: score={pathway_scores[pathway_name]:.3f}")
        
        avg_score = np.mean(list(pathway_scores.values())) if pathway_scores else 0.7
        
        return {
            'pathway_scores': pathway_scores,
            'score': avg_score
        }
    
    def _check_mutation_patterns_tcga(self, variant_data: torch.Tensor) -> Dict:
        """Check mutation patterns specific to TCGA"""
        
        variant_np = variant_data.detach().cpu().numpy()
        variant_list = self.feature_mappings['variant_list']
        checks = {}
        
        # 1. Check mutation frequency patterns
        mutation_freq = np.mean(variant_np > 0, axis=0)
        
        # TTN should be highly mutated (large gene)
        ttn_variants = [i for i, v in enumerate(variant_list) if v.startswith('TTN_')]
        if ttn_variants:
            ttn_freq = mutation_freq[ttn_variants].mean()
            checks['ttn_hypermutated'] = float(ttn_freq > 0.1)
        
        # 2. Check mutual exclusivity
        for exclusive_group in self.mutation_patterns['mutual_exclusive']:
            indices = [variant_list.index(var) for var in exclusive_group 
                      if var in variant_list]
            
            if len(indices) > 1:
                # Check co-occurrence is rare
                co_occur = np.sum(variant_np[:, indices] > 0, axis=1) > 1
                checks[f'exclusive_{exclusive_group[0]}'] = float(np.mean(co_occur) < 0.1)
        
        # 3. Driver mutations should be less frequent than passengers
        driver_indices = []
        passenger_indices = []
        
        for i, variant in enumerate(variant_list):
            gene = variant.split('_')[0]
            if gene in self.mutation_patterns['driver_mutations']:
                driver_indices.append(i)
            elif gene in self.mutation_patterns['hypermutated']:
                passenger_indices.append(i)
        
        if driver_indices and passenger_indices:
            driver_freq = mutation_freq[driver_indices].mean()
            passenger_freq = mutation_freq[passenger_indices].mean()
            checks['driver_passenger_ratio'] = float(driver_freq < passenger_freq)
        
        score = np.mean(list(checks.values())) if checks else 0.7
        
        return {
            'score': score,
            'checks': checks
        }
    
    def generate_coherence_report(self, results: Dict) -> str:
        """Generate detailed TCGA-specific coherence report"""
        
        report = []
        report.append("="*70)
        report.append("TCGA BIOLOGICAL COHERENCE EVALUATION REPORT")
        report.append("="*70)
        report.append(f"\n{'OVERALL COHERENCE SCORE:':<30} {results['overall_coherence']:.1%}")
        report.append(f"{'PASSES 95% THRESHOLD:':<30} {'✓ YES' if results['passes_95_threshold'] else '✗ NO'}")
        
        # Modality-specific scores
        if 'modality_specific' in results:
            report.append(f"\n{'='*70}")
            report.append("MODALITY-SPECIFIC COHERENCE:")
            report.append("-"*70)
            
            for modality, scores in results['modality_specific']['scores'].items():
                if 'score' in scores:
                    status = "✓" if scores['score'] >= 0.8 else "✗"
                    report.append(f"{status} {modality.upper()}: {scores['score']:.1%}")
                    
                    for check, value in scores['checks'].items():
                        check_status = "✓" if value >= 0.8 else "✗" if value < 0.6 else "!"
                        report.append(f"    {check_status} {check}: {value:.1%}")
        
        # Cross-modal relationships
        if 'cross_modal' in results:
            report.append(f"\n{'='*70}")
            report.append("CROSS-MODAL RELATIONSHIPS:")
            report.append("-"*70)
            
            for relationship, score in results['cross_modal']['scores'].items():
                status = "✓" if score >= 0.7 else "✗"
                report.append(f"{status} {relationship}: {score:.1%}")
        
        # Pathway coherence
        if 'pathway_coherence' in results and 'pathway_scores' in results['pathway_coherence']:
            report.append(f"\n{'='*70}")
            report.append("PATHWAY COHERENCE:")
            report.append("-"*70)
            
            for pathway, score in results['pathway_coherence']['pathway_scores'].items():
                status = "✓" if score >= 0.7 else "✗"
                report.append(f"{status} {pathway}: {score:.1%}")
        
        # Critical issues
        report.append(f"\n{'='*70}")
        report.append("CRITICAL ISSUES FOR 95% COHERENCE:")
        report.append("-"*70)
        
        issues = []
        if results['overall_coherence'] < 0.95:
            # Check each component
            components = ['modality_specific', 'cross_modal', 'pathway_coherence', 'mutation_patterns']
            for comp in components:
                if comp in results and 'score' in results[comp] and results[comp]['score'] < 0.8:
                    issues.append(f"- {comp.replace('_', ' ').title()}: {results[comp]['score']:.1%}")
        
        if issues:
            report.extend(issues)
        else:
            report.append("✓ No critical issues - all components above threshold")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def _check_variant_tcga(self, variant_data: torch.Tensor) -> Dict:
        """Check variant patterns for TCGA data"""
        
        variant_np = variant_data.detach().cpu().numpy()
        checks = {}
        
        # 1. Variants should be sparse (most genes not mutated)
        sparsity = np.mean(variant_np == 0)
        checks['sparsity'] = float(sparsity > 0.9)  # >90% should be zero
        
        # 2. Non-negative values
        checks['non_negative'] = float(np.all(variant_np >= 0))
        
        # 3. Mutation burden per sample (some samples hypermutated)
        mutation_burden = np.sum(variant_np > 0, axis=1)
        # Should have variation in mutation burden
        cv = np.std(mutation_burden) / (np.mean(mutation_burden) + 1e-8)
        checks['mutation_burden_variation'] = float(cv > 0.5)
        
        # 4. Some samples should be hypermutated
        hypermutated = mutation_burden > np.percentile(mutation_burden, 90)
        checks['hypermutated_samples'] = float(np.any(hypermutated))
        
        score = np.mean(list(checks.values()))
        
        return {
            'score': score,
            'checks': checks
        }
    
    def _check_metabolite_tcga(self, metabolite_data: torch.Tensor) -> Dict:
        """Check metabolite patterns for TCGA data"""
        
        metabolite_np = metabolite_data.detach().cpu().numpy()
        checks = {}
        
        # Debug: Check metabolite data statistics (commented out for cleaner output)
        # print(f"Metabolite shape: {metabolite_np.shape}, min: {metabolite_np.min():.3f}, max: {metabolite_np.max():.3f}, mean: {metabolite_np.mean():.3f}")
        
        # 1. All values should be non-negative (concentrations)
        checks['non_negative'] = float(np.all(metabolite_np >= 0))
        
        # 2. Should have reasonable dynamic range
        if metabolite_np.max() > 0:
            dynamic_range = metabolite_np.max() / (metabolite_np.min() + 1e-8)
            checks['dynamic_range'] = float(10 < dynamic_range < 10000)
            # print(f"Dynamic range: {dynamic_range:.1f}, check: {checks['dynamic_range']}")
        else:
            checks['dynamic_range'] = 0.0
        
        # 3. Distribution should be roughly log-normal
        log_data = np.log1p(metabolite_np + 1e-8)
        # Check if log-transformed data is more normal
        skewness = np.abs(np.mean((log_data - np.mean(log_data))**3) / (np.std(log_data)**3 + 1e-8))
        checks['log_normal_dist'] = float(skewness < 2.0)
        # print(f"Log-transformed skewness: {skewness:.3f}, check: {checks['log_normal_dist']}")
        
        # print(f"Metabolite checks: {checks}")
        score = np.mean(list(checks.values())) if checks else 0.7
        
        return {
            'score': score,
            'checks': checks
        }
    
    def _check_microbiome_tcga(self, microbiome_data: torch.Tensor) -> Dict:
        """Check microbiome abundance patterns"""
        
        microbiome_np = microbiome_data.detach().cpu().numpy()
        checks = {}
        
        # Debug: Check microbiome data statistics (commented out for cleaner output)
        # print(f"Microbiome shape: {microbiome_np.shape}, min: {microbiome_np.min():.3f}, max: {microbiome_np.max():.3f}, mean: {microbiome_np.mean():.3f}")
        
        # 1. All values should be non-negative (abundances)
        checks['non_negative'] = float(np.all(microbiome_np >= 0))
        
        # 2. Should sum to approximately 1 per sample (relative abundances)
        sample_sums = np.sum(microbiome_np, axis=1)
        checks['normalized'] = float(np.all(np.abs(sample_sums - 1.0) < 0.1))
        # print(f"Sample sums - min: {sample_sums.min():.3f}, max: {sample_sums.max():.3f}, mean: {sample_sums.mean():.3f}")
        
        # 3. Should be sparse (many taxa absent)
        sparsity = np.mean(microbiome_np == 0)
        checks['sparsity'] = float(0.3 < sparsity < 0.8)
        # print(f"Sparsity: {sparsity:.3f}, check: {checks['sparsity']}")
        
        # 4. Diversity check - samples should have different dominant taxa
        dominant_taxa = np.argmax(microbiome_np, axis=1)
        unique_dominant = len(np.unique(dominant_taxa))
        checks['diversity'] = float(unique_dominant > microbiome_np.shape[0] * 0.3)
        # print(f"Unique dominant taxa: {unique_dominant}/{microbiome_np.shape[0]}, check: {checks['diversity']}")
        
        # print(f"Microbiome checks: {checks}")
        score = np.mean(list(checks.values()))
        
        return {
            'score': score,
            'checks': checks
        }
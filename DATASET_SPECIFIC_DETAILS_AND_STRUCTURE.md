# Feature Mappings Documentation

## Overview

This document describes the feature mappings for the multi-omics TCGA dataset. The `feature_mappings.json` file contains mappings from feature indices to biological entity names across 7 different modalities.

## Data Summary

- **Total Samples**: 421 (294 train, 63 validation, 64 test)
- **Cancer Types**: 6 (AA, BH, BP, BR, CV, DD)
- **Modalities**: 7 (Gene Expression, Proteomics, Clinical, Methylation, Variants, metabolite, Microbiome)

## Feature Mappings by Modality

### 1. Gene Expression (5,000 features)

- **Feature Names**: Real gene symbols from TCGA RNA-seq data
- **Examples**: `5S_rRNA`, `A1BG`, `A2M`, `AAAS`, `ABAT`, `TP53`, `EGFR`, `BRCA1`
- **Source**: BigQuery `isb-cgc.TCGA_hg38_data_v0.RNAseq_gene`
- **Notes**:
  - Sorted alphabetically, starting with ribosomal RNAs
  - Includes both protein-coding and non-coding genes
  - Expression values are normalized (z-scored)

### 2. Proteomics (226 features)

- **Feature Names**: Protein names with phosphorylation states
- **Examples**:
  - `14-3-3_beta`, `14-3-3_epsilon`
  - `Akt`, `Akt_pS473`, `Akt_pT308` (phosphorylated forms)
  - `4E-BP1`, `4E-BP1_pS65`, `4E-BP1_pT70`
  - `p53`, `EGFR`, `HER2`
- **Source**: RPPA (Reverse Phase Protein Array) data from knowledge graph
- **Notes**:
  - Includes total protein and phosphorylated forms
  - Critical for understanding signaling pathways
  - Phosphorylation sites indicated (e.g., pS473 = phospho-Serine 473)

### 3. Clinical Features (100 features)

- **Real Feature Names** (first 9):
  - `submitter_id`: TCGA sample barcode
  - `primary_site`: Tumor location (e.g., Colon, Breast, Brain)
  - `disease_type`: Cancer classification
  - `gender`: Patient gender
  - `race`: Patient race
  - `ethnicity`: Patient ethnicity
  - `vital_status`: Alive/Dead
  - `age_at_diagnosis`: Age when diagnosed
  - `days_to_death`: Survival time
- **Generic Names** (remaining 91): `clinical_feat_9` through `clinical_feat_99`
- **Notes**: First 9 features are interpretable clinical variables

### 4. Methylation (100 features)

- **Feature Names**: CpG probe IDs from Illumina methylation arrays
- **Examples**: `cg00000029`, `cg00000165`, `cg00000236`, `cg00000289`
- **Source**: TCGA methylation data (450K/850K arrays)
- **Total Available**: 397,665 probes (using top 100)
- **Notes**:
  - Each probe measures methylation at specific CpG sites
  - Values represent beta values (0-1, proportion methylated)
  - Can be mapped to genes via probe annotations

### 5. Variants/Mutations (100 features)

- **Feature Names**: Gene + mutation type/location
- **Examples**:
  - `TTN_Silent`: Silent mutation in Titin gene
  - `TTN_Intron`: Intronic variant
  - `MUC16_Silent`: Silent mutation in MUC16 (CA125)
  - `DST_Intron`: Dystonin intronic variant
  - `FUT9_3'UTR`: 3' untranslated region variant
- **Source**: TCGA somatic mutation data
- **Notes**:
  - Most frequent mutations across the cohort
  - Includes coding and non-coding variants
  - TTN frequently mutated due to its large size

### 6. metabolite (3 features)

- **Feature Names**: Generic identifiers
  - `METABOLITE_000`
  - `METABOLITE_001`
  - `METABOLITE_002`
- **Notes**:
  - Limited metabolite data in TCGA
  - Would need external mapping for metabolite names
  - Likely represents key metabolites or metabolic signatures

### 7. Microbiome (1,000 features)

- **Feature Names**: OTU (Operational Taxonomic Unit) identifiers
- **Format**: `OTU_0000` through `OTU_0999`
- **Notes**:
  - Not standard TCGA data - from specialized studies
  - Each OTU represents a microbial taxon
  - Would need taxonomy mapping for species identification
  - Likely from 16S rRNA sequencing

## File Structure

### feature_mappings.json

```json
{
  "gene_list": ["5S_rRNA", "A1BG", ...],
  "gene_to_idx": {"5S_rRNA": 0, "A1BG": 1, ...},
  "idx_to_gene": {"0": "5S_rRNA", "1": "A1BG", ...},
  "total_genes": 5000,

  "protein_list": [...],
  "protein_to_idx": {...},
  "idx_to_protein": {...},
  "total_proteins": 226,

  // Similar structure for other modalities
}
```

## Usage in Biological Loss Functions

### Gene Regulatory Loss

- Uses real gene symbols to apply biologically meaningful constraints
- Can enforce known gene-gene interactions (e.g., transcription factor targets)
- Example: TP53 regulating cell cycle genes

### Protein Signaling Loss

- Models phosphorylation cascades (e.g., Akt → mTOR pathway)
- Distinguishes between total and phosphorylated protein forms
- Can enforce known kinase-substrate relationships

### Multi-omics Integration

- Gene-protein relationships (which genes code for which proteins)
- Methylation effects on gene expression (promoter methylation)
- Mutation impacts on protein function

## Data Processing Pipeline

1. **Raw Data**: BigQuery TCGA tables + Knowledge Graph
2. **Feature Extraction**:
   - Genes: Top 5,000 by expression
   - Proteins: All 226 RPPA targets
   - Methylation: Top 100 variable probes
   - Variants: Top 100 frequent mutations
3. **Normalization**: Z-scoring for continuous features
4. **Integration**: Combined into multi-modal tensors

## Notes and Limitations

- **Metabolites**: Only 3 features with generic names - minimal impact on model
- **Microbiome**: Generic OTU names - would need external taxonomy for interpretation
- **Clinical**: Mix of real (9) and generic (91) features
- **Missing Data**: Handled during preprocessing, not all samples have all modalities

## References

- TCGA Data: https://www.cancer.gov/tcga
- ISB-CGC BigQuery: https://isb-cgc.appspot.com/
- Feature Selection: Based on variance and biological relevance
- Knowledge Graph: Custom built from TCGA relationships

---

_Generated: [Current Date]_  
_Dataset: TCGA Multi-omics (AA, BH, BP, BR, CV, DD cohorts)_

# MULTI-OMICS FEATURE INFORMATION

TOTAL FEATURES: 6529

## BREAKDOWN BY MODALITY:

GENE:
Count: 5000
Type: continuous
Source: TCGA RNA-seq (FPKM/TPM)
Examples: 5S_rRNA, A1BG, TP53

PROTEIN:
Count: 226
Type: continuous
Source: RPPA (Reverse Phase Protein Array)
Examples: Akt, Akt_pS473, p53

CLINICAL:
Count: 100
Type: mixed (categorical/continuous)
Source: N/A

METHYLATION:
Count: 100
Type: continuous
Source: Illumina 450K/850K methylation array
Examples: cg00000029, cg00000165, cg00000236

VARIANT:
Count: 100
Type: binary or count
Source: TCGA somatic mutations (MAF files)
Examples: TTN_Silent, TP53_p.R273H, KRAS_p.G12D

METABOLITE:
Count: 3
Type: continuous
Source: limited metabolite data

MICROBIOME:
Count: 1000
Type: continuous
Source: 16S rRNA sequencing (non-standard TCGA)

INTERPRETABLE FEATURES: gene, protein, clinical (partial), methylation, variant
GENERIC FEATURES: metabolite, microbiome

For detailed information, import this module and access FEATURE_INFO

# 🧬 Multi-Omics Feature Quick Reference

## 📊 Dataset Overview

**421 TCGA Samples** | **7 Modalities** | **6,429 Total Features**

---

## 🔬 Feature Breakdown

### Gene Expression (5,000)

```
✓ Real gene symbols (A1BG, TP53, EGFR...)
✓ RNA-seq, z-scored
✓ Both coding & non-coding genes
```

### Proteomics (226)

```
✓ Protein + phosphorylation states
✓ Examples: Akt, Akt_pS473, p53
✓ RPPA technology
✓ Key pathways: PI3K/Akt, MAPK, mTOR
```

### Clinical (100)

```
✓ 9 real: primary_site, gender, age...
○ 91 generic: clinical_feat_9-99
```

### Methylation (100)

```
✓ CpG probe IDs: cg00000029...
✓ Beta values (0-1)
✓ 397K probes available → top 100
```

### Variants (100)

```
✓ Gene_MutationType format
✓ TTN_Silent, TP53_p.R273H...
✓ Top recurring mutations
```

### Metabolites (3)

```
○ METABOLITE_000-002
○ Generic names only
```

### Microbiome (1,000)

```
○ OTU_0000-0999
○ Need taxonomy mapping
```

---

## 📁 File Locations

```
augmented_pickle_data/feature_mappings.json
src/augmented_pickle_data/feature_mappings.json
data/data_lists/new_augmented_pickle_data/feature_mappings.json
```

## 🎯 Key Mappings

- `gene_to_idx`: Gene symbol → Index
- `idx_to_gene`: Index → Gene symbol
- `protein_list`: All protein names
- `methylation_list`: CpG probe IDs

## 🧪 Cancer Types

- AA: Adrenocortical
- BH: Breast
- BP: Unknown
- BR: Brain
- CV: Cervical
- DD: Unknown

## ✅ Status

- **Have real names**: Genes, Proteins, Clinical (partial), Methylation, Variants
- **Generic names**: Metabolites (3), Microbiome (1000), Clinical (91/100)

---

_TCGA Multi-omics Dataset | Knowledge Graph Enhanced_

{
"$schema": "http://json-schema.org/draft-07/schema#",
"title": "Multi-omics Feature Mappings Schema",
"description": "Schema for TCGA multi-omics feature mappings",
"type": "object",
"required": [
"gene_list", "gene_to_idx", "idx_to_gene", "total_genes",
"protein_list", "protein_to_idx", "idx_to_protein", "total_proteins",
"clinical_list", "clinical_to_idx", "idx_to_clinical", "total_clinical",
"methylation_list", "methylation_to_idx", "idx_to_methylation", "total_methylations",
"variant_list", "variant_to_idx", "idx_to_variant", "total_variants",
"metabolite_list", "metabolite_to_idx", "idx_to_metabolite", "total_metabolites",
"microbiome_list", "microbiome_to_idx", "idx_to_microbiome", "total_microbiomes"
],
"properties": {
"gene_list": {
"type": "array",
"items": {"type": "string"},
"minItems": 5000,
"maxItems": 5000,
"description": "List of gene symbols in order",
"examples": ["5S_rRNA", "A1BG", "TP53"]
},
"gene_to_idx": {
"type": "object",
"additionalProperties": {"type": "integer"},
"description": "Mapping from gene symbol to index"
},
"idx_to_gene": {
"type": "object",
"additionalProperties": {"type": "string"},
"description": "Mapping from index (as string) to gene symbol"
},
"total_genes": {
"type": "integer",
"const": 5000
},

    "protein_list": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 226,
      "maxItems": 226,
      "description": "List of protein names with phosphorylation states",
      "examples": ["Akt", "Akt_pS473", "p53"]
    },
    "protein_to_idx": {
      "type": "object",
      "additionalProperties": {"type": "integer"}
    },
    "idx_to_protein": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "total_proteins": {
      "type": "integer",
      "const": 226
    },

    "clinical_list": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 100,
      "maxItems": 100,
      "description": "Mix of real clinical features and generic names"
    },
    "clinical_to_idx": {
      "type": "object",
      "additionalProperties": {"type": "integer"}
    },
    "idx_to_clinical": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "total_clinical": {
      "type": "integer",
      "const": 100
    },

    "methylation_list": {
      "type": "array",
      "items": {
        "type": "string",
        "pattern": "^cg[0-9]{8}$|^methylation_feat_[0-9]+$"
      },
      "minItems": 100,
      "maxItems": 100,
      "description": "CpG probe IDs"
    },
    "methylation_to_idx": {
      "type": "object",
      "additionalProperties": {"type": "integer"}
    },
    "idx_to_methylation": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "total_methylations": {
      "type": "integer",
      "const": 100
    },

    "variant_list": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 100,
      "maxItems": 100,
      "description": "Gene_MutationType format",
      "examples": ["TTN_Silent", "TP53_p.R273H"]
    },
    "variant_to_idx": {
      "type": "object",
      "additionalProperties": {"type": "integer"}
    },
    "idx_to_variant": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "total_variants": {
      "type": "integer",
      "const": 100
    },

    "metabolite_list": {
      "type": "array",
      "items": {
        "type": "string",
        "pattern": "^METABOLITE_[0-9]{3}$"
      },
      "minItems": 3,
      "maxItems": 3
    },
    "metabolite_to_idx": {
      "type": "object",
      "additionalProperties": {"type": "integer"}
    },
    "idx_to_metabolite": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "total_metabolites": {
      "type": "integer",
      "const": 3
    },

    "microbiome_list": {
      "type": "array",
      "items": {
        "type": "string",
        "pattern": "^OTU_[0-9]{4}$"
      },
      "minItems": 1000,
      "maxItems": 1000
    },
    "microbiome_to_idx": {
      "type": "object",
      "additionalProperties": {"type": "integer"}
    },
    "idx_to_microbiome": {
      "type": "object",
      "additionalProperties": {"type": "string"}
    },
    "total_microbiomes": {
      "type": "integer",
      "const": 1000
    }

}
}

Train data path: C:\Users\wes\Desktop\federated_mlutiomics_dataset_curation\processed_data\train\train_data.pkl
Train data loaded with keys: ['clinical', 'gene', 'protein', 'methylation', 'variant', 'metabolite', 'microbiome']

Modality: clinical
Shape: (294, 100)
First 2 samples, first 5 features:
[[-1.23096112 -0.49104193 -0.08471736 -0.491821   -0.38765089]
 [-3.13758351 -0.49104193 -0.08471736 -0.491821   -0.38765089]]

Modality: gene
Shape: (294, 5000)
First 2 samples, first 5 features:
[[-0.43062422 -0.70602703 -0.7471686   0.03851176 -0.6408339 ]
 [-0.6334504   1.0576023   0.75367606  0.59170765 -0.31100386]]

Modality: protein
Shape: (294, 226)
First 2 samples, first 5 features:
[[-0.07818771  0.20007162 -0.5209513   1.0481106  -1.0177873 ]
 [ 0.39167744 -0.3878341   0.2941186   0.06578811  0.2141539]]

Modality: methylation
Shape: (294, 100)
First 2 samples, first 5 features:
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]

Modality: variant
Shape: (294, 100)
First 2 samples, first 5 features:
[[-1.8180739   1.0173802  -0.23023038 -1.7081062   1.2588152 ]
 [ 0.334385   -1.4760765  -0.4756773   0.45667413 -1.1341962]]

Modality: metabolite
Shape: (294, 3)
First 2 samples, first 5 features:
[[-0.11673714 -0.37765318  0.080884  ]
 [-0.34693405  0.15566048 -0.4953787]]

Modality: microbiome
Shape: (294, 1000)
First 2 samples, first 5 features:
[[0.24442151  0.23766036  0.23379685 -0.22618017 -0.30802864]
 [-1.1052665  -1.1048888  -1.1066854  -0.778413   -0.8768598]]

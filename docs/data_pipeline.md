# Data Pipeline

## Overview

The data pipeline transforms raw PDB structure files into training-ready tensors. Three stages: download, process, batch.

## Stage 1: Download

**Script:** `peptidefold/scripts/download_data.py`

Queries RCSB PDB's search API for peptide structures matching:
- Chain length: 10-30 residues
- Resolution: < 2.5 Angstroms
- Method: X-ray crystallography
- Polymer type: protein

Downloads raw PDB/mmCIF files to `data/raw/pdb/`. Out of 930 structures downloaded, roughly 198 pass quality filters during processing.

## Stage 2: Process

**Script:** `peptidefold/scripts/process_data.py`

For each PDB file:

1. **Parse structure** using BioPython's `PDBParser` or `MMCIFParser`
2. **Extract first chain** from the biological assembly
3. **Filter residues:** only standard amino acids (20 types), skip modified or non-standard residues
4. **Extract backbone atoms:** N, CA, C, O coordinates for each residue
5. **Handle missing atoms:** skip residues where any backbone atom is missing
6. **Encode sequence:** map 3-letter amino acid codes to integer tokens (0-20)
7. **Validate:** check length (10-30 residues), check coordinates are finite, check minimum 3 residues with complete backbone
8. **Save:** write sequence (int array), coordinates (float array, shape [seq_len, 4, 3]), and mask (binary) to compressed .npz file

### Amino acid encoding

```
ALA=0, CYS=1, ASP=2, GLU=3, PHE=4, GLY=5, HIS=6, ILE=7, LYS=8, LEU=9,
MET=10, ASN=11, PRO=12, GLN=13, ARG=14, SER=15, THR=16, VAL=17, TRP=18, TYR=19, UNK=20
```

### Coordinate format

Each residue has 4 atoms, each with 3 coordinates (x, y, z in Angstroms):

```
coordinates[i] = [[N_x, N_y, N_z],
                  [CA_x, CA_y, CA_z],
                  [C_x, C_y, C_z],
                  [O_x, O_y, O_z]]
```

### Why many structures fail processing

- Modified or non-standard amino acids
- Missing backbone atoms (common in flexible regions)
- Structures with only CA atoms (no full backbone)
- Files with parsing errors
- Chains outside the 10-30 residue range after filtering

## Stage 3: Batching

**Script:** `peptidefold/scripts/batching.py`

Groups peptides by sequence length to minimize padding waste. Peptides within similar length ranges are batched together. Each batch is padded to the length of the longest peptide in that batch, with a binary mask indicating real vs. padded positions.

### Data splits

- **Train:** 138 peptides (70%)
- **Validation:** 29 peptides (15%)
- **Test:** 31 peptides (15%)

Split is random but fixed via seed for reproducibility.

### Batch format

Each batch is a dictionary:
- `sequences`: integer tensor [batch_size, max_len]
- `coordinates`: float tensor [batch_size, max_len, 4, 3]
- `masks`: binary tensor [batch_size, max_len]
- `confidences`: float tensor [batch_size, max_len]

## Dataset statistics

- Total PDB files downloaded: 930
- Successfully processed: 198
- Sequence lengths: 8-35 residues (mean ~24)
- All structures are X-ray crystallography, resolution < 2.5 Angstroms

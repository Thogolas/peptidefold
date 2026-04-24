# PeptideFold

A compact transformer model for predicting 3D peptide structures from amino acid sequences. Built as an independent undergraduate research project to explore whether a minimal architecture (~133k parameters) can learn meaningful structural signal from sequence alone.

## What this project does

PeptideFold takes a peptide sequence (10-30 amino acids) as input and predicts the 3D coordinates of backbone atoms (N, CA, C, O) for each residue.

The model uses a standard transformer encoder with multi-head self-attention to learn sequence representations, then maps those representations directly to 3D coordinates. Training uses a combination of Kabsch-aligned RMSD loss, pairwise distance preservation loss, and coordinate consistency loss.

## Architecture

```
Sequence (amino acids)
    |
Embedding (21 tokens -> 64 dims)
    |
Sinusoidal Positional Encoding
    |
Projection (64 -> 96 dims)
    |
Transformer Encoder (2 layers, 4 heads)
    |
Coordinate Predictor (96 -> 4 atoms x 3 coords)
    |
Predicted 3D Structure
```

- **Parameters:** ~133,000
- **Input:** Integer-encoded amino acid sequence (vocab size 21)
- **Output:** 3D coordinates for N, CA, C, O atoms per residue + per-residue confidence score
- **Positional encoding:** Sinusoidal (standard transformer-style)
- **Attention:** 2 layers, 4 heads, 96-dim hidden
- **Coordinate prediction:** Direct XYZ regression (no torsion angles, no iterative refinement)

## Data pipeline

1. **Download:** Query RCSB PDB for peptide structures (10-30 residues, resolution < 2.5 A, X-ray only)
2. **Parse:** Extract backbone atom coordinates using BioPython's PDB/mmCIF parsers
3. **Process:** Encode sequences to integer tokens, extract N/CA/C/O coordinates, normalize, save as .npz
4. **Split:** Train (138) / Validation (29) / Test (31) peptides
5. **Batch:** Group peptides by length to minimize padding waste

**Total dataset:** 930 PDB files downloaded, 198 successfully processed into training-ready format.

## Loss function

Four components, weighted:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Kabsch-aligned RMSD | 0.5 | Structural similarity after optimal superposition |
| Pairwise distance preservation | 0.3 | Local backbone geometry (offsets 1-4, 6, 8) |
| Coordinate consistency (Huber) | 0.1 | Basic coordinate learning |
| Confidence calibration | 0.1 | Uncertainty estimation |

The Kabsch algorithm finds the optimal rotation to align predicted and target structures before computing RMSD, which is critical for fair structure comparison since predicted structures can be arbitrarily rotated/translated.

## Evaluation metrics

- **GDT-TS:** Global Distance Test - Total Score. Percentage of CA atoms within 1, 2, 4, and 8 Angstroms of the target after Kabsch alignment. Each threshold contributes 25%.
- **TM-score:** Template Modeling Score. Length-normalized structural similarity (0-1, >0.5 indicates similar fold).
- **RMSD:** Root Mean Square Deviation of CA atom positions after alignment.

All metrics use proper Kabsch superposition for fair comparison.

## Results

The best verified GDT-TS from saved checkpoints was **~7% on the validation set**. This is modest — for context:

| Method | Typical GDT-TS |
|--------|----------------|
| Random placement | ~3-5% |
| **PeptideFold** | **~7%** |
| ESMFold | ~60-80% |
| AlphaFold2 | ~80-95% |

The model learns some structural signal from sequence (above random), but performance is far from production tools. This is expected given:

- **No evolutionary information:** No MSA or homology features. Sequence-only input.
- **No iterative refinement:** Direct coordinate regression, no structure module recycling.
- **No equivariant processing:** Standard transformer, not SE(3)-equivariant.
- **Tiny model:** 133k parameters vs. AlphaFold2's ~93M.
- **Small dataset:** 198 peptides vs. millions of protein structures.

## What I learned

This project was about understanding what goes into protein structure prediction end-to-end, not about beating AlphaFold. Key takeaways:

- **Data pipeline engineering** is the majority of the work. Parsing PDB files, handling missing atoms, normalizing coordinates, and building efficient batching took more time than the model itself.
- **Structure prediction from sequence alone is hard.** Evolutionary information (MSAs) is what gives AlphaFold most of its power. Without it, even getting above random is nontrivial.
- **Loss function design matters.** Direct coordinate MSE performs poorly because structures can be rotated/translated. Kabsch-aligned RMSD and distance preservation losses were necessary.
- **Small datasets need strong regularization.** With 138 training peptides, overfitting is the main failure mode. Dropout, weight decay, and early stopping were essential.

## Project structure

```
peptidefold/
    models/
        config.py          # ModelConfig dataclass
        model.py           # PeptideFoldModel (transformer + structure module)
    evaluation/
        metrics.py         # GDT-TS, TM-score, RMSD with Kabsch alignment
    scripts/
        download_data.py   # Download peptide PDBs from RCSB
        process_data.py    # Parse PDB files, extract coordinates
        batching.py        # Smart batching by sequence length
    config/
        model.yaml         # Model hyperparameters
        data.yaml          # Data processing settings
        training.yaml      # Training settings
    train.py               # Training loop with validation and checkpointing
    predict.py             # Inference and visualization
docs/
    architecture.md        # Detailed architecture description
    data_pipeline.md       # Data processing documentation
    evaluation.md          # Metrics and how they work
```

## Usage

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy biopython pandas pyyaml
```

### Download and process data

```bash
python -m peptidefold.scripts.download_data
python -m peptidefold.scripts.process_data
```

### Train

```bash
python -m peptidefold.train --epochs 200
```

### Predict

```bash
python -m peptidefold.predict --model results/models/best_model.pt --sequence ACDEFGHIKLMNPQRSTVWY
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- BioPython
- NumPy
- SciPy
- Pandas

## Context

Built as an independent project during undergraduate studies in Molecular Biology (with Bioinformatics and Computer Science minors) at San Jose State University. The goal was to learn how protein structure prediction works at a fundamental level by implementing a simplified version from scratch.

## License

MIT

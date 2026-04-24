# Architecture

## Overview

PeptideFold uses a transformer encoder to map amino acid sequences to 3D backbone coordinates. The architecture has three main components:

1. **Sequence Encoder** - transforms integer-encoded amino acids into contextual representations
2. **Structure Module** - maps representations to 3D coordinates
3. **Loss Functions** - structure-aware losses for training

## Sequence Encoder

### Embedding

Each amino acid is mapped to a 64-dimensional vector using a standard learned embedding table. Vocabulary size is 21 (20 standard amino acids + unknown).

### Positional Encoding

Sinusoidal positional encoding (Vaswani et al., 2017) is added to the embeddings so the model knows the order of residues. This uses fixed sine/cosine functions at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Max sequence length is 100 (peptides are 10-30 residues, so this is more than enough).

### Projection

A linear layer projects from embedding dimension (64) to the hidden dimension (96).

### Self-Attention Layers

Two transformer encoder layers, each containing:

- **Multi-head self-attention** (4 heads, 24 dims per head): Each residue attends to all other residues in the sequence. This lets the model learn pairwise relationships between positions. Key padding masks prevent attending to padding tokens.
- **Feedforward network** (96 -> 96 -> 96): Two linear layers with ReLU activation and dropout.
- **Layer normalization** after each sub-layer (post-norm architecture).
- **Residual connections** around both attention and feedforward blocks.
- **Dropout** (15% on attention, varied on feedforward) for regularization.

### Why only 2 layers?

Peptides are short (10-30 residues). With full self-attention, every residue can see every other residue in a single layer. Two layers is enough for the model to learn pairwise and higher-order relationships. More layers on this small dataset would just overfit.

## Structure Module

The structure module is deliberately simple: a two-layer MLP that maps each residue's 96-dim representation to 12 values (4 backbone atoms x 3 coordinates each).

```
hidden_dim (96) -> Linear -> ReLU -> Dropout -> Linear -> 12 values -> reshape to (4, 3)
```

Coordinates are clamped to [-30, 30] Angstroms to prevent numerical instability.

A separate confidence predictor outputs a per-residue confidence score (0-1) via a smaller MLP with sigmoid activation.

### Why direct coordinate prediction?

AlphaFold2 predicts torsion angles and reconstructs coordinates using known bond lengths/angles. This is more physically principled. PeptideFold takes the simpler approach of directly predicting XYZ coordinates because:

1. It's simpler to implement and debug
2. For short peptides, the search space is smaller
3. The loss functions (especially distance preservation) provide implicit geometric constraints

The tradeoff is that the model can predict physically unrealistic bond lengths or angles. A more sophisticated version would predict torsion angles instead.

## Loss Functions

### 1. Kabsch-Aligned RMSD (weight: 0.5)

The most important loss. For each sample in the batch:

1. Extract CA atom coordinates from predicted and target structures
2. Center both sets of coordinates (subtract mean)
3. Compute the cross-covariance matrix H = P^T Q
4. SVD decomposition: H = U S V^T
5. Optimal rotation: R = V U^T (with reflection correction if det(R) < 0)
6. Apply rotation to predicted coordinates
7. Compute RMSD: sqrt(mean((P_aligned - Q)^2))

This is critical because without alignment, two identical structures in different orientations would have large RMSD.

### 2. Pairwise Distance Preservation (weight: 0.3)

Computes distances between CA atoms at offsets of 1, 2, 3, 4 (and 6, 8 for longer peptides). The loss is the relative error between predicted and target distances:

```
loss = |d_pred - d_target| / (d_target + 1.0)
```

Closer residue pairs are weighted more heavily (weight = 1/offset). This encourages the model to get local backbone geometry right.

### 3. Coordinate Consistency (weight: 0.1)

Huber loss (delta=2.0) on raw CA coordinates. A basic regression loss that provides gradient signal early in training before the alignment loss becomes useful.

### 4. Confidence Calibration (weight: 0.1)

Smooth L1 loss between predicted and target confidence scores. Encourages the model to learn which predictions are reliable.

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| Embedding (21 x 64) | 1,344 |
| Positional encoding | 0 (fixed) |
| Projection (64 -> 96) | 6,240 |
| Attention layer 1 | ~37,000 |
| Attention layer 2 | ~37,000 |
| Layer norm | 192 |
| Coordinate predictor | ~10,000 |
| Confidence predictor | ~2,500 |
| **Total** | **~133,000** |

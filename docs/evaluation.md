# Evaluation Metrics

## GDT-TS (Global Distance Test - Total Score)

The primary evaluation metric for protein structure prediction. Measures what percentage of predicted CA atom positions are close to the target after optimal superposition.

### How it works

1. Align predicted and target structures using the Kabsch algorithm (see below)
2. Compute the distance between each predicted and target CA atom
3. Count the fraction of atoms within each distance threshold
4. Average the four fractions

```
GDT-TS = 25 * (f_1A + f_2A + f_4A + f_8A)
```

Where `f_xA` is the fraction of CA atoms within x Angstroms of the target.

### Interpretation

| GDT-TS | Meaning |
|--------|---------|
| 0-5% | Random / no structural similarity |
| 5-15% | Slight structural signal |
| 15-30% | Weak fold recognition |
| 30-50% | Partial fold prediction |
| 50-70% | Good fold prediction |
| 70-90% | High-quality prediction |
| 90-100% | Near-experimental accuracy |

## TM-score (Template Modeling Score)

A length-normalized measure of structural similarity. Unlike RMSD, TM-score is less sensitive to local errors because it uses a length-dependent distance scale.

```
TM-score = (1/L) * sum(1 / (1 + (d_i / d0)^2))
```

Where:
- L = target length
- d_i = distance between i-th CA pair after alignment
- d0 = 1.24 * (L - 15)^(1/3) - 1.8 (length-dependent scale factor)

### Interpretation

| TM-score | Meaning |
|----------|---------|
| < 0.17 | Random structural similarity |
| 0.17-0.5 | Same general fold family (maybe) |
| > 0.5 | Same fold |
| > 0.8 | High structural similarity |

## RMSD (Root Mean Square Deviation)

The simplest metric: average distance between corresponding atoms after alignment.

```
RMSD = sqrt(mean((P_aligned - Q)^2))
```

Lower is better. Units are Angstroms.

### Limitation

RMSD is dominated by outliers. A prediction with 90% of atoms perfectly placed but one loop in the wrong position can have a high RMSD. This is why GDT-TS and TM-score are preferred for structure prediction evaluation.

## Kabsch Algorithm

Used by all three metrics to optimally align predicted and target structures before comparison.

### Steps

1. **Center:** Subtract the centroid from both coordinate sets
2. **Cross-covariance:** Compute H = P_centered^T @ Q_centered
3. **SVD:** Decompose H = U S V^T
4. **Rotation:** R = V @ U^T
5. **Reflection check:** If det(R) < 0, flip the sign of the last column of V and recompute R
6. **Apply:** Rotate the predicted coordinates by R

This finds the rotation that minimizes the sum of squared distances between corresponding points. It's the standard method for structural superposition in computational biology.

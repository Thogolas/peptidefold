#!/usr/bin/env python3
"""
PeptideFold: Evaluation Metrics
Implementation of GDT-TS, TM-score, and RMSD as specified in PRD
"""

import torch
import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def calculate_rmsd(pred_coords: torch.Tensor, target_coords: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Deviation (RMSD) between predicted and target coordinates
    
    Args:
        pred_coords: [seq_len, 4, 3] predicted coordinates
        target_coords: [seq_len, 4, 3] target coordinates  
        mask: [seq_len] valid residue mask
    
    Returns:
        float: RMSD in Angstroms
    """
    # Use CA coordinates (index 1)
    pred_ca = pred_coords[:, 1, :]  # [seq_len, 3]
    target_ca = target_coords[:, 1, :]
    
    # Apply mask
    valid_indices = mask.bool()
    if valid_indices.sum() == 0:
        return float('inf')
    
    pred_valid = pred_ca[valid_indices]
    target_valid = target_ca[valid_indices]
    
    # Remove NaN/inf values
    finite_mask = torch.isfinite(pred_valid).all(dim=1) & torch.isfinite(target_valid).all(dim=1)
    if finite_mask.sum() == 0:
        return float('inf')
    
    pred_clean = pred_valid[finite_mask]
    target_clean = target_valid[finite_mask]
    
    # Calculate RMSD
    mse = torch.mean((pred_clean - target_clean) ** 2)
    rmsd = torch.sqrt(mse).item()
    
    return rmsd


def calculate_gdt_ts(pred_coords: torch.Tensor, target_coords: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Calculate Global Distance Test - Total Score (GDT-TS)
    
    GDT-TS measures the percentage of residues under distance thresholds after optimal superposition.
    Uses thresholds: 1Å, 2Å, 4Å, 8Å
    
    Args:
        pred_coords: [seq_len, 4, 3] predicted coordinates
        target_coords: [seq_len, 4, 3] target coordinates
        mask: [seq_len] valid residue mask
    
    Returns:
        float: GDT-TS score (0-100%)
    """
    # Use CA coordinates
    pred_ca = pred_coords[:, 1, :].cpu().numpy()  # [seq_len, 3]
    target_ca = target_coords[:, 1, :].cpu().numpy()
    
    # Apply mask
    valid_indices = mask.bool().cpu().numpy()
    if valid_indices.sum() == 0:
        return 0.0
    
    pred_valid = pred_ca[valid_indices]
    target_valid = target_ca[valid_indices]
    
    # Remove NaN/inf values
    finite_mask = np.isfinite(pred_valid).all(axis=1) & np.isfinite(target_valid).all(axis=1)
    if finite_mask.sum() == 0:
        return 0.0
    
    pred_clean = pred_valid[finite_mask]
    target_clean = target_valid[finite_mask]
    
    if len(pred_clean) < 3:  # Need at least 3 points for superposition
        return 0.0
    
    # Optimal superposition using Kabsch algorithm
    pred_aligned, target_aligned = kabsch_alignment(pred_clean, target_clean)
    
    # Calculate distances after alignment
    distances = np.linalg.norm(pred_aligned - target_aligned, axis=1)
    
    # GDT-TS thresholds
    thresholds = [1.0, 2.0, 4.0, 8.0]
    n_residues = len(distances)
    
    gdt_ts_score = 0.0
    for threshold in thresholds:
        fraction_under_threshold = np.sum(distances < threshold) / n_residues
        gdt_ts_score += fraction_under_threshold * 25.0  # Each threshold contributes 25%
    
    return gdt_ts_score


def calculate_tm_score(pred_coords: torch.Tensor, target_coords: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Calculate Template Modeling Score (TM-score)
    
    TM-score measures structural similarity with length normalization.
    Score ranges from 0 to 1, where >0.5 indicates similar fold.
    
    Args:
        pred_coords: [seq_len, 4, 3] predicted coordinates
        target_coords: [seq_len, 4, 3] target coordinates  
        mask: [seq_len] valid residue mask
    
    Returns:
        float: TM-score (0-1)
    """
    # Use CA coordinates
    pred_ca = pred_coords[:, 1, :].cpu().numpy()
    target_ca = target_coords[:, 1, :].cpu().numpy()
    
    # Apply mask
    valid_indices = mask.bool().cpu().numpy()
    if valid_indices.sum() == 0:
        return 0.0
    
    pred_valid = pred_ca[valid_indices]
    target_valid = target_ca[valid_indices]
    
    # Remove NaN/inf values
    finite_mask = np.isfinite(pred_valid).all(axis=1) & np.isfinite(target_valid).all(axis=1)
    if finite_mask.sum() == 0:
        return 0.0
    
    pred_clean = pred_valid[finite_mask]
    target_clean = target_valid[finite_mask]
    
    if len(pred_clean) < 3:
        return 0.0
    
    # Optimal superposition
    pred_aligned, target_aligned = kabsch_alignment(pred_clean, target_clean)
    
    # Calculate TM-score
    n_target = len(target_clean)
    d0 = 1.24 * (n_target - 15) ** (1.0/3) - 1.8  # Length-dependent scale
    d0 = max(d0, 0.5)  # Minimum d0
    
    distances = np.linalg.norm(pred_aligned - target_aligned, axis=1)
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / n_target
    
    return tm_score


def kabsch_alignment(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm for optimal superposition of two point sets
    
    Args:
        P: [n_points, 3] predicted coordinates
        Q: [n_points, 3] target coordinates
    
    Returns:
        Tuple of aligned P and Q coordinates
    """
    # Center the coordinates
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    
    # Compute the covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation to P
    P_aligned = P_centered @ R.T + np.mean(Q, axis=0)
    
    return P_aligned, Q


def evaluate_structure_prediction(predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Comprehensive evaluation of structure prediction
    
    Args:
        predictions: Model predictions with 'coordinates' key
        targets: Ground truth with 'coordinates' and 'masks' keys
    
    Returns:
        Dictionary with RMSD, GDT-TS, and TM-score
    """
    pred_coords = predictions['coordinates'][0]  # Take first sample
    target_coords = targets['coordinates'][0]
    mask = targets['masks'][0]
    
    # Calculate metrics
    rmsd = calculate_rmsd(pred_coords, target_coords, mask)
    gdt_ts = calculate_gdt_ts(pred_coords, target_coords, mask)
    tm_score = calculate_tm_score(pred_coords, target_coords, mask)
    
    return {
        'rmsd': rmsd,
        'gdt_ts': gdt_ts,
        'tm_score': tm_score
    }


def batch_evaluate(predictions: Dict[str, torch.Tensor], 
                  targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Evaluate a batch of predictions
    
    Returns:
        Dictionary with averaged metrics
    """
    batch_size = predictions['coordinates'].shape[0]
    
    rmsds = []
    gdt_ts_scores = []
    tm_scores = []
    
    for i in range(batch_size):
        pred_coords = predictions['coordinates'][i]
        target_coords = targets['coordinates'][i]
        mask = targets['masks'][i]
        
        rmsd = calculate_rmsd(pred_coords, target_coords, mask)
        gdt_ts = calculate_gdt_ts(pred_coords, target_coords, mask)
        tm_score = calculate_tm_score(pred_coords, target_coords, mask)
        
        if rmsd != float('inf'):
            rmsds.append(rmsd)
        if gdt_ts > 0:
            gdt_ts_scores.append(gdt_ts)
        if tm_score > 0:
            tm_scores.append(tm_score)
    
    return {
        'rmsd': np.mean(rmsds) if rmsds else float('inf'),
        'gdt_ts': np.mean(gdt_ts_scores) if gdt_ts_scores else 0.0,
        'tm_score': np.mean(tm_scores) if tm_scores else 0.0,
        'n_evaluated': len(rmsds)
    }
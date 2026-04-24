#!/usr/bin/env python3
"""
PeptideFold Model
Optimized architecture for short peptides (10-30 residues)

Key design principles:
1. Smaller model capacity (peptides simpler than full proteins)
2. Faster training (shorter sequences = less computation)
3. Higher target performance (25-30% GDT-TS achievable)
4. Direct structural optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .config import ModelConfig


class PeptideFoldModel(nn.Module):
    """
    Specialized model for peptide structure prediction
    
    Optimizations for peptides:
    - Smaller model (peptides simpler than proteins)
    - Direct GDT-TS optimization from day 1
    - Efficient for 10-30 residue sequences
    - Target: 25-30% GDT-TS performance
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Peptide-optimized components
        self.sequence_encoder = PeptideSequenceEncoder(config)
        self.structure_module = PeptideStructureModule(config)
        
        # GDT-TS optimized loss weights
        self.alignment_rmsd_weight = 0.5      # Direct alignment optimization (key!)
        self.distance_preservation_weight = 0.3  # Local structure preservation
        self.coordinate_consistency_weight = 0.1  # Basic coordinate learning
        self.confidence_weight = 0.1              # Uncertainty estimation
        
        # Numerical stability
        self.numerical_eps = 1e-8
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass optimized for peptides"""
        sequences = batch['sequences']
        masks = batch['masks']
        
        # Encode peptide sequences
        sequence_repr = self.sequence_encoder(sequences, masks)
        
        # Predict structure 
        structure_output = self.structure_module(sequence_repr, masks)
        
        return structure_output
    
    def calculate_peptide_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Loss function specifically optimized for peptide GDT-TS performance
        Optimized for peptide GDT-TS performance
        """
        pred_coords = predictions['coordinates']
        target_coords = targets['coordinates']
        masks = targets['masks']
        
        # Initialize with numerical stability
        total_loss = torch.tensor(0.0, device=pred_coords.device, requires_grad=True)
        
        # 1. ALIGNMENT RMSD LOSS (50% - most critical for GDT-TS)
        alignment_loss = self.calculate_peptide_alignment_loss(pred_coords, target_coords, masks)
        if isinstance(alignment_loss, torch.Tensor) and torch.isfinite(alignment_loss):
            total_loss = total_loss + self.alignment_rmsd_weight * alignment_loss
        
        # 2. DISTANCE PRESERVATION LOSS (30% - local structure)
        distance_loss = self.calculate_peptide_distance_loss(pred_coords, target_coords, masks)
        if isinstance(distance_loss, torch.Tensor) and torch.isfinite(distance_loss):
            total_loss = total_loss + self.distance_preservation_weight * distance_loss
        
        # 3. COORDINATE CONSISTENCY LOSS (10% - basic learning)
        coord_loss = self.calculate_coordinate_loss(pred_coords, target_coords, masks)
        if isinstance(coord_loss, torch.Tensor) and torch.isfinite(coord_loss):
            total_loss = total_loss + self.coordinate_consistency_weight * coord_loss
        
        # 4. CONFIDENCE LOSS (10% - uncertainty)
        if 'confidence_score' in predictions and 'confidences' in targets:
            conf_loss = self.calculate_confidence_loss(predictions, targets, masks)
            if isinstance(conf_loss, torch.Tensor) and torch.isfinite(conf_loss):
                total_loss = total_loss + self.confidence_weight * conf_loss
        
        # Ensure finite loss
        if not torch.isfinite(total_loss) or total_loss <= 0:
            total_loss = torch.tensor(1.0, device=pred_coords.device, requires_grad=True)
        
        return total_loss
    
    def calculate_peptide_alignment_loss(self, pred_coords: torch.Tensor, target_coords: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Peptide-optimized alignment loss using stable Kabsch algorithm
        Optimized for short sequences (10-30 residues)
        """
        batch_size = pred_coords.shape[0]
        total_alignment_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            mask = masks[b]
            valid_indices = mask.bool()
            
            if valid_indices.sum() < 3:  # Need minimum 3 points for alignment
                continue
            
            # Extract CA atoms (most important for structure alignment)
            pred_ca = pred_coords[b, valid_indices, 1, :]  # CA atoms
            target_ca = target_coords[b, valid_indices, 1, :]
            
            # Check for valid coordinates
            if not (torch.isfinite(pred_ca).all() and torch.isfinite(target_ca).all()):
                continue
            
            try:
                # Optimized Kabsch alignment for peptides
                aligned_rmsd = self.peptide_kabsch_rmsd(pred_ca, target_ca)
                if torch.isfinite(aligned_rmsd):
                    total_alignment_loss += aligned_rmsd
                    valid_samples += 1
            except:
                # Fallback to centered RMSD
                pred_centered = pred_ca - pred_ca.mean(dim=0)
                target_centered = target_ca - target_ca.mean(dim=0)
                fallback_rmsd = torch.sqrt(torch.mean((pred_centered - target_centered) ** 2) + self.numerical_eps)
                total_alignment_loss += fallback_rmsd
                valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=pred_coords.device, requires_grad=True)
        
        return total_alignment_loss / valid_samples
    
    def peptide_kabsch_rmsd(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable Kabsch RMSD for peptides
        Optimized for short sequences with robust error handling
        """
        # Center coordinates
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        
        # Cross-covariance matrix
        H = pred_centered.T @ target_centered
        
        try:
            # Stable SVD
            U, S, V = torch.svd(H + self.numerical_eps * torch.eye(3, device=H.device))
            
            # Optimal rotation matrix
            R = V @ U.T
            
            # Ensure proper rotation (no reflection)
            if torch.det(R) < 0:
                V_corrected = V.clone()
                V_corrected[:, -1] = -V_corrected[:, -1]  # Fixed: no in-place operation
                R = V_corrected @ U.T
            
            # Apply rotation and calculate RMSD
            pred_aligned = pred_centered @ R.T
            rmsd = torch.sqrt(torch.mean((pred_aligned - target_centered) ** 2) + self.numerical_eps)
            
            return rmsd
            
        except:
            # Robust fallback
            return torch.sqrt(torch.mean((pred_centered - target_centered) ** 2) + self.numerical_eps)
    
    def calculate_peptide_distance_loss(self, pred_coords: torch.Tensor, target_coords: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Distance preservation loss optimized for peptides
        Focus on local distances that matter for short sequences
        """
        pred_ca = pred_coords[:, :, 1, :]
        target_ca = target_coords[:, :, 1, :]
        
        batch_size, seq_len = pred_ca.shape[:2]
        total_distance_loss = 0.0
        num_distances = 0
        
        # For peptides (10-30 residues), focus on local distances
        important_offsets = [1, 2, 3, 4]  # Mostly local structure
        if seq_len > 15:  # For longer peptides, add some medium-range
            important_offsets.extend([6, 8])
        
        for offset in important_offsets:
            if seq_len <= offset:
                continue
            
            # Calculate pairwise distances
            pred_i = pred_ca[:, :-offset, :]
            pred_j = pred_ca[:, offset:, :]
            target_i = target_ca[:, :-offset, :]
            target_j = target_ca[:, offset:, :]
            
            pred_dist = torch.norm(pred_i - pred_j, dim=2) + self.numerical_eps
            target_dist = torch.norm(target_i - target_j, dim=2) + self.numerical_eps
            
            # Apply masks (ensure boolean type)
            mask_i = masks[:, :-offset].bool() if masks.dtype != torch.bool else masks[:, :-offset]
            mask_j = masks[:, offset:].bool() if masks.dtype != torch.bool else masks[:, offset:]
            valid_mask = mask_i & mask_j
            finite_mask = torch.isfinite(pred_dist) & torch.isfinite(target_dist)
            combined_mask = valid_mask & finite_mask
            
            if combined_mask.sum() > 0:
                valid_pred_dist = pred_dist[combined_mask]
                valid_target_dist = target_dist[combined_mask]
                
                # Use relative error (works well for different distance scales)
                relative_error = torch.abs(valid_pred_dist - valid_target_dist) / (valid_target_dist + 1.0)
                distance_loss = torch.mean(relative_error)
                
                # Weight by importance (closer = more important)
                weight = 1.0 / offset
                total_distance_loss += weight * distance_loss
                num_distances += weight
        
        return total_distance_loss / max(num_distances, self.numerical_eps)
    
    def calculate_coordinate_loss(self, pred_coords: torch.Tensor, target_coords: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Basic coordinate consistency loss"""
        pred_ca = pred_coords[:, :, 1, :]
        target_ca = target_coords[:, :, 1, :]
        
        valid_mask = masks.unsqueeze(-1).expand_as(pred_ca)
        finite_mask = torch.isfinite(target_ca) & torch.isfinite(pred_ca)
        combined_mask = valid_mask & finite_mask
        
        if combined_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_coords.device, requires_grad=True)
        
        valid_pred = pred_ca[combined_mask]
        valid_target = target_ca[combined_mask]
        
        # Huber loss for robustness
        return F.huber_loss(valid_pred, valid_target, delta=2.0)
    
    def calculate_confidence_loss(self, predictions: Dict, targets: Dict, masks: torch.Tensor) -> torch.Tensor:
        """Confidence calibration loss"""
        pred_conf = predictions['confidence_score']
        target_conf = targets.get('confidences', torch.ones_like(pred_conf))
        
        valid_mask = masks & torch.isfinite(target_conf) & torch.isfinite(pred_conf)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_conf.device, requires_grad=True)
        
        valid_pred = pred_conf[valid_mask]
        valid_target = target_conf[valid_mask]
        
        return F.smooth_l1_loss(valid_pred, valid_target)


class PeptideSequenceEncoder(nn.Module):
    """
    Sequence encoder optimized for peptides (10-30 residues)
    Smaller and more efficient than full protein encoders
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Smaller embedding for peptides
        self.embedding = nn.Embedding(config.vocab_size, config.sequence_dim)
        self.pos_encoding = PeptidePositionalEncoding(config.sequence_dim)
        
        # Smaller hidden dimension (peptides are simpler)
        hidden_dim = min(config.hidden_dim, 96)  
        self.projection = nn.Linear(config.sequence_dim, hidden_dim)
        
        # Fewer attention layers (peptides don't need deep reasoning)
        self.layers = nn.ModuleList([
            PeptideAttentionLayer(hidden_dim, config.num_heads, dropout=0.15)
            for _ in range(2)  # Only 2 layers for efficiency
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequences)
        x = self.pos_encoding(x)
        x = self.projection(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, masks)
        
        x = self.layer_norm(x)
        return x


class PeptideAttentionLayer(nn.Module):
    """Attention layer optimized for peptides"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.15):
        super().__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        num_heads = min(num_heads, hidden_dim // 16)  # At least 16 dims per head
        if hidden_dim % num_heads != 0:
            num_heads = hidden_dim // 16
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Smaller feedforward for peptides
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # Self-attention
        mask_bool = masks.bool() if masks.dtype != torch.bool else masks
        attended, _ = self.attention(x, x, x, key_padding_mask=~mask_bool)
        x = self.layer_norm1(x + attended)
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.layer_norm2(x + ff_out)
        
        return x


class PeptideStructureModule(nn.Module):
    """
    Structure prediction module for peptides
    Simpler than full protein structure modules
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        hidden_dim = min(config.hidden_dim, 96)
        
        # Simple coordinate prediction
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, config.num_backbone_atoms * 3)
        )
        
        # Simple confidence prediction
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence_repr: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, hidden_dim = sequence_repr.shape
        
        # Predict coordinates
        coord_logits = self.coord_predictor(sequence_repr)
        coordinates = coord_logits.view(batch_size, seq_len, self.config.num_backbone_atoms, 3)
        
        # Reasonable coordinate bounds for peptides
        coordinates = torch.clamp(coordinates, -30.0, 30.0)
        
        # Predict confidence
        confidence_score = self.confidence_predictor(sequence_repr).squeeze(-1)
        
        return {
            'coordinates': coordinates,
            'confidence_score': confidence_score
        }


class PeptidePositionalEncoding(nn.Module):
    """Positional encoding optimized for short peptides"""
    
    def __init__(self, d_model: int, max_len: int = 100):  # Shorter max_len for peptides
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


def create_peptide_fold_model(config_dict: Dict = None) -> PeptideFoldModel:
    """Create PeptideFold model optimized for 25-30% GDT-TS on peptides"""
    if config_dict is None:
        config_dict = {}
    
    # Optimized configuration for peptides
    default_config = {
        'vocab_size': 21,
        'sequence_dim': 64,       # Smaller for peptides
        'hidden_dim': 96,         # Smaller for efficiency  
        'num_heads': 4,           # Fewer heads
        'num_layers': 2,          # Fewer layers
        'dropout': 0.15,          # Moderate dropout
        'coordinate_dim': 3,
        'num_backbone_atoms': 4,
        'max_sequence_length': 40,  # Peptide-appropriate
        'confidence_bins': 20
    }
    
    default_config.update(config_dict)
    config = ModelConfig(**default_config)
    
    return PeptideFoldModel(config)
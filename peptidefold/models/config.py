#!/usr/bin/env python3
"""
PeptideFold: Model Configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 21
    sequence_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    coordinate_dim: int = 3
    num_backbone_atoms: int = 4
    max_sequence_length: int = 150
    confidence_bins: int = 50

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.output(attended)
        return self.layer_norm(x + output)

class SequenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.sequence_dim)
        self.projection = nn.Linear(config.sequence_dim, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            MultiHeadAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Convert sequences to indices if they're strings
        if sequences.dtype != torch.long:
            sequences = self._encode_sequences(sequences)
        
        x = self.embedding(sequences)
        x = self.projection(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
    def _encode_sequences(self, sequences) -> torch.Tensor:
        # Simple encoding for demonstration
        aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
        }
        
        if isinstance(sequences, list):
            max_len = max(len(seq) for seq in sequences)
            encoded = torch.zeros(len(sequences), max_len, dtype=torch.long)
            
            for i, seq in enumerate(sequences):
                for j, aa in enumerate(seq):
                    encoded[i, j] = aa_to_idx.get(aa, aa_to_idx['X'])
            
            return encoded
        
        return sequences

class StructureModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.input_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.structure_layers = nn.ModuleList([
            MultiHeadAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(3)
        ])
        
        self.coord_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_backbone_atoms * config.coordinate_dim)
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.confidence_bins)
        )
        
    def forward(self, sequence_repr: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = sequence_repr.shape
        
        x = self.input_projection(sequence_repr)
        
        for layer in self.structure_layers:
            x = layer(x, mask)
        
        # Predict coordinates
        coord_logits = self.coord_predictor(x)
        coordinates = coord_logits.view(batch_size, seq_len, self.config.num_backbone_atoms, 3)
        
        # Predict confidence
        confidence_logits = self.confidence_predictor(x)
        confidence = F.softmax(confidence_logits, dim=-1)
        
        # Convert to single score
        confidence_bins = torch.linspace(0, 1, self.config.confidence_bins, device=x.device)
        confidence_score = torch.sum(confidence * confidence_bins, dim=-1)
        
        return {
            'coordinates': coordinates,
            'confidence_logits': confidence_logits,
            'confidence_score': confidence_score
        }

class BaseFoldModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.sequence_encoder = SequenceEncoder(config)
        self.structure_module = StructureModule(config)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sequences = batch['sequences']  # Already encoded integers from collate_fn
        masks = batch['masks']
        
        # Encode sequences
        sequence_repr = self.sequence_encoder(sequences, masks)
        
        # Predict structure
        structure_output = self.structure_module(sequence_repr, masks)
        
        return structure_output
    
    def predict_structure(self, sequences: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = {'sequences': sequences, 'masks': masks}
        return self.forward(batch)

def create_base_model(config_dict: Dict = None) -> BaseFoldModel:
    if config_dict is None:
        config_dict = {}
    
    default_config = {
        'vocab_size': 21,
        'sequence_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'coordinate_dim': 3,
        'num_backbone_atoms': 4,
        'max_sequence_length': 150,
        'confidence_bins': 50
    }
    
    default_config.update(config_dict)
    config = ModelConfig(**default_config)
    
    return BaseFoldModel(config)

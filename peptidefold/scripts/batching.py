#!/usr/bin/env python3
"""
Peptide Smart Batching System
Optimized for short peptides (10-30 residues) with larger dataset (500+ samples)

- Smaller length range (10-30 vs 80-350 residues)
- Larger batch sizes possible due to shorter sequences
- Less padding waste due to more uniform lengths
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import List, Dict, Tuple
import json


class PeptideDataset(Dataset):
    """Dataset for peptide structures (10-30 residues)"""
    
    def __init__(self, pdb_ids: List[str], data_dir: str = "data/processed"):
        self.pdb_ids = pdb_ids
        self.data_dir = Path(data_dir)
        
        # Load and cache basic info for efficient batching
        self.peptide_info = {}
        self.lengths = []
        
        print(f"📊 Loading peptide dataset info for {len(pdb_ids)} peptides...")
        
        for pdb_id in pdb_ids:
            data_file = self.data_dir / f"{pdb_id}.npz"
            if data_file.exists():
                try:
                    data = np.load(data_file)
                    length = len(data['sequence'])
                    
                    self.peptide_info[pdb_id] = {
                        'length': length,
                        'file_path': data_file
                    }
                    self.lengths.append(length)
                    
                except Exception as e:
                    print(f"  ⚠️  Failed to load {pdb_id}: {e}")
                    continue
        
        # Filter out failed loads
        self.pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id in self.peptide_info]
        self.lengths = [self.peptide_info[pdb_id]['length'] for pdb_id in self.pdb_ids]
        
        print(f"✅ Loaded {len(self.pdb_ids)} peptides")
        print(f"   Length range: {min(self.lengths)}-{max(self.lengths)} residues")
        print(f"   Mean length: {np.mean(self.lengths):.1f} residues")
    
    def __len__(self):
        return len(self.pdb_ids)
    
    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        data_file = self.peptide_info[pdb_id]['file_path']
        
        try:
            data = np.load(data_file)
            
            return {
                'pdb_id': pdb_id,
                'sequence': torch.from_numpy(data['sequence']).long(),
                'coordinates': torch.from_numpy(data['coordinates']).float(),
                'mask': torch.from_numpy(data['masks']).bool(),  # Use 'masks' field
                'length': self.peptide_info[pdb_id]['length']
            }
            
        except Exception as e:
            print(f"Error loading {pdb_id}: {e}")
            # Return dummy data to prevent crash
            dummy_length = 15  # Average peptide length
            return {
                'pdb_id': pdb_id,
                'sequence': torch.zeros(dummy_length, dtype=torch.long),
                'coordinates': torch.zeros(dummy_length, 4, 3, dtype=torch.float),
                'mask': torch.ones(dummy_length, dtype=torch.bool),
                'length': dummy_length
            }


class PeptideLengthBasedBatchSampler(Sampler):
    """
    Batch sampler that groups peptides by similar lengths
    Optimized for peptides with smaller length variation
    """
    
    def __init__(self, dataset: PeptideDataset, batch_size: int = 16, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create length-based groups for peptides
        self.length_groups = self._create_length_groups()
        
        print(f"📊 Peptide batching groups:")
        for group_id, indices in self.length_groups.items():
            lengths_in_group = [dataset.lengths[i] for i in indices]
            print(f"   Group {group_id}: {len(indices)} peptides, "
                  f"lengths {min(lengths_in_group)}-{max(lengths_in_group)}")
    
    def _create_length_groups(self) -> Dict[int, List[int]]:
        """Create groups based on peptide length with smaller bins"""
        
        length_groups = {}
        
        # For peptides (10-30 residues), use smaller bins
        # Bin size of 5 residues should work well
        for idx, length in enumerate(self.dataset.lengths):
            # Group by 5-residue bins: 10-14, 15-19, 20-24, 25-30
            if 10 <= length <= 14:
                group_id = 0
            elif 15 <= length <= 19:
                group_id = 1
            elif 20 <= length <= 24:
                group_id = 2
            elif 25 <= length <= 30:
                group_id = 3
            else:
                # Fallback for outliers
                group_id = length // 5
            
            if group_id not in length_groups:
                length_groups[group_id] = []
            length_groups[group_id].append(idx)
        
        return length_groups
    
    def __iter__(self):
        # Create batches from each group
        all_batches = []
        
        for group_id, indices in self.length_groups.items():
            # Shuffle indices within group
            np.random.shuffle(indices)
            
            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle all batches
        np.random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        total_batches = 0
        for indices in self.length_groups.values():
            group_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                group_batches += 1
            total_batches += group_batches
        return total_batches


def peptide_collate_fn(batch):
    """
    Collate function for peptide batches with minimal padding
    """
    
    # Find maximum length in batch
    max_length = max(item['length'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    sequences = torch.zeros(batch_size, max_length, dtype=torch.long)
    coordinates = torch.zeros(batch_size, max_length, 4, 3, dtype=torch.float)
    masks = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    pdb_ids = []
    lengths = []
    
    for i, item in enumerate(batch):
        seq_len = item['length']
        
        # Copy data
        sequences[i, :seq_len] = item['sequence']
        coordinates[i, :seq_len] = item['coordinates'] 
        masks[i, :seq_len] = item['mask']
        
        pdb_ids.append(item['pdb_id'])
        lengths.append(seq_len)
    
    return {
        'pdb_ids': pdb_ids,
        'sequences': sequences,
        'coordinates': coordinates,
        'masks': masks,
        'lengths': lengths,
        'max_length': max_length
    }


def load_peptide_splits(splits_dir: str = "data/splits") -> Dict[str, List[str]]:
    """Load train/validation/test splits for peptides"""
    
    splits_dir = Path(splits_dir)
    splits = {}
    
    # Try robust splits first, fallback to original
    for split_name in ['train', 'validation', 'test']:
        # Try robust version first
        split_file = splits_dir / f"{split_name}_robust.csv"
        if not split_file.exists():
            split_file = splits_dir / f"{split_name}.csv"
        
        if split_file.exists():
            df = pd.read_csv(split_file)
            splits[split_name] = df['pdb_id'].tolist()
            print(f"📊 Loaded {split_name}: {len(splits[split_name])} peptides")
        else:
            print(f"❌ Split file not found: {split_file}")
            splits[split_name] = []
    
    return splits


def create_peptide_data_loaders(batch_size: int = 16, 
                              num_workers: int = 4,
                              data_dir: str = "data/processed",
                              splits_dir: str = "data/splits") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimized data loaders for peptide training
    
    Much larger batch sizes possible due to shorter sequences (10-30 vs 100+ residues)
    """
    
    print("🧬 CREATING PEPTIDE DATA LOADERS")
    print("=" * 50)
    print(f"Expected peptide lengths: 10-30 residues")
    print()
    
    # Load splits
    splits = load_peptide_splits(splits_dir)
    
    if not all(splits.values()):
        print("❌ Missing splits! Run process_peptide_data.py first.")
        return None, None, None
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = PeptideDataset(splits['train'], data_dir)
    val_dataset = PeptideDataset(splits['validation'], data_dir)
    test_dataset = PeptideDataset(splits['test'], data_dir)
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return None, None, None
    
    # Create batch samplers (don't drop last for small datasets)
    train_sampler = PeptideLengthBasedBatchSampler(train_dataset, batch_size=batch_size, drop_last=False)
    val_sampler = PeptideLengthBasedBatchSampler(val_dataset, batch_size=batch_size, drop_last=False)
    test_sampler = PeptideLengthBasedBatchSampler(test_dataset, batch_size=batch_size, drop_last=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=peptide_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=peptide_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=peptide_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ PEPTIDE DATA LOADERS CREATED")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} peptides)")
    print(f"   Validation: {len(val_loader)} batches ({len(val_dataset)} peptides)")
    print(f"   Test: {len(test_loader)} batches ({len(test_dataset)} peptides)")
    print(f"   Batch size: {batch_size} peptides per batch")
    
    return train_loader, val_loader, test_loader


def analyze_peptide_batching_efficiency(data_loader: DataLoader, max_batches: int = 10):
    """Analyze batching efficiency for peptides"""
    
    print(f"\n📊 PEPTIDE BATCHING EFFICIENCY ANALYSIS")
    print("=" * 50)
    
    total_residues = 0
    total_padded_residues = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        
        lengths = batch['lengths']
        max_length = batch['max_length']
        batch_size = len(lengths)
        
        # Calculate efficiency
        actual_residues = sum(lengths)
        padded_residues = batch_size * max_length
        efficiency = actual_residues / padded_residues
        
        total_residues += actual_residues
        total_padded_residues += padded_residues
        batch_count += 1
        
        print(f"  Batch {batch_idx}: lengths {min(lengths)}-{max(lengths)}, "
              f"efficiency {efficiency:.1%}")
    
    overall_efficiency = total_residues / total_padded_residues
    print(f"\nOverall efficiency: {overall_efficiency:.1%}")
    print(f"Padding waste: {(1-overall_efficiency):.1%}")


if __name__ == "__main__":
    print("🧬 PeptideFold: Smart Batching System")
    print("Optimized for peptides (10-30 residues)")
    print()
    
    # Create data loaders with larger batch size
    train_loader, val_loader, test_loader = create_peptide_data_loaders(batch_size=16)
    
    if train_loader is not None:
        # Test efficiency
        print(f"\nTesting batching efficiency...")
        analyze_peptide_batching_efficiency(train_loader, max_batches=5)
        
        print(f"\n🎯 PEPTIDEFOLD BATCHING READY!")
        print(f"   Larger batch sizes possible (16 vs 4)")
        print(f"   Less memory usage per peptide")
        print(f"   Ready for fast training!")
    else:
        print("❌ Failed to create peptide data loaders")
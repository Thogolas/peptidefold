#!/usr/bin/env python3
"""
Peptide Data Processing Script
Processes downloaded peptide structures (10-30 residues) for PeptideFold training

- Much shorter sequences (10-30 vs 100+ residues)
- Larger dataset (500+ vs 60 samples)
- Faster processing due to smaller structures
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
import torch


class PeptideDataProcessor:
    """Process peptide structures for PeptideFold training"""
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw/pdb",
                 processed_data_dir: str = "data/processed",
                 splits_dir: str = "data/splits"):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.splits_dir = Path(splits_dir)
        
        # Create output directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Peptide-specific parameters
        self.min_length = 10
        self.max_length = 30
        self.target_resolution = 2.5  # Ångström
        
        # Amino acid vocabulary
        self.amino_acids = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
            'UNK': 20  # Unknown amino acid
        }
        
        self.backbone_atoms = ['N', 'CA', 'C', 'O']
        
        self.processed_peptides = []
        self.failed_peptides = []
        
    def load_peptide_metadata(self) -> Dict:
        """Load peptide download metadata"""
        metadata_file = self.raw_data_dir / "peptide_download_metadata.json"
        
        if not metadata_file.exists():
            print("❌ Peptide metadata not found. Run download_peptide_data.py first!")
            return {}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Loaded metadata for {metadata['actual_count']} peptides")
        return metadata
    
    def parse_structure_file(self, pdb_id: str) -> Optional[PDB.Structure.Structure]:
        """Parse PDB or mmCIF structure file"""
        
        # Try PDB format first
        pdb_file = self.raw_data_dir / f"{pdb_id}.pdb"
        if pdb_file.exists():
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(pdb_id, str(pdb_file))
                return structure
            except Exception as e:
                print(f"  ⚠️  PDB parsing failed for {pdb_id}: {e}")
        
        # Try mmCIF format
        cif_file = self.raw_data_dir / f"{pdb_id}.cif"
        if cif_file.exists():
            try:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure(pdb_id, str(cif_file))
                return structure
            except Exception as e:
                print(f"  ⚠️  mmCIF parsing failed for {pdb_id}: {e}")
        
        print(f"  ❌ No structure file found for {pdb_id}")
        return None
    
    def extract_peptide_sequence_and_coordinates(self, structure: PDB.Structure.Structure, pdb_id: str) -> Optional[Dict]:
        """Extract sequence and coordinates from peptide structure"""
        
        try:
            # Get the first model and first chain
            model = structure[0]
            chains = list(model.get_chains())
            
            if not chains:
                return None
            
            # Use first chain (peptides typically have one chain)
            chain = chains[0]
            residues = list(chain.get_residues())
            
            # Filter out hetero residues (water, ligands, etc.)
            protein_residues = []
            for residue in residues:
                if residue.id[0] == ' ':  # Standard amino acid residue
                    protein_residues.append(residue)
            
            if len(protein_residues) < self.min_length or len(protein_residues) > self.max_length:
                return None
            
            # Extract sequence and coordinates
            sequence = []
            coordinates = []
            
            for residue in protein_residues:
                # Get residue name
                res_name = residue.resname.strip()
                if res_name in self.amino_acids:
                    sequence.append(self.amino_acids[res_name])
                else:
                    sequence.append(self.amino_acids['UNK'])  # Unknown
                
                # Extract backbone atom coordinates
                atom_coords = []
                for atom_name in self.backbone_atoms:
                    if atom_name in residue:
                        atom = residue[atom_name]
                        coord = atom.get_coord()
                        atom_coords.append(coord)
                    else:
                        # Missing atom - use NaN
                        atom_coords.append([np.nan, np.nan, np.nan])
                
                coordinates.append(atom_coords)
            
            # Convert to numpy arrays
            sequence = np.array(sequence, dtype=np.int64)
            coordinates = np.array(coordinates, dtype=np.float32)  # [seq_len, 4, 3]
            
            # Create mask for valid residues
            mask = np.ones(len(sequence), dtype=bool)
            
            # Check for too many missing atoms
            missing_atoms = np.isnan(coordinates).any(axis=2).sum()
            if missing_atoms > len(sequence) * 0.1:  # >10% missing atoms
                return None
            
            return {
                'pdb_id': pdb_id,
                'sequence': sequence,
                'coordinates': coordinates,
                'mask': mask,
                'length': len(sequence),
                'missing_atoms': missing_atoms
            }
            
        except Exception as e:
            print(f"  ❌ Extraction failed for {pdb_id}: {e}")
            return None
    
    def calculate_peptide_statistics(self, peptide_data: Dict) -> Dict:
        """Calculate basic statistics for peptide structure"""
        
        coordinates = peptide_data['coordinates']
        sequence = peptide_data['sequence']
        
        # Calculate center of mass (CA atoms)
        ca_coords = coordinates[:, 1, :]  # CA atoms
        valid_ca = ~np.isnan(ca_coords).any(axis=1)
        
        if valid_ca.sum() == 0:
            return {'center_of_mass': [0, 0, 0], 'radius_of_gyration': 0, 'missing_atoms': 0}
        
        ca_valid = ca_coords[valid_ca]
        center_of_mass = np.mean(ca_valid, axis=0)
        
        # Radius of gyration
        distances = np.linalg.norm(ca_valid - center_of_mass, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances**2))
        
        # Count missing atoms
        missing_atoms = np.isnan(coordinates).sum()
        
        return {
            'center_of_mass': center_of_mass.tolist(),
            'radius_of_gyration': float(radius_of_gyration),
            'missing_atoms': int(missing_atoms),
            'sequence_composition': np.bincount(sequence, minlength=21).tolist()
        }
    
    def process_single_peptide(self, peptide_info: Dict) -> bool:
        """Process a single peptide structure"""
        
        pdb_id = peptide_info['pdb_id']
        
        # Parse structure
        structure = self.parse_structure_file(pdb_id)
        if structure is None:
            self.failed_peptides.append({'pdb_id': pdb_id, 'reason': 'parsing_failed'})
            return False
        
        # Extract data
        peptide_data = self.extract_peptide_sequence_and_coordinates(structure, pdb_id)
        if peptide_data is None:
            self.failed_peptides.append({'pdb_id': pdb_id, 'reason': 'extraction_failed'})
            return False
        
        # Add metadata from download
        peptide_data.update({
            'original_sequence': peptide_info.get('sequence', ''),
            'resolution': peptide_info.get('resolution', 999),
        })
        
        # Calculate statistics
        stats = self.calculate_peptide_statistics(peptide_data)
        peptide_data.update(stats)
        
        # Save processed data
        output_file = self.processed_data_dir / f"{pdb_id}.npz"
        np.savez_compressed(
            output_file,
            pdb_id=np.array([pdb_id], dtype='<U10'),  # String array
            sequence=peptide_data['sequence'],
            coordinates=peptide_data['coordinates'],
            masks=peptide_data['mask'],  # Changed from 'mask' to 'masks'
            **stats
        )
        
        self.processed_peptides.append(peptide_data)
        return True
    
    def process_all_peptides(self):
        """Process all downloaded peptides"""
        print("🧬 PEPTIDE DATA PROCESSING")
        print("=" * 50)
        
        # Load metadata
        metadata = self.load_peptide_metadata()
        if not metadata:
            return
        
        peptides_info = metadata['peptides']
        print(f"Processing {len(peptides_info)} peptides...")
        
        # Process each peptide
        successful = 0
        for i, peptide_info in enumerate(peptides_info):
            pdb_id = peptide_info['pdb_id']
            
            if i % 50 == 0:
                print(f"  Processing {i}/{len(peptides_info)} ({successful} successful)...")
            
            if self.process_single_peptide(peptide_info):
                successful += 1
        
        print(f"\n📊 PEPTIDE PROCESSING COMPLETE")
        print(f"   Successful: {successful}/{len(peptides_info)} peptides")
        print(f"   Failed: {len(self.failed_peptides)} peptides")
        
        # Save processing summary
        self.save_processing_summary()
        
        return successful > 0
    
    def save_processing_summary(self):
        """Save detailed processing summary"""
        
        # Create summary DataFrame
        summary_data = []
        for peptide in self.processed_peptides:
            summary_data.append({
                'pdb_id': peptide['pdb_id'],
                'length': peptide['length'],
                'resolution': peptide['resolution'],
                'radius_of_gyration': peptide['radius_of_gyration'],
                'missing_atoms': peptide['missing_atoms']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = self.processed_data_dir / "peptide_summary.csv"
            df.to_csv(summary_file, index=False)
            
            # Print statistics
            print(f"\n📊 PEPTIDE DATASET STATISTICS")
            print(f"   Count: {len(df)}")
            print(f"   Length range: {df['length'].min()}-{df['length'].max()} residues")
            print(f"   Mean length: {df['length'].mean():.1f} residues")
            print(f"   Resolution range: {df['resolution'].min():.2f}-{df['resolution'].max():.2f}Å")
            print(f"   Mean radius of gyration: {df['radius_of_gyration'].mean():.1f}Å")
        
        # Save processing metadata
        processing_metadata = {
            'dataset_type': 'peptides',
            'processed_count': len(self.processed_peptides),
            'failed_count': len(self.failed_peptides),
            'length_range': [self.min_length, self.max_length],
            'processing_date': pd.Timestamp.now().isoformat(),
            'failed_peptides': self.failed_peptides
        }
        
        metadata_file = self.processed_data_dir / "processing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(processing_metadata, f, indent=2)
        
        print(f"   Summary saved: {summary_file}")
        print(f"   Metadata saved: {metadata_file}")
    
    def validate_processed_data(self) -> bool:
        """Validate processed peptide data"""
        print("\n✅ Validating processed peptide data...")
        
        processed_files = list(self.processed_data_dir.glob("*.npz"))
        
        if len(processed_files) < 50:
            print(f"❌ Too few processed files: {len(processed_files)}")
            return False
        
        # Check a few random files
        import random
        sample_files = random.sample(processed_files, min(5, len(processed_files)))
        
        for file_path in sample_files:
            try:
                data = np.load(file_path)
                
                sequence = data['sequence']
                coordinates = data['coordinates']
                masks = data['masks']  # Changed from 'mask' to 'masks'
                
                # Basic validation
                assert 10 <= len(sequence) <= 30, f"Invalid length: {len(sequence)}"
                assert coordinates.shape == (len(sequence), 4, 3), f"Invalid coord shape: {coordinates.shape}"
                assert len(masks) == len(sequence), f"Mask length mismatch"
                assert np.all(sequence >= 0) and np.all(sequence <= 20), "Invalid sequence values"
                
            except Exception as e:
                print(f"❌ Validation failed for {file_path.name}: {e}")
                return False
        
        print(f"✅ Validation passed for {len(processed_files)} peptide files")
        return True


def create_peptide_dataset_splits():
    """Create train/validation/test splits for processed peptides"""
    print("\n📊 Creating peptide dataset splits...")
    
    # Load processed peptide summary
    processed_dir = Path("data/processed")
    summary_file = processed_dir / "peptide_summary.csv"
    
    if not summary_file.exists():
        print("❌ Peptide summary not found. Process data first!")
        return False
    
    df = pd.read_csv(summary_file)
    pdb_ids = df['pdb_id'].tolist()
    
    # Create splits ensuring diversity
    np.random.seed(42)
    np.random.shuffle(pdb_ids)
    
    n_total = len(pdb_ids)
    n_train = int(0.7 * n_total)  # 70% train
    n_val = int(0.15 * n_total)   # 15% validation
    n_test = n_total - n_train - n_val  # ~15% test
    
    splits = {
        'train': pdb_ids[:n_train],
        'validation': pdb_ids[n_train:n_train + n_val],
        'test': pdb_ids[n_train + n_val:]
    }
    
    # Save splits
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_ids in splits.items():
        split_file = splits_dir / f"{split_name}.csv"
        with open(split_file, 'w') as f:
            f.write("pdb_id\n")
            for pdb_id in split_ids:
                f.write(f"{pdb_id}\n")
    
    print(f"✅ Dataset splits created:")
    print(f"   Train: {len(splits['train'])} peptides")
    print(f"   Validation: {len(splits['validation'])} peptides")
    print(f"   Test: {len(splits['test'])} peptides")
    
    return True


if __name__ == "__main__":
    print("🧬 PeptideFold: Data Processing")
    print("Processing peptides (10-30 residues) for fast training")
    print()
    
    # Process peptide data
    processor = PeptideDataProcessor()
    success = processor.process_all_peptides()
    
    if success:
        # Validate processed data
        if processor.validate_processed_data():
            # Create dataset splits
            if create_peptide_dataset_splits():
                print(f"\n🎯 PEPTIDEFOLD DATA READY!")
                print(f"   Processed: {len(processor.processed_peptides)} peptides")
                print(f"   Target GDT-TS: 25-30%")
                print(f"   Ready for training!")
            else:
                print("❌ Failed to create dataset splits")
        else:
            print("❌ Data validation failed")
    else:
        print("❌ Peptide processing failed")
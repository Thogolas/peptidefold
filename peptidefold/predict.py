#!/usr/bin/env python3
"""
PeptideFold Prediction and PyMOL Visualization
Generate structure predictions and visualize them in PyMOL
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import subprocess
import tempfile
from typing import Optional, Tuple, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.core.peptide_fold_model import PeptideFoldModel
from peptidefold.models.config import ModelConfig
# from scripts.process_peptide_data_robust import PeptideDataProcessor  # Not needed for prediction

class PeptideFoldPredictor:
    """Predict peptide structures and visualize with PyMOL"""
    
    def __init__(self, model_path: str = "results/models/peptidefold_100epoch.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.model = None
# self.processor = PeptideDataProcessor()  # Not needed for prediction
        
        # Amino acid mapping
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
            'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20  # X for unknown
        }
        
    def load_model(self) -> bool:
        """Load trained PeptideFold model"""
        if not self.model_path.exists():
            print(f"❌ Model not found at {self.model_path}")
            print("   Train the model first with: python3 train_peptide_100.py")
            return False
            
        try:
            # Initialize model architecture with exact training config
            config = ModelConfig(
                vocab_size=21,
                sequence_dim=64,       # Match training config
                hidden_dim=96,
                num_heads=4,           # Match training config  
                num_layers=2,
                dropout=0.15,
                coordinate_dim=3,
                num_backbone_atoms=4,
                max_sequence_length=40
            )
            self.model = PeptideFoldModel(config)
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded model from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to tensor"""
        sequence = sequence.upper().strip()
        
        # Validate sequence
        for aa in sequence:
            if aa not in self.aa_to_idx:
                print(f"⚠️  Unknown amino acid '{aa}', replacing with 'X'")
                sequence = sequence.replace(aa, 'X')
        
        # Convert to indices
        indices = [self.aa_to_idx[aa] for aa in sequence]
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_structure(self, sequence: str) -> Tuple[np.ndarray, float]:
        """Predict 3D structure from sequence"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate sequence length
        if len(sequence) < 8 or len(sequence) > 50:
            print(f"⚠️  Sequence length {len(sequence)} outside optimal range (8-50)")
        
        # Convert sequence to tensor
        seq_tensor = self.sequence_to_tensor(sequence)
        
        # Create batch in expected format for model
        batch = {
            'sequences': seq_tensor,
            'masks': torch.ones(seq_tensor.shape, dtype=torch.bool).to(self.device)
        }
        
        # Make prediction
        with torch.no_grad():
            output = self.model(batch)
            coordinates = output['coordinates'].cpu().numpy().squeeze()
            
            # Try different confidence key names
            if 'confidence_score' in output:
                confidence = output['confidence_score'].cpu().numpy().squeeze()
            elif 'confidence' in output:
                confidence = output['confidence'].cpu().numpy().squeeze()
            else:
                print(f"Available output keys: {list(output.keys())}")
                confidence = 0.5  # Default confidence
            
            # Handle confidence score (might be per-residue)
            if hasattr(confidence, 'ndim') and confidence.ndim > 0:
                confidence = confidence.mean()
        
        print(f"📊 Prediction confidence: {confidence:.3f}")
        return coordinates, confidence
    
    def save_as_pdb(self, coordinates: np.ndarray, sequence: str, output_path: str) -> None:
        """Save predicted structure as PDB file"""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            # PDB header
            f.write(f"HEADER    PEPTIDEFOLD PREDICTION\n")
            f.write(f"TITLE     PREDICTED STRUCTURE FOR {sequence}\n")
            f.write(f"REMARK    Generated by PeptideFold\n")
            
            # Atomic coordinates (CA atoms only for simplicity)
            for i, aa in enumerate(sequence):
                # Extract CA coordinates (atom index 1 is typically CA)
                if coordinates.ndim == 3 and coordinates.shape[1] > 1:
                    ca_coord = coordinates[i, 1, :]  # CA atom
                elif coordinates.ndim == 2:
                    ca_coord = coordinates[i, :]
                else:
                    ca_coord = coordinates[i, 0, :]  # Fallback to first atom
                
                f.write(
                    f"ATOM  {i+1:5d}  CA  {aa}   A{i+1:4d}    "
                    f"{float(ca_coord[0]):8.3f}{float(ca_coord[1]):8.3f}{float(ca_coord[2]):8.3f}"
                    f"  1.00 20.00           C\n"
                )
            
            f.write("END\n")
        
        print(f"💾 Saved PDB to {output_path}")
    
    def check_pymol(self) -> bool:
        """Check if PyMOL is available"""
        try:
            result = subprocess.run(['pymol', '-c', '-q'], 
                                  capture_output=True, 
                                  timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def visualize_with_pymol(self, pdb_path: str, sequence: str) -> None:
        """Launch PyMOL with predicted structure"""
        pdb_path = Path(pdb_path)
        
        if not self.check_pymol():
            print("❌ PyMOL not found. Install with:")
            print("   conda install -c conda-forge pymol-open-source")
            print("   or visit: https://pymol.org/")
            return
        
        # Create PyMOL script
        pymol_script = f"""
# Load predicted structure
load {pdb_path.absolute()}, prediction

# Basic visualization
show cartoon, prediction
color cyan, prediction
center prediction
zoom prediction

# Add sequence as label
set label_size, 14
label first ca, "{sequence}"

# Optimize display
set cartoon_smooth_loops, 1
set antialias, 2
set ray_shadows, 0

print "PeptideFold Prediction Loaded!"
print "Sequence: {sequence}"
print "Use 'ray' command to create high-quality image"
"""
        
        # Save script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pml', delete=False) as f:
            f.write(pymol_script)
            script_path = f.name
        
        # Launch PyMOL
        try:
            print("🚀 Launching PyMOL...")
            subprocess.Popen(['pymol', script_path])
            print("✅ PyMOL opened with prediction!")
            print("   💡 Tips:")
            print("     - Mouse: rotate, zoom, pan")
            print("     - Type 'ray' for high-quality rendering")
            print("     - File > Export Image to save")
        except Exception as e:
            print(f"❌ Error launching PyMOL: {e}")
    
    def predict_and_visualize(self, sequence: str, output_dir: str = "predictions") -> None:
        """Complete pipeline: predict structure and visualize"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🧬 Predicting structure for: {sequence}")
        
        # Load model if needed
        if self.model is None:
            if not self.load_model():
                return
        
        # Predict structure
        coordinates, confidence = self.predict_structure(sequence)
        
        # Save as PDB
        pdb_path = output_dir / f"{sequence}_prediction.pdb"
        self.save_as_pdb(coordinates, sequence, pdb_path)
        
        # Visualize
        self.visualize_with_pymol(pdb_path, sequence)
        
        print(f"🎯 Complete! Structure saved to {pdb_path}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Predict peptide structures with PeptideFold')
    parser.add_argument('sequence', type=str, help='Amino acid sequence (8-50 residues)')
    parser.add_argument('--model', type=str, default='results/models/peptidefold_100epoch.pt',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions',
                       help='Output directory for PDB files')
    parser.add_argument('--no-pymol', action='store_true',
                       help='Skip PyMOL visualization')
    
    args = parser.parse_args()
    
    # Validate sequence
    sequence = args.sequence.upper().strip()
    if not sequence:
        print("❌ Empty sequence provided")
        return
    
    if len(sequence) < 3:
        print("❌ Sequence too short (minimum 3 residues)")
        return
    
    # Create predictor
    predictor = PeptideFoldPredictor(args.model)
    
    if args.no_pymol:
        # Just predict and save PDB
        if predictor.load_model():
            coordinates, confidence = predictor.predict_structure(sequence)
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            pdb_path = output_dir / f"{sequence}_prediction.pdb"
            predictor.save_as_pdb(coordinates, sequence, pdb_path)
    else:
        # Full pipeline with visualization
        predictor.predict_and_visualize(sequence, args.output)

if __name__ == "__main__":
    main()
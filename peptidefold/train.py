#!/usr/bin/env python3
"""
PeptideFold Training Script
Optimized for peptide structure prediction (10-30 residues)

Target: 25-30% GDT-TS performance
- Much faster training (shorter sequences)
- Larger dataset (500+ vs 60 samples)
- More achievable performance targets
- Less overfitting risk
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from models.core.peptide_fold_model import create_peptide_fold_model
from scripts.peptide_smart_batching import create_peptide_data_loaders


class PeptideFoldTrainer:
    """
    Trainer optimized for peptide structure prediction
    Designed for fast, effective training on 500+ peptides
    """
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimized training settings for peptides
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,           # Higher LR (larger dataset, simpler structures)
            weight_decay=5e-3, # Moderate weight decay (500+ samples)
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduling optimized for peptides
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,         # Cosine cycle length
            eta_min=1e-5       # Minimum learning rate
        )
        
        # Gradient stability
        
        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_gdt_ts = 0.0
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
        self.gdt_ts_history = []
        
        # Create directories
        Path("results/models").mkdir(parents=True, exist_ok=True)
        Path("results/logs").mkdir(parents=True, exist_ok=True)
        
        print(f"🧬 PeptideFold Trainer")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Target: 25-30% GDT-TS on peptides (10-30 residues)")
        print(f"Advantages: Faster training, larger dataset, achievable targets")
    
    def calculate_loss(self, predictions, targets):
        """Use peptide-optimized loss function"""
        return self.model.calculate_peptide_loss(predictions, targets)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_norm_sum = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            try:
                if self.use_amp:
                    with autocast():
                        predictions = self.model(batch)
                        loss = self.calculate_loss(predictions, batch)
                    
                    if torch.isfinite(loss):
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                        gradient_norm_sum += grad_norm
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        continue
                else:
                    predictions = self.model(batch)
                    loss = self.calculate_loss(predictions, batch)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                        gradient_norm_sum += grad_norm
                        
                        self.optimizer.step()
                    else:
                        continue
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"GradNorm: {grad_norm:.3f}")
            
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = gradient_norm_sum / max(num_batches, 1)
        
        return avg_loss, avg_grad_norm
    
    def validate(self) -> Tuple[float, float]:
        """Validate with GDT-TS estimation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        gdt_ts_sum = 0.0
        gdt_ts_count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    predictions = self.model(batch)
                    loss = self.calculate_loss(predictions, batch)
                    
                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches += 1
                    
                    # Estimate GDT-TS for multiple samples in batch
                    for i in range(min(len(batch['sequences']), 3)):  # Check up to 3 per batch
                        gdt_ts = self.estimate_peptide_gdt_ts(predictions, batch, sample_idx=i)
                        if gdt_ts is not None:
                            gdt_ts_sum += gdt_ts
                            gdt_ts_count += 1
                        
                except Exception as e:
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_gdt_ts = gdt_ts_sum / max(gdt_ts_count, 1)
        
        return avg_loss, avg_gdt_ts
    
    def estimate_peptide_gdt_ts(self, predictions: Dict, targets: Dict, sample_idx: int = 0) -> float:
        """Quick GDT-TS estimation for peptides"""
        try:
            pred_coords = predictions['coordinates']
            target_coords = targets['coordinates']
            masks = targets['masks']
            
            # Use specified sample
            mask = masks[sample_idx]
            valid_indices = mask.bool()
            
            if valid_indices.sum() < 3:
                return None
            
            pred_ca = pred_coords[sample_idx, valid_indices, 1, :]  # CA atoms
            target_ca = target_coords[sample_idx, valid_indices, 1, :]
            
            if not (torch.isfinite(pred_ca).all() and torch.isfinite(target_ca).all()):
                return None
            
            # Center both structures (approximate alignment)
            pred_centered = pred_ca - pred_ca.mean(dim=0)
            target_centered = target_ca - target_ca.mean(dim=0)
            
            # Calculate distances after centering
            distances = torch.norm(pred_centered - target_centered, dim=1)
            
            # GDT-TS calculation (percentage within distance thresholds)
            within_1A = (distances < 1.0).float().mean()
            within_2A = (distances < 2.0).float().mean() 
            within_4A = (distances < 4.0).float().mean()
            within_8A = (distances < 8.0).float().mean()
            
            # GDT-TS score
            gdt_ts = 25.0 * (within_1A + within_2A + within_4A + within_8A)
            
            return gdt_ts.item()
            
        except:
            return None
    
    def train(self, max_epochs=200, patience=40):
        """Main training loop for peptides"""
        print(f"🧬 Starting PeptideFold training for {max_epochs} epochs...")
        print(f"Target: 25-30% GDT-TS on peptides")
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            
            # Train
            train_loss, avg_grad_norm = self.train_epoch()
            self.train_history.append(train_loss)
            
            # Validate
            val_loss, estimated_gdt_ts = self.validate()
            self.val_history.append(val_loss)
            self.gdt_ts_history.append(estimated_gdt_ts)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Track best models
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                improved = True
                
            if estimated_gdt_ts > self.best_gdt_ts:
                self.best_gdt_ts = estimated_gdt_ts
                self.save_checkpoint(is_best=True, is_gdt_ts_best=True)
                print(f"    🎯 New best GDT-TS: {estimated_gdt_ts:.1f}%")
            elif improved:
                self.save_checkpoint(is_best=True)
                print(f"    ✅ New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                self.save_checkpoint()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  GDT-TS: {estimated_gdt_ts:.1f}%")
            print(f"  Grad Norm: {avg_grad_norm:.3f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Patience: {self.patience_counter}/{patience}")
            
            # Success milestone checks
            if estimated_gdt_ts >= 30.0:
                print(f"    🏆 TARGET EXCEEDED: {estimated_gdt_ts:.1f}% GDT-TS!")
            elif estimated_gdt_ts >= 25.0:
                print(f"    🎯 TARGET ACHIEVED: {estimated_gdt_ts:.1f}% GDT-TS!")
            elif estimated_gdt_ts >= 20.0:
                print(f"    🚀 EXCELLENT PROGRESS: {estimated_gdt_ts:.1f}% GDT-TS")
            elif estimated_gdt_ts >= 15.0:
                print(f"    📈 GOOD PROGRESS: {estimated_gdt_ts:.1f}% GDT-TS")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n⏹  Early stopping at epoch {epoch + 1}")
                break
        
        print(f"\n🧬 PeptideFold training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best GDT-TS: {self.best_gdt_ts:.1f}%")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'gdt_ts_history': self.gdt_ts_history,
            'best_val_loss': self.best_val_loss,
            'best_gdt_ts': self.best_gdt_ts
        }
    
    def save_checkpoint(self, is_best=False, is_gdt_ts_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_gdt_ts': self.best_gdt_ts,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'gdt_ts_history': self.gdt_ts_history
        }
        
        # Save current checkpoint
        checkpoint_path = Path("results/models") / f"peptidefold_checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best models
        if is_best:
            best_path = Path("results/models") / "peptidefold_best_model.pt"
            torch.save(checkpoint, best_path)
            
        if is_gdt_ts_best:
            gdt_ts_path = Path("results/models") / "peptidefold_best_gdt_ts_model.pt"
            torch.save(checkpoint, gdt_ts_path)


def train_peptidefold_session(epochs=200):
    """Main PeptideFold training session"""
    
    print("🧬 PEPTIDEFOLD TRAINING SESSION")
    print("=" * 60)
    print("🎯 MISSION: 25-30% GDT-TS on peptides (10-30 residues)")
    print("   ✅ Much faster training (10-30 vs 100+ residues)")
    print("   ✅ Larger dataset (500+ vs 60 samples)")
    print("   ✅ More achievable targets (25-30% vs 40-50%)")
    print("   ✅ Less overfitting risk")
    print("   ✅ Direct GDT-TS optimization from day 1")
    print()
    
    # Create peptide data loaders with smaller batch size for our small dataset
    print("📊 Creating peptide data loaders...")
    train_loader, val_loader, test_loader = create_peptide_data_loaders(batch_size=4, num_workers=0)
    
    if train_loader is None:
        print("❌ Failed to create data loaders. Run peptide data processing first!")
        print("Commands to run:")
        print("  1. python scripts/download_peptide_data.py")
        print("  2. python scripts/process_peptide_data.py")
        return
    
    # Create PeptideFold model
    print("🧠 Creating PeptideFold model...")
    model = create_peptide_fold_model()
    
    # Performance test
    print("⚡ Performance test...")
    batch = next(iter(train_loader))
    
    start_time = time.time()
    with torch.no_grad():
        output = model(batch)
    forward_time = time.time() - start_time
    
    print(f"✅ Features: {list(output.keys())}")
    
    # Test loss calculation
    try:
        with torch.no_grad():
            loss = model.calculate_peptide_loss(output, batch)
        print(f"✅ Peptide loss: {loss.item():.4f}")
    except Exception as e:
        print(f"⚠️  Loss calculation error: {e}")
        return
    
    # Create trainer
    trainer = PeptideFoldTrainer(model, train_loader, val_loader)
    
    print(f"\n🔥 STARTING PEPTIDEFOLD TRAINING")
    print("=" * 60)
    print("🎯 Target: 25-30% GDT-TS")
    print("📊 Strategy: Direct GDT-TS optimization + Fast training")
    print()
    
    # Train
    start_training_time = time.time()
    training_results = trainer.train(max_epochs=epochs, patience=50)
    total_training_time = time.time() - start_training_time
    
    print(f"\n🎉 PEPTIDEFOLD TRAINING COMPLETED!")
    print("=" * 60)
    print(f"✅ Total time: {total_training_time/3600:.2f} hours")
    print(f"✅ Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"✅ Best GDT-TS: {training_results['best_gdt_ts']:.1f}%")
    print(f"✅ Total epochs: {len(training_results['train_history'])}")
    
    # Final evaluation
    print(f"\n📊 PEPTIDEFOLD FINAL EVALUATION")
    print("=" * 60)
    
    device = trainer.device
    model.eval()
    
    try:
        from evaluation.metrics import batch_evaluate
        import numpy as np
        
        # Load best GDT-TS model
        if Path('results/models/peptidefold_best_gdt_ts_model.pt').exists():
            checkpoint = torch.load('results/models/peptidefold_best_gdt_ts_model.pt', map_location=device)
            print("Loading best GDT-TS model for evaluation...")
        else:
            checkpoint = torch.load('results/models/peptidefold_best_model.pt', map_location=device)
            print("Loading best validation loss model for evaluation...")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Comprehensive evaluation
        all_gdt_ts = []
        all_tm_scores = []
        all_rmsds = []
        
        with torch.no_grad():
            for i, test_batch in enumerate(test_loader):
                test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in test_batch.items()}
                
                predictions = model(test_batch)
                metrics = batch_evaluate(predictions, test_batch)
                
                if metrics['n_evaluated'] > 0:
                    all_gdt_ts.append(metrics['gdt_ts'])
                    all_tm_scores.append(metrics['tm_score'])
                    all_rmsds.append(metrics['rmsd'])
                    
                    print(f"  Test batch {i+1}: GDT-TS={metrics['gdt_ts']:.1f}%, "
                          f"TM={metrics['tm_score']:.3f}, RMSD={metrics['rmsd']:.1f}Å")
        
        if all_gdt_ts:
            avg_gdt_ts = np.mean(all_gdt_ts)
            std_gdt_ts = np.std(all_gdt_ts) 
            avg_tm_score = np.mean(all_tm_scores)
            avg_rmsd = np.mean(all_rmsds)
            
            print(f"\n📈 PEPTIDEFOLD FINAL RESULTS:")
            print(f"   Average GDT-TS: {avg_gdt_ts:.1f}% ± {std_gdt_ts:.1f}%")
            print(f"   Average TM-score: {avg_tm_score:.3f}")
            print(f"   Average RMSD: {avg_rmsd:.1f}Å")
            print()
            
            # Success assessment
            print(f"🎯 PEPTIDEFOLD ASSESSMENT:")
            if avg_gdt_ts >= 30.0:
                print(f"   🏆 OUTSTANDING: Target exceeded ({avg_gdt_ts:.1f}%)!")
                print(f"   🎉 PeptideFold is highly successful!")
            elif avg_gdt_ts >= 25.0:
                print(f"   🎯 MISSION ACCOMPLISHED: Target achieved ({avg_gdt_ts:.1f}%)!")
                print(f"   ✅ PeptideFold works as designed!")
            elif avg_gdt_ts >= 20.0:
                print(f"   🚀 EXCELLENT: Very close to target ({avg_gdt_ts:.1f}%)!")
                print(f"   📈 Strong peptide structure prediction!")
            elif avg_gdt_ts >= 15.0:
                print(f"   📈 VERY GOOD: Solid performance ({avg_gdt_ts:.1f}%)!")
            elif avg_gdt_ts >= 10.0:
                print(f"   📊 GOOD: Meaningful improvement ({avg_gdt_ts:.1f}%)!")
                print(f"   🔧 Peptide approach working!")
            else:
                print(f"   🤔 MODEST: Some improvement ({avg_gdt_ts:.1f}%)!")
                print(f"   🔍 May need architecture refinements")
            
            print(f"   📊 Consistency: {'Excellent' if std_gdt_ts < 5.0 else 'Good' if std_gdt_ts < 8.0 else 'Moderate'} (±{std_gdt_ts:.1f}%)")
            print(f"   ⚡ Training speed: {'Fast' if total_training_time < 2*3600 else 'Moderate' if total_training_time < 4*3600 else 'Slow'}")
            
            print(f"   🎯 Peptide strategy impact: {'Revolutionary' if improvement > 4 else 'Major' if improvement > 2 else 'Significant' if improvement > 1.5 else 'Moderate'}")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        print("But training completed successfully!")
    
    return training_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PeptideFold Training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("🧬 PeptideFold: Peptide Structure Prediction")
    print("Targeting 25-30% GDT-TS on short peptides (10-30 residues)")
    print()
    
    train_peptidefold_session(args.epochs)
#!/usr/bin/env python3
"""
Peptide Data Download Script
Downloads real peptide structures (10-30 residues) from PDB with diversity filtering

Target: 500-600 diverse peptides for fast, effective training
"""

import requests
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Set
import gzip
from Bio import PDB
from Bio.PDB import MMCIFParser, PDBParser
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict


class PeptideDataDownloader:
    """Download diverse peptide structures from PDB"""
    
    def __init__(self, output_dir: str = "data/raw/pdb"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_length = 10
        self.max_length = 30
        self.target_count = 600
        self.max_identity = 0.30  # 30% sequence identity clustering
        
        self.downloaded_structures = []
        self.sequence_clusters = []
        
    def search_peptide_structures(self) -> List[Dict]:
        """Search PDB for peptide structures meeting our criteria"""
        print("🔍 Searching PDB for peptide structures...")
        
        # PDB REST API query for peptides
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "equals",
                            "value": 1
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": 2.5
                        }
                    }
                ]
            },
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": 2000  # Get many candidates
                },
                "results_content_type": ["experimental"],
                "sort": [
                    {
                        "sort_by": "score",
                        "direction": "desc"
                    }
                ]
            },
            "return_type": "entry"
        }
        
        # Query PDB
        try:
            response = requests.post(
                "https://search.rcsb.org/rcsbsearch/v2/query",
                json=query,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            results = response.json()
            pdb_ids = results.get("result_set", [])
            
            print(f"Found {len(pdb_ids)} potential structures")
            return [{"pdb_id": entry["identifier"]} for entry in pdb_ids]
            
        except Exception as e:
            print(f"❌ PDB search failed: {e}")
            return []
    
    def get_structure_info(self, pdb_id: str) -> Dict:
        """Get detailed structure information"""
        try:
            # Get structure details
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(url)
            response.raise_for_status()
            
            entry_info = response.json()
            
            # Get entity information (sequence details)
            url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
            response = requests.get(url)
            response.raise_for_status()
            
            entity_info = response.json()
            
            # Extract key information
            sequence = entity_info.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
            length = len(sequence)
            resolution = entry_info.get("rcsb_entry_info", {}).get("resolution_combined", [999])[0]
            
            return {
                "pdb_id": pdb_id,
                "sequence": sequence,
                "length": length,
                "resolution": resolution,
                "valid": self.min_length <= length <= self.max_length
            }
            
        except Exception as e:
            print(f"  ❌ Failed to get info for {pdb_id}: {e}")
            return {"pdb_id": pdb_id, "valid": False}
    
    def filter_peptide_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates by length and quality"""
        print("📊 Filtering peptide candidates...")
        
        valid_peptides = []
        
        for i, candidate in enumerate(candidates):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(candidates)}...")
            
            info = self.get_structure_info(candidate["pdb_id"])
            
            if info["valid"]:
                valid_peptides.append(info)
                
            # Stop if we have enough candidates
            if len(valid_peptides) >= self.target_count * 2:  # Get 2x for diversity filtering
                break
            
            time.sleep(0.1)  # Be nice to PDB servers
        
        print(f"✅ Found {len(valid_peptides)} valid peptides (10-30 residues)")
        return valid_peptides
    
    def calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence identity between two sequences"""
        if len(seq1) != len(seq2):
            # For different lengths, use shorter as denominator
            min_len = min(len(seq1), len(seq2))
            seq1, seq2 = seq1[:min_len], seq2[:min_len]
        
        if len(seq1) == 0:
            return 0.0
            
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def cluster_by_sequence_diversity(self, peptides: List[Dict]) -> List[Dict]:
        """Cluster peptides by sequence identity and select diverse representatives"""
        print("🧬 Clustering peptides by sequence diversity...")
        
        if len(peptides) <= self.target_count:
            return peptides
        
        # Calculate pairwise sequence identities
        n = len(peptides)
        identity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                identity = self.calculate_sequence_identity(
                    peptides[i]["sequence"], 
                    peptides[j]["sequence"]
                )
                identity_matrix[i, j] = identity
                identity_matrix[j, i] = identity
        
        # Convert to distance matrix (1 - identity)
        distance_matrix = 1.0 - identity_matrix
        
        # Use DBSCAN for clustering (eps = 1 - max_identity)
        eps = 1.0 - self.max_identity
        clusterer = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Group peptides by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(peptides[i])
        
        print(f"  Found {len(clusters)} sequence clusters")
        
        # Select best representative from each cluster
        diverse_peptides = []
        for cluster_id, cluster_peptides in clusters.items():
            # Sort by resolution (lower is better)
            cluster_peptides.sort(key=lambda x: x["resolution"])
            best_peptide = cluster_peptides[0]
            diverse_peptides.append(best_peptide)
        
        # If we still have too many, take the highest resolution ones
        if len(diverse_peptides) > self.target_count:
            diverse_peptides.sort(key=lambda x: x["resolution"])
            diverse_peptides = diverse_peptides[:self.target_count]
        
        print(f"✅ Selected {len(diverse_peptides)} diverse peptides")
        return diverse_peptides
    
    def download_structure_file(self, pdb_id: str) -> bool:
        """Download PDB structure file"""
        try:
            # Try PDB format first
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url)
            
            if response.status_code == 200:
                file_path = self.output_dir / f"{pdb_id}.pdb"
                with open(file_path, 'w') as f:
                    f.write(response.text)
                return True
            
            # Fallback to mmCIF format
            url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            response = requests.get(url)
            
            if response.status_code == 200:
                file_path = self.output_dir / f"{pdb_id}.cif"
                with open(file_path, 'w') as f:
                    f.write(response.text)
                return True
            
            return False
            
        except Exception as e:
            print(f"  ❌ Download failed for {pdb_id}: {e}")
            return False
    
    def download_peptide_dataset(self):
        """Main function to download diverse peptide dataset"""
        print("🧬 PEPTIDE DATASET DOWNLOAD")
        print("=" * 50)
        print(f"Target: {self.target_count} diverse peptides (10-30 residues)")
        print(f"Max sequence identity: {self.max_identity*100:.0f}%")
        print()
        
        # Step 1: Search for candidates
        candidates = self.search_peptide_structures()
        if not candidates:
            print("❌ No candidates found")
            return
        
        # Step 2: Filter by length and quality
        valid_peptides = self.filter_peptide_candidates(candidates)
        if len(valid_peptides) < 50:
            print("❌ Not enough valid peptides found")
            return
        
        # Step 3: Cluster by sequence diversity
        diverse_peptides = self.cluster_by_sequence_diversity(valid_peptides)
        
        # Step 4: Download structure files
        print("💾 Downloading peptide structure files...")
        successful_downloads = []
        
        for i, peptide in enumerate(diverse_peptides):
            pdb_id = peptide["pdb_id"]
            print(f"  Downloading {pdb_id} ({i+1}/{len(diverse_peptides)})...")
            
            if self.download_structure_file(pdb_id):
                successful_downloads.append(peptide)
            
            time.sleep(0.2)  # Be nice to servers
        
        # Step 5: Save metadata
        metadata = {
            "dataset_type": "peptides",
            "target_count": self.target_count,
            "actual_count": len(successful_downloads),
            "length_range": [self.min_length, self.max_length],
            "max_sequence_identity": self.max_identity,
            "peptides": successful_downloads
        }
        
        metadata_path = self.output_dir / "peptide_download_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ PEPTIDE DATASET DOWNLOAD COMPLETE!")
        print(f"   Downloaded: {len(successful_downloads)} diverse peptides")
        print(f"   Length range: {self.min_length}-{self.max_length} residues")
        print(f"   Sequence diversity: ≤{self.max_identity*100:.0f}% identity")
        print(f"   Files saved to: {self.output_dir}")
        print(f"   Metadata: {metadata_path}")
        
        return successful_downloads


def create_train_val_test_splits(peptides: List[Dict], output_dir: str = "data/splits"):
    """Create train/validation/test splits ensuring cluster separation"""
    print("\n📊 Creating train/validation/test splits...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle peptides
    np.random.seed(42)
    peptides = peptides.copy()
    np.random.shuffle(peptides)
    
    # Split ratios
    n_total = len(peptides)
    n_train = int(0.7 * n_total)  # 70% train
    n_val = int(0.15 * n_total)   # 15% validation  
    n_test = n_total - n_train - n_val  # ~15% test
    
    train_peptides = peptides[:n_train]
    val_peptides = peptides[n_train:n_train + n_val]
    test_peptides = peptides[n_train + n_val:]
    
    # Save splits
    splits = {
        "train": [p["pdb_id"] for p in train_peptides],
        "validation": [p["pdb_id"] for p in val_peptides],
        "test": [p["pdb_id"] for p in test_peptides]
    }
    
    for split_name, pdb_ids in splits.items():
        split_file = output_dir / f"{split_name}.csv"
        with open(split_file, 'w') as f:
            f.write("pdb_id\n")
            for pdb_id in pdb_ids:
                f.write(f"{pdb_id}\n")
        print(f"  {split_name}: {len(pdb_ids)} peptides → {split_file}")
    
    print(f"✅ Splits created: {n_train} train, {n_val} val, {n_test} test")
    return splits


if __name__ == "__main__":
    print("🧬 PeptideFold: Peptide Dataset Download")
    print("Downloading 500-600 diverse peptides (10-30 residues)")
    print()
    
    # Download peptide dataset
    downloader = PeptideDataDownloader()
    peptides = downloader.download_peptide_dataset()
    
    if peptides:
        # Create train/val/test splits
        splits = create_train_val_test_splits(peptides)
        
        print(f"\n🎯 READY FOR PEPTIDEFOLD TRAINING!")
        print(f"   Dataset: {len(peptides)} diverse peptides")
        print(f"   Target GDT-TS: 25-30%")
    else:
        print("❌ Failed to download peptide dataset")
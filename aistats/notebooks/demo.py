#!/usr/bin/env python3
"""
Minimal demo script for C²BA model.

This script provides a quick demonstration of the C²BA model training
and evaluation pipeline with minimal configuration for testing purposes.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from c2ba import (
    HeartDiseaseDataset, DataProcessor, C2BAModel, ModelTrainer,
    MetricsCalculator, set_random_seeds, load_and_preprocess_data
)
from c2ba.data import create_distribution_shifts


def main():
    """Run minimal demo of C²BA model."""
    
    print("=" * 60)
    print("C²BA MINIMAL DEMO")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\nLoading data...")
    df = load_and_preprocess_data()
    processor = DataProcessor()
    X, y = processor.preprocess_data(df, is_training=True)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Create simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create datasets and loaders
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    test_dataset = HeartDiseaseDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = C2BAModel(
        input_dim=X.shape[1],
        num_classes=len(np.unique(y)),
        foundation_dim=16,  # Small for demo
        adapter_rank=4
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ModelTrainer(model, device=device, save_dir='demo_checkpoints')
    
    # Quick training (1 epoch for demo)
    print("\nTraining for 1 epoch (demo)...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Use test as val for demo
        num_epochs=1,
        patience=1,
        lr_foundation=1e-2,
        lr_adapter=2e-2,
        lr_others=1e-2
    )
    
    # Evaluation
    print("\nEvaluating model...")
    results = trainer.evaluate(test_loader)
    metrics = MetricsCalculator.compute_comprehensive_metrics(
        results['predictions'], results['targets'], len(np.unique(y))
    )
    
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n✓ Demo completed successfully!")
    print("For full training, use: python scripts/train.py --config configs/default.yaml")


if __name__ == '__main__':
    main()

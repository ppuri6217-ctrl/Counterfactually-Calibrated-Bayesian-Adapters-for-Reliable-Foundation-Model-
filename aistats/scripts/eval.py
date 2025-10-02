#!/usr/bin/env python3
"""
Evaluation script for C²BA (Counterfactually-Calibrated Bayesian Adapter) model.

This script loads a trained model and evaluates it on various metrics including
calibration, uncertainty quantification, and distribution shift analysis.

Usage:
    python scripts/eval.py --config configs/default.yaml --ckpt checkpoints/best_c2ba_model.pt
    python scripts/eval.py --config configs/default.yaml --ckpt checkpoints/best_c2ba_model.pt --output results.json
"""

import argparse
import os
import sys
import yaml
import json
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from c2ba import (
    HeartDiseaseDataset, DataProcessor, C2BAModel, ModelTrainer,
    MetricsCalculator, set_random_seeds, load_and_preprocess_data,
    print_model_summary, create_results_summary, print_results_summary
)
from c2ba.data import create_distribution_shifts


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_reliability_diagram(y_prob, y_true, save_path=None):
    """Create and save reliability diagram."""
    bin_data = MetricsCalculator.reliability_diagram_data(y_prob, y_true, n_bins=10)
    
    if not bin_data:
        print("No data for reliability diagram")
        return
    
    # Extract data for plotting
    bin_centers = [(d['bin_lower'] + d['bin_upper']) / 2 for d in bin_data]
    accuracies = [d['accuracy'] for d in bin_data]
    confidences = [d['confidence'] for d in bin_data]
    counts = [d['count'] for d in bin_data]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.scatter(confidences, accuracies, s=[c/10 for c in counts], alpha=0.7, label='Bin Accuracy')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence histogram
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_uncertainty_analysis_plots(uncertainties, correct_predictions, save_dir=None):
    """Create uncertainty analysis plots."""
    
    # Uncertainty vs accuracy plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bin uncertainties and compute accuracy in each bin
    n_bins = 10
    uncertainty_bins = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
    bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(correct_predictions[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    # Plot uncertainty vs accuracy
    ax1.bar(bin_centers, bin_accuracies, width=(bin_centers[1] - bin_centers[0]) * 0.8, alpha=0.7)
    ax1.set_xlabel('Uncertainty')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Uncertainty')
    ax1.grid(True, alpha=0.3)
    
    # Plot uncertainty distribution
    ax2.hist(uncertainties, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Count')
    ax2.set_title('Uncertainty Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'uncertainty_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty analysis plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate C²BA model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Path to save results JSON')
    parser.add_argument('--plots', type=str, help='Directory to save plots')
    parser.add_argument('--data', type=str, help='Path to evaluation data (if different from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    # Set random seeds
    set_random_seeds(config.get('seed', 42))
    
    print("=" * 80)
    print("C²BA MODEL EVALUATION")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    
    if args.data:
        # Load custom data
        import pandas as pd
        df = pd.read_csv(args.data)
    else:
        df = load_and_preprocess_data()
    
    processor = DataProcessor()
    X, y = processor.preprocess_data(df, is_training=True)
    
    # Update input dimension in config
    config['model']['input_dim'] = X.shape[1]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create distribution shift
    shift_type = config['data'].get('shift_type', 'age')
    train_mask, test_mask = create_distribution_shifts(X, y, shift_type=shift_type)
    
    X_train_shift, y_train_shift = X[train_mask], y[train_mask]
    X_test_shift, y_test_shift = X[test_mask], y[test_mask]
    
    # Standard train/validation split on shifted training data
    val_split = config['data'].get('val_split', 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_shift, y_train_shift, 
        test_size=val_split, 
        random_state=config.get('seed', 42), 
        stratify=y_train_shift
    )
    
    # Create datasets and data loaders
    val_dataset = HeartDiseaseDataset(X_val, y_val)
    test_dataset = HeartDiseaseDataset(X_test_shift, y_test_shift)
    
    batch_size = config['data']['batch_size']
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)
    
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create and load model
    model = C2BAModel(
        input_dim=config['model']['input_dim'],
        num_classes=config['model']['num_classes'],
        foundation_dim=config['model']['foundation_dim'],
        adapter_rank=config['model']['adapter_rank']
    )
    
    print_model_summary(model)
    
    # Load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    
    checkpoint = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✓ Model loaded from {args.ckpt}")
    
    # Create trainer for evaluation
    trainer = ModelTrainer(model, device=device)
    
    # Evaluation
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Evaluate on validation set
    print("Validation Set Evaluation:")
    val_results = trainer.evaluate(val_loader)
    val_metrics = MetricsCalculator.compute_comprehensive_metrics(
        val_results['predictions'], val_results['targets'], 
        config['model']['num_classes']
    )
    
    for metric, value in val_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Evaluate on test set (distribution shifted)
    print("\nTest Set Evaluation (Distribution Shifted):")
    test_results = trainer.evaluate(test_loader)
    test_metrics = MetricsCalculator.compute_comprehensive_metrics(
        test_results['predictions'], test_results['targets'],
        config['model']['num_classes']
    )
    
    for metric, value in test_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Distribution shift analysis
    print("\nDistribution Shift Analysis:")
    shift_analysis = trainer.analyze_distribution_shift(val_loader, test_loader)
    
    print(f"  MMD Score: {shift_analysis['mmd_score']:.6f}")
    print(f"  Energy Distance: {shift_analysis['energy_distance']:.6f}")
    
    # Uncertainty analysis
    uncertainty_results = None
    if config['uncertainty']['enable']:
        print("\nUncertainty Analysis:")
        uncertainty_results = trainer.get_uncertainty_estimates(
            test_loader, num_mc_samples=config['uncertainty']['num_mc_samples']
        )
        
        uncertainties = uncertainty_results['uncertainties']
        predictions = uncertainty_results['predictions']
        targets = uncertainty_results['targets']
        
        pred_classes = np.argmax(predictions, axis=1)
        correct = (pred_classes == targets).astype(int)
        
        uncertainty_metrics = MetricsCalculator.uncertainty_quality_metrics(
            uncertainties, correct
        )
        
        print(f"  Uncertainty-Error Correlation: {uncertainty_metrics['uncertainty_error_correlation']:.4f}")
        print(f"  Mean Uncertainty: {uncertainties.mean():.4f}")
        print(f"  Std Uncertainty: {uncertainties.std():.4f}")
    
    # Create and print results summary
    summary = create_results_summary(val_metrics, test_metrics, shift_analysis)
    print_results_summary(summary)
    
    # Generate plots if requested
    if args.plots:
        os.makedirs(args.plots, exist_ok=True)
        
        # Reliability diagram
        test_probs = test_results['predictions']
        test_targets = test_results['targets']
        
        if config['model']['num_classes'] == 2:
            # For binary classification, use class 1 probabilities
            reliability_path = os.path.join(args.plots, 'reliability_diagram.png')
            create_reliability_diagram(
                test_probs[:, 1], test_targets, save_path=reliability_path
            )
        
        # Uncertainty analysis plots
        if uncertainty_results is not None:
            create_uncertainty_analysis_plots(
                uncertainty_results['uncertainties'],
                (np.argmax(uncertainty_results['predictions'], axis=1) == uncertainty_results['targets']).astype(int),
                save_dir=args.plots
            )
    
    # Save results if requested
    if args.output:
        results = {
            'config': config,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'distribution_shift': {
                'mmd_score': shift_analysis['mmd_score'],
                'energy_distance': shift_analysis['energy_distance']
            },
            'summary': summary
        }
        
        if uncertainty_results is not None:
            results['uncertainty_analysis'] = {
                'mean_uncertainty': float(uncertainty_results['uncertainties'].mean()),
                'std_uncertainty': float(uncertainty_results['uncertainties'].std()),
                'uncertainty_error_correlation': float(uncertainty_metrics['uncertainty_error_correlation'])
            }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {args.output}")
    
    print(f"\n✓ Evaluation completed successfully!")


if __name__ == '__main__':
    main()

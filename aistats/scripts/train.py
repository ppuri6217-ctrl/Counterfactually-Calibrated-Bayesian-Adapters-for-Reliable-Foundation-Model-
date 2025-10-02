#!/usr/bin/env python3
"""
Training script for C²BA (Counterfactually-Calibrated Bayesian Adapter) model.

This script supports both single GPU and distributed training using torchrun.
It loads configuration from YAML files and provides comprehensive training
with evaluation metrics and model checkpointing.

Usage:
    # Single GPU training
    python scripts/train.py --config configs/default.yaml
    
    # Distributed training (2 GPUs)
    torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml
    
    # Override config parameters
    python scripts/train.py --config configs/default.yaml --override optim.lr_foundation=2e-3
"""

import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from c2ba import (
    HeartDiseaseDataset, DataProcessor, C2BAModel, ModelTrainer,
    MetricsCalculator, set_random_seeds, load_and_preprocess_data,
    print_model_summary, create_results_summary, print_results_summary
)
from c2ba.data import create_distribution_shifts


def load_config(config_path, overrides=None):
    """Load configuration from YAML file with optional overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        for override in overrides:
            if '=' not in override:
                continue
            key, value = override.split('=', 1)
            
            # Parse value type
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
            
            # Set nested key
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    return config


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_data_loaders(config, processor, X, y, rank=0, world_size=1):
    """Create data loaders with optional distributed sampling."""
    
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
    
    # Create datasets
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    val_dataset = HeartDiseaseDataset(X_val, y_val)
    test_dataset = HeartDiseaseDataset(X_test_shift, y_test_shift)
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for evaluation
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, (train_sampler, val_sampler, test_sampler)


def main():
    parser = argparse.ArgumentParser(description='Train C²BA model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--override', nargs='*', default=[], help='Override config parameters')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Load configuration
    config = load_config(args.config, args.override)
    
    # Set device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    
    # Set random seeds
    set_random_seeds(config.get('seed', 42))
    
    if is_main_process:
        print("=" * 80)
        print("C²BA: COUNTERFACTUALLY-CALIBRATED BAYESIAN ADAPTER")
        print("=" * 80)
        print(f"Configuration: {args.config}")
        print(f"Device: {device}")
        print(f"Distributed: {world_size > 1} (rank {rank}/{world_size})")
        
    # Load and preprocess data
    if is_main_process:
        print("\nLoading and preprocessing data...")
    
    df = load_and_preprocess_data()
    processor = DataProcessor()
    X, y = processor.preprocess_data(df, is_training=True)
    
    # Update input dimension in config
    config['model']['input_dim'] = X.shape[1]
    
    if is_main_process:
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {np.bincount(y)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, samplers = create_data_loaders(
        config, processor, X, y, rank, world_size
    )
    train_sampler, val_sampler, test_sampler = samplers
    
    if is_main_process:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = C2BAModel(
        input_dim=config['model']['input_dim'],
        num_classes=config['model']['num_classes'],
        foundation_dim=config['model']['foundation_dim'],
        adapter_rank=config['model']['adapter_rank']
    )
    
    if is_main_process:
        print_model_summary(model)
    
    # Setup distributed model
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create trainer
    save_dir = config['training']['save_dir']
    if world_size > 1:
        save_dir = f"{save_dir}_rank{rank}"
    
    trainer = ModelTrainer(model, device=device, save_dir=save_dir)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_model(args.resume)
        if is_main_process:
            print(f"Resumed training from {args.resume}")
    
    # Training
    if is_main_process:
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        lr_foundation=config['optim']['lr_foundation'],
        lr_adapter=config['optim']['lr_adapter'],
        lr_others=config['optim']['lr_others']
    )
    
    # Evaluation (only on main process to avoid duplication)
    if is_main_process:
        print("\n" + "=" * 80)
        print("EVALUATION")
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
        if config['evaluation'].get('shift_analysis', True):
            print("\nDistribution Shift Analysis:")
            shift_analysis = trainer.analyze_distribution_shift(val_loader, test_loader)
            
            print(f"  MMD Score: {shift_analysis['mmd_score']:.6f}")
            print(f"  Energy Distance: {shift_analysis['energy_distance']:.6f}")
        
        # Uncertainty analysis
        if config['evaluation'].get('uncertainty_analysis', True) and config['uncertainty']['enable']:
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
        
        # Create and print results summary
        summary = create_results_summary(
            val_metrics, test_metrics, 
            shift_analysis if config['evaluation'].get('shift_analysis', True) else None
        )
        print_results_summary(summary)
        
        # Save final model
        final_model_path = os.path.join(save_dir, 'final_model.pt')
        trainer.save_model(final_model_path)
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Best model saved in: {save_dir}")
        print(f"✓ Final model saved as: {final_model_path}")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == '__main__':
    main()

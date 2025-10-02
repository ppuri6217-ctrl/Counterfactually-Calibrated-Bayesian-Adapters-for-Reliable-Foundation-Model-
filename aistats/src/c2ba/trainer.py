"""
Training pipeline and utilities for C²BA model.

This module contains the ModelTrainer class which handles the complete training
loop including early stopping, learning rate scheduling, and comprehensive
evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os


class ModelTrainer:
    """Comprehensive training pipeline for C²BA model"""
    
    def __init__(self, model, device='cpu', save_dir='checkpoints'):
        """
        Initialize the model trainer.
        
        Args:
            model: The C2BA model to train
            device (str): Device to use for training ('cpu' or 'cuda')
            save_dir (str): Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def train_epoch(self, train_loader, optimizer, kl_weight=1.0):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            kl_weight (float): Weight for KL divergence loss (for annealing)
            
        Returns:
            dict: Training statistics for the epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data, compute_uncertainty=True)
            
            # Compute losses
            loss_dict = self.model.compute_loss(outputs, targets)
            
            # Apply KL annealing
            loss_dict['total_loss'] = (loss_dict['ce_loss'] + 
                                     kl_weight * loss_dict['kl_loss'] + 
                                     loss_dict['shift_loss'])
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss_dict['total_loss'].item()
            pred = outputs['probabilities'].argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def evaluate(self, val_loader):
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            dict: Evaluation statistics
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data, compute_uncertainty=False)
                loss_dict = self.model.compute_loss(outputs, targets)
                
                val_loss += loss_dict['total_loss'].item()
                pred = outputs['probabilities'].argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
                
                all_probs.extend(outputs['probabilities'].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Compute additional metrics
        pred_classes = np.argmax(all_probs, axis=1)
        f1 = f1_score(all_targets, pred_classes, average='weighted')
        
        return {
            'loss': val_loss / len(val_loader),
            'accuracy': 100. * correct / total,
            'f1_score': f1,
            'predictions': all_probs,
            'targets': all_targets
        }
    
    def train(self, train_loader, val_loader, num_epochs=100, patience=10, 
              lr_foundation=1e-3, lr_adapter=5e-3, lr_others=1e-3):
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs (int): Maximum number of epochs
            patience (int): Early stopping patience
            lr_foundation (float): Learning rate for foundation model
            lr_adapter (float): Learning rate for Bayesian adapter
            lr_others (float): Learning rate for other components
            
        Returns:
            dict: Training history
        """
        # Optimizer with different learning rates for different components
        foundation_params = list(self.model.foundation.parameters())
        adapter_params = list(self.model.bayesian_adapter.parameters())
        other_params = (list(self.model.shift_detector.parameters()) + 
                       list(self.model.calibration.parameters()))
        
        optimizer = torch.optim.AdamW([
            {'params': foundation_params, 'lr': lr_foundation},
            {'params': adapter_params, 'lr': lr_adapter},
            {'params': other_params, 'lr': lr_others}
        ], weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # KL annealing
            kl_weight = min(1.0, epoch / 50.0)
            
            # Training
            train_stats = self.train_epoch(train_loader, optimizer, kl_weight)
            
            # Validation
            val_stats = self.evaluate(val_loader)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_stats['loss'])
            history['train_acc'].append(train_stats['accuracy'])
            history['val_loss'].append(val_stats['loss'])
            history['val_acc'].append(val_stats['accuracy'])
            
            # Early stopping
            if val_stats['loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['loss']
                self.patience_counter = 0
                # Save best model
                best_model_path = os.path.join(self.save_dir, 'best_c2ba_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"✓ New best model saved at epoch {epoch+1}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Train Loss: {train_stats["loss"]:.4f}, Train Acc: {train_stats["accuracy"]:.2f}%')
                print(f'  Val Loss: {val_stats["loss"]:.4f}, Val Acc: {val_stats["accuracy"]:.2f}%')
                print(f'  Val F1: {val_stats["f1_score"]:.4f}')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_c2ba_model.pt')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"✓ Loaded best model from {best_model_path}")
        
        return history
    
    def save_model(self, path):
        """
        Save the current model state.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a saved model state.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        print(f"Model loaded from {path}")
    
    def get_uncertainty_estimates(self, data_loader, num_mc_samples=10):
        """
        Get uncertainty estimates for a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            num_mc_samples (int): Number of Monte Carlo samples
            
        Returns:
            dict: Uncertainty estimates and predictions
        """
        self.model.eval()
        all_uncertainties = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                
                # Multiple forward passes for uncertainty
                predictions = []
                for _ in range(num_mc_samples):
                    outputs = self.model(data, compute_uncertainty=False)
                    predictions.append(outputs['probabilities'].cpu().numpy())
                
                predictions = np.stack(predictions)  # [num_samples, batch_size, num_classes]
                
                # Compute uncertainty as prediction variance
                uncertainty = np.var(predictions, axis=0).mean(axis=1)  # [batch_size]
                mean_prediction = np.mean(predictions, axis=0)  # [batch_size, num_classes]
                
                all_uncertainties.extend(uncertainty)
                all_predictions.extend(mean_prediction)
                all_targets.extend(targets.numpy())
        
        return {
            'uncertainties': np.array(all_uncertainties),
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets)
        }
    
    def analyze_distribution_shift(self, train_loader, test_loader):
        """
        Analyze distribution shift between training and test data.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            
        Returns:
            dict: Distribution shift analysis results
        """
        self.model.eval()
        
        # Extract features from both datasets
        train_features = []
        test_features = []
        
        with torch.no_grad():
            # Training features
            for data, _ in train_loader:
                data = data.to(self.device)
                outputs = self.model(data, compute_uncertainty=False)
                train_features.append(outputs['features'].cpu())
            
            # Test features
            for data, _ in test_loader:
                data = data.to(self.device)
                outputs = self.model(data, compute_uncertainty=False)
                test_features.append(outputs['features'].cpu())
        
        train_features = torch.cat(train_features, dim=0)
        test_features = torch.cat(test_features, dim=0)
        
        # Compute shift metrics
        shift_detector = self.model.shift_detector
        mmd_score = shift_detector.compute_mmd(train_features, test_features).item()
        energy_distance = shift_detector.compute_energy_distance(train_features, test_features).item()
        
        return {
            'mmd_score': mmd_score,
            'energy_distance': energy_distance,
            'train_features': train_features,
            'test_features': test_features
        }

"""
Utility functions and metrics for C²BA model.

This module contains helper functions for evaluation metrics, random seed setting,
and other utilities used throughout the codebase.
"""

import numpy as np
import torch
import random
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_random_seeds(seed=42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def expected_calibration_error(y_prob, y_true, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_prob (np.ndarray): Predicted probabilities
            y_true (np.ndarray): True binary labels (0 or 1)
            n_bins (int): Number of bins for calibration
            
        Returns:
            float: Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def brier_score(y_prob, y_true, num_classes):
        """
        Calculate Brier Score.
        
        Args:
            y_prob (np.ndarray): Predicted probabilities [n_samples, n_classes]
            y_true (np.ndarray): True class labels
            num_classes (int): Number of classes
            
        Returns:
            float: Brier Score
        """
        y_true_onehot = np.eye(num_classes)[y_true]
        return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    
    @staticmethod
    def reliability_diagram_data(y_prob, y_true, n_bins=10):
        """
        Compute data for reliability diagram.
        
        Args:
            y_prob (np.ndarray): Predicted probabilities
            y_true (np.ndarray): True binary labels
            n_bins (int): Number of bins
            
        Returns:
            dict: Reliability diagram data
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = y_true[in_bin].mean()
                bin_confidence = y_prob[in_bin].mean()
                bin_count = in_bin.sum()
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': bin_accuracy,
                    'confidence': bin_confidence,
                    'count': bin_count
                })
        
        return bin_data
    
    @staticmethod
    def compute_comprehensive_metrics(y_prob, y_true, num_classes=2):
        """
        Compute all evaluation metrics.
        
        Args:
            y_prob (np.ndarray): Predicted probabilities [n_samples, n_classes]
            y_true (np.ndarray): True class labels
            num_classes (int): Number of classes
            
        Returns:
            dict: Comprehensive metrics
        """
        y_pred = np.argmax(y_prob, axis=1)
        y_prob_max = np.max(y_prob, axis=1)
        y_true_binary = (y_pred == y_true).astype(int)
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # For binary classification, compute AUC
        auc = None
        if num_classes == 2:
            try:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            except:
                auc = None
        
        # Calibration metrics
        ece = MetricsCalculator.expected_calibration_error(y_prob_max, y_true_binary)
        brier = MetricsCalculator.brier_score(y_prob, y_true, num_classes)
        
        # Confidence metrics
        avg_confidence = np.mean(y_prob_max)
        confidence_std = np.std(y_prob_max)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'ece': ece,
            'brier_score': brier,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std
        }
        
        if auc is not None:
            metrics['auc'] = auc
        
        return metrics
    
    @staticmethod
    def uncertainty_quality_metrics(uncertainties, correct_predictions):
        """
        Compute uncertainty quality metrics.
        
        Args:
            uncertainties (np.ndarray): Uncertainty estimates
            correct_predictions (np.ndarray): Binary array of correct predictions
            
        Returns:
            dict: Uncertainty quality metrics
        """
        # Correlation between uncertainty and errors
        errors = 1 - correct_predictions
        uncertainty_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
        
        # Area under the uncertainty-accuracy curve
        sorted_indices = np.argsort(uncertainties)[::-1]  # High to low uncertainty
        sorted_correct = correct_predictions[sorted_indices]
        
        # Compute accuracy at different uncertainty thresholds
        thresholds = np.percentile(uncertainties, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        threshold_accuracies = []
        
        for threshold in thresholds:
            high_uncertainty_mask = uncertainties >= threshold
            if high_uncertainty_mask.sum() > 0:
                accuracy = correct_predictions[high_uncertainty_mask].mean()
                threshold_accuracies.append(accuracy)
            else:
                threshold_accuracies.append(0.0)
        
        return {
            'uncertainty_error_correlation': uncertainty_error_corr,
            'threshold_accuracies': threshold_accuracies,
            'thresholds': thresholds
        }


def load_and_preprocess_data():
    """
    Load and preprocess UCI Heart Disease dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    from .data import load_heart_disease_data
    
    # Load the heart disease dataset
    df = load_heart_disease_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic statistics
    target_col = 'target' if 'target' in df.columns else 'num'
    if target_col in df.columns:
        print(f"Target distribution: {df[target_col].value_counts().sort_index().to_dict()}")
    
    # If we have a multi-class target (num column), convert to binary for this implementation
    if 'num' in df.columns and df['num'].max() > 1:
        # Convert multi-class to binary (0 = no disease, 1+ = disease)
        df['target'] = (df['num'] > 0).astype(int)
        print(f"Converted to binary classification. New target distribution: {df['target'].value_counts().to_dict()}")
    elif 'target' not in df.columns:
        # If no target column exists, create one for demonstration
        print("No target column found, creating synthetic target...")
        np.random.seed(42)
        # Create realistic target based on some features
        if 'age' in df.columns and 'cp' in df.columns:
            risk_score = (df['age'] - df['age'].mean()) / df['age'].std()
            if 'cp' in df.columns:
                risk_score += 0.5 * (df['cp'] == df['cp'].mode()[0])
            df['target'] = (risk_score > 0).astype(int)
        else:
            df['target'] = np.random.binomial(1, 0.45, len(df))
        print(f"Created synthetic target distribution: {df['target'].value_counts().to_dict()}")
    
    return df


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    
    # Print module-wise parameter counts
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s}: {module_params:8,} parameters")
    
    print("=" * 60)


def create_results_summary(val_metrics, test_metrics, shift_analysis=None):
    """
    Create a comprehensive results summary.
    
    Args:
        val_metrics (dict): Validation metrics
        test_metrics (dict): Test metrics
        shift_analysis (dict, optional): Distribution shift analysis
        
    Returns:
        dict: Results summary
    """
    summary = {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'performance_degradation': {
            'accuracy_drop': val_metrics['accuracy'] - test_metrics['accuracy'],
            'f1_drop': val_metrics['f1_score'] - test_metrics['f1_score'],
            'calibration_degradation': test_metrics['ece'] - val_metrics['ece']
        }
    }
    
    if shift_analysis:
        summary['distribution_shift'] = {
            'mmd_score': shift_analysis['mmd_score'],
            'energy_distance': shift_analysis['energy_distance']
        }
    
    return summary


def print_results_summary(summary):
    """
    Print a formatted results summary.
    
    Args:
        summary (dict): Results summary from create_results_summary
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Validation metrics
    print("\nVALIDATION METRICS:")
    val_metrics = summary['validation_metrics']
    for metric, value in val_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Test metrics
    print("\nTEST METRICS:")
    test_metrics = summary['test_metrics']
    for metric, value in test_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Performance degradation
    print("\nPERFORMANCE DEGRADATION:")
    degradation = summary['performance_degradation']
    for metric, value in degradation.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Distribution shift (if available)
    if 'distribution_shift' in summary:
        print("\nDISTRIBUTION SHIFT ANALYSIS:")
        shift = summary['distribution_shift']
        for metric, value in shift.items():
            print(f"  {metric:20s}: {value:.6f}")
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    accuracy_drop = degradation['accuracy_drop']
    calibration_deg = degradation['calibration_degradation']
    
    if accuracy_drop < 5.0:  # Less than 5% drop
        print("  ✓ Model shows good robustness to distribution shift")
    else:
        print("  ⚠ Significant performance degradation detected")
    
    if test_metrics['ece'] < 0.1:  # Well-calibrated
        print("  ✓ Model maintains good calibration")
    else:
        print("  ⚠ Calibration degraded under distribution shift")
    
    print("=" * 80)

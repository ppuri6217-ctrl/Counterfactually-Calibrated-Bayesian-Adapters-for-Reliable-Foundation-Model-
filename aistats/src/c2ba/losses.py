"""
Loss functions for C²BA model.

This module contains custom loss functions including Focal Loss for handling
class imbalance and other specialized loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for rare class (default: 1.0)
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits [N, C]
            targets (torch.Tensor): Ground truth labels [N]
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization.
    """
    
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes (int): Number of classes
            smoothing (float): Smoothing parameter (default: 0.1)
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute label smoothing loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits [N, C]
            targets (torch.Tensor): Ground truth labels [N]
            
        Returns:
            torch.Tensor: Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -smooth_targets * log_probs
        
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)


class CalibrationLoss(nn.Module):
    """
    Calibration loss to improve model calibration.
    """
    
    def __init__(self, n_bins=10, reduction='mean'):
        """
        Initialize Calibration Loss.
        
        Args:
            n_bins (int): Number of bins for calibration (default: 10)
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super(CalibrationLoss, self).__init__()
        self.n_bins = n_bins
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Compute calibration loss.
        
        Args:
            logits (torch.Tensor): Predicted logits [N, C]
            targets (torch.Tensor): Ground truth labels [N]
            
        Returns:
            torch.Tensor: Calibration loss value
        """
        probs = F.softmax(logits, dim=1)
        confidences = torch.max(probs, dim=1)[0]
        predictions = torch.argmax(probs, dim=1)
        accuracies = (predictions == targets).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                calibration_error += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return calibration_error


class UncertaintyLoss(nn.Module):
    """
    Uncertainty-aware loss that combines prediction loss with uncertainty regularization.
    """
    
    def __init__(self, base_loss='cross_entropy', uncertainty_weight=0.1):
        """
        Initialize Uncertainty Loss.
        
        Args:
            base_loss (str): Base loss function ('cross_entropy', 'focal')
            uncertainty_weight (float): Weight for uncertainty regularization
        """
        super(UncertaintyLoss, self).__init__()
        self.uncertainty_weight = uncertainty_weight
        
        if base_loss == 'cross_entropy':
            self.base_loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif base_loss == 'focal':
            self.base_loss_fn = FocalLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(self, logits, targets, uncertainties=None):
        """
        Compute uncertainty-aware loss.
        
        Args:
            logits (torch.Tensor): Predicted logits [N, C]
            targets (torch.Tensor): Ground truth labels [N]
            uncertainties (torch.Tensor, optional): Uncertainty estimates [N]
            
        Returns:
            torch.Tensor: Uncertainty-aware loss value
        """
        base_loss = self.base_loss_fn(logits, targets)
        
        if uncertainties is not None:
            # Weight loss by inverse uncertainty (more certain predictions get higher weight)
            weights = 1.0 / (1.0 + uncertainties)
            weighted_loss = base_loss * weights
            
            # Add uncertainty regularization (encourage appropriate uncertainty)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            correct = (predictions == targets).float()
            
            # Uncertainty should be high for incorrect predictions
            uncertainty_reg = torch.mean((1 - correct) * torch.exp(-uncertainties) + 
                                       correct * uncertainties)
            
            total_loss = weighted_loss.mean() + self.uncertainty_weight * uncertainty_reg
        else:
            total_loss = base_loss.mean()
        
        return total_loss


class DistributionShiftLoss(nn.Module):
    """
    Loss function for distribution shift detection.
    """
    
    def __init__(self, shift_weight=1.0):
        """
        Initialize Distribution Shift Loss.
        
        Args:
            shift_weight (float): Weight for shift detection loss
        """
        super(DistributionShiftLoss, self).__init__()
        self.shift_weight = shift_weight
    
    def forward(self, shift_logits, shift_labels):
        """
        Compute distribution shift detection loss.
        
        Args:
            shift_logits (torch.Tensor): Shift detection logits [N]
            shift_labels (torch.Tensor): Shift labels (0=source, 1=target) [N]
            
        Returns:
            torch.Tensor: Shift detection loss
        """
        return self.shift_weight * F.binary_cross_entropy_with_logits(
            shift_logits, shift_labels.float()
        )

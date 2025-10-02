"""
Neural network models for C²BA (Counterfactually-Calibrated Bayesian Adapter).

This module contains the core neural network architectures including:
- FoundationModel: Deep tabular neural network with attention
- BayesianAdapter: Low-rank Bayesian linear layer with Horseshoe prior
- DistributionShiftDetector: Detect and quantify distribution shift
- CalibrationSystem: Temperature scaling with density ratio correction
- C2BAModel: Complete integrated model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for tabular data"""
    
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.W_o(attn_output)


class FoundationModel(nn.Module):
    """Deep tabular neural network with attention"""
    
    def __init__(self, input_dim, hidden_dims=[128, 96], output_dim=32):
        """
        Initialize the foundation model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dims (list): Hidden layer dimensions
            output_dim (int): Output feature dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input processing
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(0.1)
        
        # Projection layer for residual connection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Layer 1: Linear transformation with residual
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.layer1_bn = nn.BatchNorm1d(hidden_dims[0])
        
        # Layer 2: Multi-head attention
        self.attention = MultiHeadAttention(hidden_dims[0], n_heads=4)
        self.attn_norm = nn.LayerNorm(hidden_dims[0])
        
        # Layer 3: Compression layer
        self.layer3 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3_bn = nn.BatchNorm1d(hidden_dims[1])
        self.layer3_proj = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Final representation layer
        self.output_layer = nn.Linear(hidden_dims[1], output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the foundation model.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Output features
        """
        # Input processing
        x_bn = self.input_bn(x)
        x_drop = self.input_dropout(x_bn)
        
        # Layer 1 with residual connection
        h1 = F.relu(self.layer1_bn(self.layer1(x_drop)))
        h1 = h1 + self.input_proj(x_drop)
        
        # Layer 2: Multi-head attention
        h1_unsqueezed = h1.unsqueeze(1)  # Add sequence dimension
        attn_out = self.attention(h1_unsqueezed).squeeze(1)
        h2 = self.attn_norm(attn_out + h1)
        
        # Layer 3: Compression with residual
        h3 = F.relu(self.layer3_bn(self.layer3(h2)))
        h3 = h3 + self.layer3_proj(h2)
        
        # Final representation
        output = torch.tanh(self.output_layer(h3))
        
        return output


class BayesianAdapter(nn.Module):
    """Low-rank Bayesian linear layer with Horseshoe prior"""
    
    def __init__(self, input_dim, output_dim, rank=8):
        """
        Initialize the Bayesian adapter.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension  
            rank (int): Rank of the low-rank decomposition
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        
        # Variational parameters for U matrix
        self.U_mean = nn.Parameter(torch.randn(input_dim, rank) * 0.1)
        self.U_logvar = nn.Parameter(torch.ones(input_dim, rank) * (-2))
        
        # Variational parameters for V matrix  
        self.V_mean = nn.Parameter(torch.randn(output_dim, rank) * 0.1)
        self.V_logvar = nn.Parameter(torch.ones(output_dim, rank) * (-2))
        
        # Global and local shrinkage parameters
        self.tau_mean = nn.Parameter(torch.tensor(0.0))
        self.tau_logvar = nn.Parameter(torch.tensor(-1.0))
        
        self.lambda_mean = nn.Parameter(torch.zeros(rank))
        self.lambda_logvar = nn.Parameter(torch.ones(rank) * (-1))
        
    def sample_weights(self, num_samples=1):
        """
        Sample weight matrices using reparameterization trick.
        
        Args:
            num_samples (int): Number of weight samples to draw
            
        Returns:
            torch.Tensor: Sampled weight matrices
        """
        # Sample shrinkage parameters
        tau_std = torch.exp(0.5 * self.tau_logvar)
        tau_samples = self.tau_mean + tau_std * torch.randn(num_samples, device=self.tau_mean.device)
        
        lambda_std = torch.exp(0.5 * self.lambda_logvar)
        lambda_samples = self.lambda_mean + lambda_std * torch.randn(num_samples, self.rank, device=self.lambda_mean.device)
        
        # Compute effective variances
        tau_expanded = tau_samples.unsqueeze(-1)  # [num_samples, 1]
        lambda_expanded = lambda_samples  # [num_samples, rank]
        effective_var = tau_expanded * lambda_expanded  # [num_samples, rank]
        
        weights = []
        for i in range(num_samples):
            # Sample U and V matrices
            U_std = torch.exp(0.5 * self.U_logvar) * effective_var[i]
            V_std = torch.exp(0.5 * self.V_logvar) * effective_var[i]
            
            U_sample = self.U_mean + U_std * torch.randn_like(self.U_mean)
            V_sample = self.V_mean + V_std * torch.randn_like(self.V_mean)
            
            # Compute weight matrix W = UV^T
            W_sample = torch.mm(U_sample, V_sample.t())
            weights.append(W_sample)
        
        return torch.stack(weights)
    
    def kl_divergence(self):
        """
        Compute KL divergence with Horseshoe prior.
        
        Returns:
            torch.Tensor: KL divergence value
        """
        # KL for tau (global shrinkage)
        tau_var = torch.exp(self.tau_logvar)
        kl_tau = 0.5 * (tau_var + self.tau_mean**2 - 1 - self.tau_logvar)
        
        # KL for lambda (local shrinkage)
        lambda_var = torch.exp(self.lambda_logvar)
        kl_lambda = 0.5 * torch.sum(lambda_var + self.lambda_mean**2 - 1 - self.lambda_logvar)
        
        # KL for U and V matrices (approximated with unit variance prior)
        U_var = torch.exp(self.U_logvar)
        V_var = torch.exp(self.V_logvar)
        
        kl_U = 0.5 * torch.sum(U_var + self.U_mean**2 - 1 - self.U_logvar)
        kl_V = 0.5 * torch.sum(V_var + self.V_mean**2 - 1 - self.V_logvar)
        
        return kl_tau + kl_lambda + kl_U + kl_V
    
    def forward(self, x, num_samples=1):
        """
        Forward pass with Monte Carlo sampling.
        
        Args:
            x (torch.Tensor): Input features
            num_samples (int): Number of Monte Carlo samples
            
        Returns:
            torch.Tensor: Output logits
        """
        if self.training:
            weights = self.sample_weights(num_samples)
            outputs = []
            for i in range(num_samples):
                outputs.append(torch.mm(x, weights[i]))
            return torch.stack(outputs).mean(dim=0)
        else:
            # Use mean weights for inference
            W_mean = torch.mm(self.U_mean, self.V_mean.t())
            return torch.mm(x, W_mean)


class DistributionShiftDetector(nn.Module):
    """Detect and quantify distribution shift"""
    
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initialize the distribution shift detector.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Forward pass for shift detection.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Shift detection logits
        """
        return self.classifier(x)
    
    def compute_mmd(self, X_train, X_test, gamma=1.0):
        """
        Compute Maximum Mean Discrepancy.
        
        Args:
            X_train (torch.Tensor): Training features
            X_test (torch.Tensor): Test features
            gamma (float): RBF kernel parameter
            
        Returns:
            torch.Tensor: MMD value
        """
        n, m = X_train.size(0), X_test.size(0)
        
        # RBF kernel computation
        def rbf_kernel(X, Y, gamma):
            X_norm = (X**2).sum(1).view(-1, 1)
            Y_norm = (Y**2).sum(1).view(1, -1)
            K = torch.exp(-gamma * (X_norm + Y_norm - 2 * torch.mm(X, Y.t())))
            return K
        
        # MMD computation
        Kxx = rbf_kernel(X_train, X_train, gamma)
        Kyy = rbf_kernel(X_test, X_test, gamma)
        Kxy = rbf_kernel(X_train, X_test, gamma)
        
        mmd_squared = (Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 
                      2 * Kxy.sum() / (n * m))
        
        return torch.sqrt(torch.clamp(mmd_squared, min=1e-8))
    
    def compute_energy_distance(self, X_train, X_test):
        """
        Compute energy distance between distributions.
        
        Args:
            X_train (torch.Tensor): Training features
            X_test (torch.Tensor): Test features
            
        Returns:
            torch.Tensor: Energy distance
        """
        n, m = X_train.size(0), X_test.size(0)
        
        # Pairwise distances
        def pairwise_distances(X, Y):
            return torch.cdist(X, Y, p=2)
        
        # Energy distance components
        d_xy = pairwise_distances(X_train, X_test).mean()
        d_xx = pairwise_distances(X_train, X_train).mean()
        d_yy = pairwise_distances(X_test, X_test).mean()
        
        return 2 * d_xy - d_xx - d_yy


class CalibrationSystem(nn.Module):
    """Temperature scaling with density ratio correction"""
    
    def __init__(self, num_classes=2):
        """
        Initialize the calibration system.
        
        Args:
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(1))
        self.density_ratio_weight = nn.Parameter(torch.tensor(0.1))
        
    def temperature_scale(self, logits):
        """
        Apply temperature scaling.
        
        Args:
            logits (torch.Tensor): Input logits
            
        Returns:
            torch.Tensor: Temperature-scaled logits
        """
        return logits / self.temperature
    
    def forward(self, logits, density_ratios=None):
        """
        Apply calibration with optional density ratio correction.
        
        Args:
            logits (torch.Tensor): Input logits
            density_ratios (torch.Tensor, optional): Density ratios for correction
            
        Returns:
            torch.Tensor: Calibrated probabilities
        """
        scaled_logits = self.temperature_scale(logits)
        
        if density_ratios is not None and self.training:
            # Apply density ratio correction
            correction = self.density_ratio_weight * torch.log(density_ratios + 1e-8)
            scaled_logits = scaled_logits + correction.unsqueeze(-1)
        
        return F.softmax(scaled_logits, dim=-1)


class C2BAModel(nn.Module):
    """Complete Counterfactually-Calibrated Bayesian Adapter model"""
    
    def __init__(self, input_dim, num_classes=2, foundation_dim=32, adapter_rank=8):
        """
        Initialize the complete C2BA model.
        
        Args:
            input_dim (int): Input feature dimension
            num_classes (int): Number of output classes
            foundation_dim (int): Foundation model output dimension
            adapter_rank (int): Rank for Bayesian adapter
        """
        super().__init__()
        self.num_classes = num_classes
        self.foundation = FoundationModel(input_dim, output_dim=foundation_dim)
        self.bayesian_adapter = BayesianAdapter(foundation_dim, num_classes, rank=adapter_rank)
        self.shift_detector = DistributionShiftDetector(foundation_dim)
        self.calibration = CalibrationSystem(num_classes)
        
        # Store training data statistics for shift detection
        self.register_buffer('train_features_mean', torch.zeros(foundation_dim))
        self.register_buffer('train_features_std', torch.ones(foundation_dim))
        
    def forward(self, x, compute_uncertainty=True, num_mc_samples=5):
        """
        Forward pass through the complete model.
        
        Args:
            x (torch.Tensor): Input features
            compute_uncertainty (bool): Whether to compute uncertainty estimates
            num_mc_samples (int): Number of Monte Carlo samples for uncertainty
            
        Returns:
            dict: Dictionary containing model outputs
        """
        # Extract foundation features
        features = self.foundation(x)
        
        if compute_uncertainty and self.training:
            # Multiple forward passes for uncertainty estimation
            predictions = []
            for _ in range(num_mc_samples):
                logits = self.bayesian_adapter(features, num_samples=1)
                predictions.append(logits)
            
            logits = torch.stack(predictions).mean(dim=0)
            uncertainty = torch.stack(predictions).std(dim=0).mean(dim=-1)
        else:
            logits = self.bayesian_adapter(features)
            uncertainty = None
        
        # Detect distribution shift
        shift_score = self.shift_detector(features).sigmoid()
        
        # Apply calibration
        calibrated_probs = self.calibration(logits, shift_score)
        
        return {
            'logits': logits,
            'probabilities': calibrated_probs,
            'features': features,
            'shift_score': shift_score,
            'uncertainty': uncertainty
        }
    
    def compute_loss(self, outputs, targets, train_features=None):
        """
        Compute total loss including ELBO and calibration terms.
        
        Args:
            outputs (dict): Model outputs
            targets (torch.Tensor): Target labels
            train_features (torch.Tensor, optional): Training features for shift detection
            
        Returns:
            dict: Dictionary containing loss components
        """
        logits = outputs['logits']
        shift_scores = outputs['shift_score']
        
        # Classification loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # Bayesian adapter KL divergence
        kl_loss = self.bayesian_adapter.kl_divergence()
        
        # Shift detection loss (when training data is available)
        shift_loss = torch.tensor(0.0, device=logits.device)
        if train_features is not None:
            # Create labels for shift detection
            batch_size = logits.size(0)
            train_batch_size = train_features.size(0)
            
            shift_labels = torch.cat([
                torch.zeros(train_batch_size),  # training data
                torch.ones(batch_size)  # current batch (potentially shifted)
            ]).to(logits.device)
            
            combined_features = torch.cat([train_features, outputs['features']], dim=0)
            shift_logits = self.shift_detector(combined_features).squeeze()
            shift_loss = F.binary_cross_entropy_with_logits(shift_logits, shift_labels)
        
        # Total loss with weighting
        total_loss = ce_loss + 0.01 * kl_loss + 0.5 * shift_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'shift_loss': shift_loss
        }

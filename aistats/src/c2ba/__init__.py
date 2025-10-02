"""
C²BA: Counterfactually-Calibrated Bayesian Adapter

A neural Darwinism framework for multimodal AI architectures that evolve to learn.
This package implements evolutionary cross-modal routing with Bayesian adaptation
for robust classification under distribution shift.
"""

__version__ = "1.0.0"
__author__ = "Neural Darwinism Research Team"

from .data import HeartDiseaseDataset, DataProcessor
from .models import (
    FoundationModel,
    BayesianAdapter, 
    DistributionShiftDetector,
    CalibrationSystem,
    C2BAModel
)
from .trainer import ModelTrainer
from .utils import MetricsCalculator, set_random_seeds
from .losses import FocalLoss

__all__ = [
    "HeartDiseaseDataset",
    "DataProcessor", 
    "FoundationModel",
    "BayesianAdapter",
    "DistributionShiftDetector", 
    "CalibrationSystem",
    "C2BAModel",
    "ModelTrainer",
    "MetricsCalculator",
    "FocalLoss",
    "set_random_seeds"
]

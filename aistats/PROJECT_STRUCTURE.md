# C²BA Project Structure

This document provides an overview of the complete project structure for the C²BA (Counterfactually-Calibrated Bayesian Adapter) implementation.

## 📁 Directory Structure

```
c2ba/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 CHANGELOG.md                 # Version history and changes
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 pyproject.toml              # Modern Python packaging config
├── 📄 setup.py                    # Legacy setup script
├── 📄 Dockerfile                  # Container configuration
├── 📄 .gitignore                  # Git ignore patterns
│
├── 📂 src/c2ba/                   # Main Python package
│   ├── 📄 __init__.py            # Package initialization and exports
│   ├── 📄 data.py                # Data processing and dataset classes
│   ├── 📄 models.py              # Neural network architectures
│   ├── 📄 trainer.py             # Training pipeline and utilities
│   ├── 📄 utils.py               # Utility functions and metrics
│   └── 📄 losses.py              # Custom loss functions
│
├── 📂 scripts/                   # Executable scripts
│   ├── 📄 train.py              # Main training script (supports DDP)
│   └── 📄 eval.py               # Evaluation and analysis script
│
├── 📂 configs/                   # Configuration files
│   ├── 📄 default.yaml          # Default single-GPU configuration
│   ├── 📄 ddp_2xa100.yaml       # Distributed training (2×A100)
│   └── 📄 quick_test.yaml       # Quick testing configuration
│
├── 📂 notebooks/                 # Jupyter notebooks and demos
│   ├── 📄 demo.py               # Minimal demo script
│   └── 📄 .gitkeep              # Keep directory in git
│
├── 📂 data/                      # Dataset directory
│   └── 📄 .gitkeep              # Keep directory in git
│
└── 📂 Original Files/            # Original research materials
    ├── 📄 AISTATS_Bayesian.pdf  # Research paper (PDF)
    └── 📄 notebookf797675715.ipynb # Original Jupyter notebook
```

## 🧩 Module Breakdown

### Core Package (`src/c2ba/`)

#### `__init__.py`
- Package initialization
- Public API exports
- Version information

#### `data.py`
- `HeartDiseaseDataset`: PyTorch dataset class
- `DataProcessor`: Comprehensive preprocessing pipeline
- `load_heart_disease_data()`: Dataset loading utilities
- `create_distribution_shifts()`: Shift simulation functions

#### `models.py`
- `MultiHeadAttention`: Attention mechanism for tabular data
- `FoundationModel`: Deep neural network with attention
- `BayesianAdapter`: Low-rank Bayesian linear layer
- `DistributionShiftDetector`: MMD and energy distance computation
- `CalibrationSystem`: Temperature scaling with corrections
- `C2BAModel`: Complete integrated architecture

#### `trainer.py`
- `ModelTrainer`: Comprehensive training pipeline
- Early stopping and checkpointing
- Distributed training support
- Uncertainty estimation utilities
- Distribution shift analysis

#### `utils.py`
- `MetricsCalculator`: Comprehensive evaluation metrics
- `set_random_seeds()`: Reproducibility utilities
- Calibration and uncertainty analysis functions
- Results formatting and visualization helpers

#### `losses.py`
- `FocalLoss`: Class imbalance handling
- `LabelSmoothingLoss`: Regularization technique
- `CalibrationLoss`: Calibration improvement
- `UncertaintyLoss`: Uncertainty-aware training
- `DistributionShiftLoss`: Shift detection training

### Scripts (`scripts/`)

#### `train.py`
- Command-line training interface
- Configuration loading and override system
- Single and distributed GPU support
- Comprehensive logging and checkpointing
- Integration with all model components

#### `eval.py`
- Model evaluation and analysis
- Uncertainty quantification assessment
- Distribution shift analysis
- Visualization generation (reliability diagrams, etc.)
- Results export (JSON, plots)

### Configuration (`configs/`)

#### `default.yaml`
- Baseline single-GPU configuration
- Moderate model size and training parameters
- Suitable for development and testing

#### `ddp_2xa100.yaml`
- Optimized for 2×A100 GPU training
- Larger batch sizes and model dimensions
- Advanced optimization settings

#### `quick_test.yaml`
- Minimal configuration for rapid iteration
- Small model and short training
- Debugging and development use

## 🔧 Key Features

### Modular Architecture
- Clean separation of concerns
- Extensible design patterns
- Easy to modify and extend

### Professional Standards
- Comprehensive documentation
- Type hints and docstrings
- Linting and formatting compliance
- Test-ready structure

### Reproducibility
- Deterministic random seeding
- Configuration-driven experiments
- Docker containerization
- Version-controlled dependencies

### Scalability
- Distributed training support
- Efficient data loading
- Memory-optimized implementations
- GPU acceleration throughout

## 🚀 Usage Patterns

### Development Workflow
1. Modify code in `src/c2ba/`
2. Test with `notebooks/demo.py`
3. Full training with `scripts/train.py`
4. Evaluation with `scripts/eval.py`

### Experiment Management
1. Create new config in `configs/`
2. Run training with specific config
3. Analyze results with evaluation script
4. Document findings in notebooks

### Production Deployment
1. Build Docker container
2. Use distributed training configs
3. Monitor with comprehensive metrics
4. Export models and results

## 📊 Data Flow

```
Raw Data → DataProcessor → HeartDiseaseDataset → DataLoader
    ↓
FoundationModel → BayesianAdapter → CalibrationSystem
    ↓
Loss Computation ← DistributionShiftDetector
    ↓
Optimizer → Model Updates → Checkpointing
    ↓
Evaluation → Metrics → Results Export
```

## 🔬 Research Integration

The modular structure facilitates research by:

- **Easy experimentation**: Swap components independently
- **Ablation studies**: Disable specific modules via config
- **Extension**: Add new models, losses, or metrics
- **Comparison**: Benchmark against baseline implementations

## 🛠️ Maintenance

### Code Quality
- All modules include comprehensive docstrings
- Type hints for better IDE support
- Consistent naming conventions
- Error handling and validation

### Testing Strategy
- Unit tests for individual components
- Integration tests for full pipeline
- Configuration validation
- Reproducibility verification

### Documentation
- README for quick start
- CONTRIBUTING for development
- CHANGELOG for version tracking
- Inline documentation for complex logic

This structure provides a solid foundation for both research and production use of the C²BA framework.

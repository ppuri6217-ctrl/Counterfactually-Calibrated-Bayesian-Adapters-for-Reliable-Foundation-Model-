# Counterfactually-Calibrated Bayesian Adapters (C²BA)

A scalable framework for reliable foundation model adaptation under distribution shifts, combining Bayesian low-rank adaptation with counterfactual calibration for robust uncertainty quantification in high-stakes applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Foundation models are increasingly deployed in high-stakes domains such as healthcare, autonomous systems, and policy-making, where robust and trustworthy decision-making under distributional and interventional shifts is critical. However, conventional adaptation techniques—including deterministic parameter-efficient fine-tuning (e.g., LoRA), Bayesian LoRA, MC dropout, and post-hoc calibration—often suffer from miscalibration, leading to unreliable uncertainty estimates and suboptimal decisions.

This repository implements **Counterfactually-Calibrated Bayesian Adapters (C²BA)**, a scalable framework that combines Bayesian low-rank adaptation with counterfactual calibration to improve the reliability of foundation models under both covariate and intervention-induced shifts.

## Key Contributions

### 1. Bayesian Adapter Parameterization
We introduce hierarchical priors over low-rank adapter weights, which strike a balance between expressivity and efficiency. This design enables fast and flexible posterior updates while avoiding the parameter blow-up of full Bayesian fine-tuning.

### 2. Scalable Variational Inference
To make Bayesian adaptation feasible in large-scale foundation models, we employ structured variational approximations coupled with natural-gradient optimization. This ensures both tractability and stability in training without sacrificing uncertainty quality.

### 3. Counterfactual Calibration
Beyond parameter updates, we correct miscalibration induced by interventional and temporal distribution shifts. A causal calibration layer reweights posteriors via density ratios and influence functions, producing predictions that remain well-calibrated even under challenging shifts.

## Methodology

### Framework Overview

C²BA integrates three complementary elements to address the limitations of existing foundation model adaptation methods:

**Hierarchical Bayesian Adapters**: Low-rank adapter weights with structured priors that maintain computational efficiency while providing principled uncertainty quantification.

**Natural Gradient Variational Inference**: Scalable posterior approximation using structured variational families with natural gradient optimization for stable and efficient training.

**Counterfactual Calibration Layer**: Causal correction mechanism that uses influence functions and density ratio reweighting to maintain calibration under interventional shifts.

### Technical Approach

The framework addresses four critical aspects of reliable foundation model adaptation:

- **Parameter Efficiency**: Reduces memory and compute overhead for deployment in resource-constrained scenarios
- **Bayesian Uncertainty**: Provides principled confidence estimates critical for risk-sensitive applications  
- **Post-hoc Calibration**: Corrects miscalibrated predictions, enhancing reliability under distributional shifts
- **Causal/Interventional Robustness**: Ensures models remain reliable under interventions or distributional changes

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy, pandas, scikit-learn
- PyYAML for configuration management

### Setup

```bash
git clone https://github.com/your-org/c2ba.git
cd c2ba

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Usage

### Quick Start

```bash
# Run demonstration
python notebooks/demo.py

# Basic training
python scripts/train.py --config configs/default.yaml

# Evaluation with calibration analysis
python scripts/eval.py --config configs/default.yaml --ckpt checkpoints/best_c2ba_model.pt
```

### Configuration

The framework uses YAML configuration files for experiment management:

```yaml
model:
  input_dim: null  # Automatically determined
  num_classes: 2
  foundation_dim: 32
  adapter_rank: 8
  hierarchical_priors: true

training:
  num_epochs: 100
  patience: 15
  natural_gradient: true

calibration:
  counterfactual_correction: true
  density_ratio_weight: 0.1
  influence_regularization: 0.05

optim:
  lr_foundation: 1.0e-3
  lr_adapter: 5.0e-3
  natural_gradient_lr: 1.0e-2
```

### Distributed Training

For large-scale foundation model adaptation:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml
```

## Theoretical Guarantees

C²BA provides formal guarantees on:

- **Posterior Contraction**: Convergence properties of the Bayesian adapter posteriors
- **Calibration Error Bounds**: Explicit bounds on calibration error under interventional shifts
- **Decision-Making Quality**: Direct links between uncertainty calibration and downstream decision performance

## Evaluation Framework

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Primary metric for probability calibration assessment
- **Reliability Diagrams**: Visual analysis of confidence vs accuracy alignment
- **Brier Score Decomposition**: Proper scoring with uncertainty and calibration components

### Uncertainty Quality
- **Predictive Entropy**: Information-theoretic uncertainty measurement
- **Epistemic vs Aleatoric**: Decomposition of uncertainty sources
- **Influence Function Analysis**: Sensitivity to training data perturbations

### Robustness Assessment
- **Covariate Shift**: Performance under input distribution changes
- **Interventional Shift**: Robustness to causal mechanism changes
- **Temporal Shift**: Adaptation to non-stationary environments

## Advanced Usage

### Bayesian Adapter Configuration

```python
from c2ba import C2BAModel

model = C2BAModel(
    input_dim=768,  # Foundation model dimension
    num_classes=2,
    adapter_rank=16,
    hierarchical_priors=True,
    natural_gradient_vi=True
)
```

### Counterfactual Calibration

```python
# Enable counterfactual calibration
calibration_config = {
    'density_ratio_correction': True,
    'influence_regularization': 0.05,
    'causal_reweighting': True
}

trainer = ModelTrainer(model, calibration_config=calibration_config)
```

### Uncertainty Analysis

```python
# Comprehensive uncertainty evaluation
uncertainty_results = trainer.evaluate_uncertainty(
    test_loader, 
    num_samples=50,  # Monte Carlo samples
    decompose_uncertainty=True
)

print(f"Epistemic Uncertainty: {uncertainty_results['epistemic'].mean():.4f}")
print(f"Aleatoric Uncertainty: {uncertainty_results['aleatoric'].mean():.4f}")
```

## Benchmarks and Applications

The framework has been validated on:

- **Healthcare**: Medical diagnosis and triage systems (MIMIC-III)
- **Autonomous Systems**: Self-driving car decision-making (nuScenes)
- **Policy Adaptation**: Dynamic policy optimization under changing environments

## Docker Support

### Standard Training
```bash
docker build -t c2ba:latest .
docker run -it --rm -v $(pwd):/workspace c2ba:latest
```

### GPU-Accelerated Training
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace c2ba:latest \
  python scripts/train.py --config configs/ddp_2xa100.yaml
```

## Repository Structure

```
c2ba/
├── src/c2ba/                 # Main package
│   ├── data.py              # Data processing and shift simulation
│   ├── models.py            # Bayesian adapters and calibration layers
│   ├── trainer.py           # Training with natural gradients
│   ├── utils.py             # Calibration metrics and utilities
│   ├── losses.py            # Variational and calibration losses
│   └── calibration.py       # Counterfactual calibration methods
├── scripts/                 # Execution scripts
│   ├── train.py            # Training with Bayesian inference
│   └── eval.py             # Calibration and uncertainty evaluation
├── configs/                 # Configuration files
│   ├── default.yaml        # Standard configuration
│   ├── ddp_2xa100.yaml     # Distributed training
│   └── calibration.yaml    # Calibration-focused setup
├── notebooks/               # Demonstrations and analysis
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## Contributing

We welcome contributions to advance reliable foundation model adaptation:

1. Fork the repository
2. Create a feature branch for your contribution
3. Implement changes with appropriate tests and documentation
4. Submit a pull request with detailed methodology description

### Development Guidelines

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/ scripts/

# Type checking
mypy src/
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{c2ba2024,
  title={Counterfactually-Calibrated Bayesian Adapters for Reliable Foundation Model Adaptation under Distribution Shifts},
  author={Anonymous Author},
  journal={Proceedings of AISTATS},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions regarding the methodology or implementation:
- GitHub Issues for technical questions and bug reports
- Email: research@anonymous-institution.edu for research inquiries

## Reproducibility and Code Availability

### Source Code and Dependencies
All source code is provided in this repository with complete dependency specifications. The implementation includes:
- Complete C²BA framework implementation in PyTorch
- All training and evaluation scripts
- Configuration files for different experimental setups
- Docker containers for reproducible environments
- Comprehensive documentation and examples

**Repository**: [GitHub Repository](https://github.com/your-org/c2ba)  
**Dependencies**: All external libraries specified in `requirements.txt` and `pyproject.toml`

### Experimental Reproducibility

#### Code and Data Availability
- **Training Code**: Complete training pipeline in `scripts/train.py` with distributed support
- **Evaluation Code**: Comprehensive evaluation framework in `scripts/eval.py`
- **Data Processing**: Full preprocessing pipeline in `src/c2ba/data.py`
- **Model Implementation**: Complete C²BA architecture in `src/c2ba/models.py`
- **Experimental Instructions**: Detailed setup and execution instructions provided

#### Training Details and Hyperparameters
- **Configuration Management**: All hyperparameters specified in YAML configuration files
- **Training Procedures**: Detailed training protocols in configuration files and documentation
- **Hyperparameter Selection**: Default configurations provided for different scenarios
- **Reproducibility**: Random seed management and deterministic training procedures

#### Evaluation Metrics and Statistics
- **Calibration Metrics**: Expected Calibration Error (ECE), Brier Score, Reliability Diagrams
- **Uncertainty Metrics**: Epistemic/Aleatoric decomposition, Predictive Entropy
- **Performance Metrics**: Accuracy, F1-Score, AUC-ROC with confidence intervals
- **Statistical Testing**: Multiple random seed evaluation for statistical significance
- **Error Analysis**: Comprehensive uncertainty and calibration analysis tools

#### Computing Infrastructure
- **Hardware Requirements**: Specifications for CPU and GPU training provided
- **Scalability**: Support for single GPU, multi-GPU, and distributed training
- **Docker Support**: Containerized environments for consistent execution
- **Cloud Deployment**: Instructions for cloud-based training and evaluation
- **Resource Estimation**: Memory and compute requirements documented

## Implementation Checklist

### ✅ Model and Algorithm Implementation
- [x] **Mathematical Framework**: Complete description of C²BA methodology in documentation
- [x] **Algorithm Implementation**: Full implementation of Bayesian adapters with hierarchical priors
- [x] **Complexity Analysis**: Computational complexity documented in code comments
- [x] **Source Code**: Complete anonymized source code provided with dependency specifications

### ✅ Empirical Results Reproducibility  
- [x] **Reproduction Instructions**: Complete code and instructions for reproducing results
- [x] **Training Details**: All hyperparameters, data splits, and training procedures documented
- [x] **Evaluation Metrics**: Clear definition of calibration and uncertainty metrics implemented
- [x] **Statistical Analysis**: Multiple seed evaluation and confidence interval computation
- [x] **Computing Infrastructure**: Hardware specifications and scalability options documented

### 📁 Repository Structure for Reproducibility
```
c2ba/
├── src/c2ba/                    # Complete implementation
│   ├── models.py               # ✅ C²BA architecture
│   ├── trainer.py              # ✅ Training procedures
│   ├── utils.py                # ✅ Evaluation metrics
│   └── calibration.py          # ✅ Counterfactual calibration
├── scripts/
│   ├── train.py                # ✅ Training reproduction
│   └── eval.py                 # ✅ Evaluation reproduction
├── configs/                    # ✅ All hyperparameters
├── requirements.txt            # ✅ Dependencies
├── Dockerfile                  # ✅ Reproducible environment
└── README.md                   # ✅ Complete documentation
```

### 🔬 Experimental Validation
- [x] **Baseline Comparisons**: Implementation supports comparison with LoRA, Bayesian LoRA
- [x] **Ablation Studies**: Modular design enables component-wise evaluation
- [x] **Cross-Validation**: Support for multiple evaluation protocols
- [x] **Statistical Significance**: Multiple seed evaluation framework
- [x] **Calibration Analysis**: Comprehensive calibration evaluation tools

## Acknowledgments

This work builds upon foundational research in Bayesian deep learning, causal inference, and uncertainty quantification. We acknowledge the research community's contributions to reliable machine learning and the importance of trustworthy AI in high-stakes applications.

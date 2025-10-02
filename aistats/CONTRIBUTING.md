# Contributing to C²BA

We welcome contributions to the C²BA (Counterfactually-Calibrated Bayesian Adapter) project! This document provides guidelines for contributing to the codebase.

## 🚀 Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/c2ba.git
   cd c2ba
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Set up pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## 🔧 Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

Run these tools before submitting:

```bash
# Format code
black src/ scripts/ tests/

# Check linting
flake8 src/ scripts/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add uncertainty calibration module
fix: resolve memory leak in training loop
docs: update installation instructions
test: add unit tests for BayesianAdapter
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=c2ba

# Run specific test file
pytest tests/test_models.py

# Run tests with specific markers
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_bayesian_adapter_forward_pass`
- Include both unit tests and integration tests
- Test edge cases and error conditions

Example test structure:

```python
import pytest
import torch
from c2ba.models import BayesianAdapter

class TestBayesianAdapter:
    def test_forward_pass(self):
        adapter = BayesianAdapter(input_dim=10, output_dim=2, rank=4)
        x = torch.randn(32, 10)
        output = adapter(x)
        assert output.shape == (32, 2)
    
    def test_kl_divergence(self):
        adapter = BayesianAdapter(input_dim=10, output_dim=2, rank=4)
        kl = adapter.kl_divergence()
        assert kl.item() >= 0
```

## 📝 Documentation

### Code Documentation

- Use clear docstrings for all public functions and classes
- Follow Google-style docstrings:

```python
def compute_loss(self, outputs, targets, train_features=None):
    """
    Compute total loss including ELBO and calibration terms.
    
    Args:
        outputs (dict): Model outputs containing logits and features
        targets (torch.Tensor): Ground truth labels
        train_features (torch.Tensor, optional): Training features for shift detection
        
    Returns:
        dict: Dictionary containing loss components
        
    Raises:
        ValueError: If outputs dictionary is missing required keys
    """
```

### README Updates

When adding new features:
- Update the README.md with usage examples
- Add new configuration options to the documentation
- Update the repository structure if needed

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - PyTorch version
   - Operating system
   - GPU information (if applicable)

2. **Reproduction steps**:
   - Minimal code example
   - Configuration files used
   - Expected vs actual behavior

3. **Error messages**:
   - Full stack trace
   - Log outputs

## ✨ Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the motivation** for the feature
3. **Provide implementation details** if possible
4. **Consider backward compatibility**

## 🔄 Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update CHANGELOG.md** if applicable
5. **Request review** from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
```

## 🏷️ Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: research@example.com for direct contact

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing to C²BA!

# Multi-stage Dockerfile for C²BA
# Supports both CPU and GPU training with CUDA

ARG CUDA_VERSION=11.8
ARG PYTHON_VERSION=3.10

# Base image with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode
COPY src/ src/
RUN pip install -e .

# Copy the rest of the code
COPY . .

# Create necessary directories
RUN mkdir -p data checkpoints outputs logs

# Set permissions
RUN chmod +x scripts/*.py

# Expose port for Jupyter (optional)
EXPOSE 8888

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]

# Development stage with additional tools
FROM base as dev

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    pytest \
    black \
    flake8 \
    mypy

# Install optional experiment tracking tools
RUN pip install --no-cache-dir \
    wandb \
    optuna \
    hydra-core \
    plotly \
    rich

# Start Jupyter by default in dev mode
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

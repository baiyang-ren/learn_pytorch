# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This repository uses a conda environment called `opencv-env` with Python 3.13.5. The environment includes:
- PyTorch 2.8.0 with CUDA 12.6 support
- torchvision 0.23.0 for computer vision datasets and transforms
- Additional packages: torchinfo, torchviz, matplotlib, numpy, jupyter

To activate the environment:
```bash
conda activate opencv-env
```

## Repository Structure

This is a PyTorch learning repository containing Jupyter notebooks that demonstrate core PyTorch concepts:

### Core Learning Notebooks
1. **PyTorch_for_Beginners.ipynb** - Introduction to PyTorch tensors, basic operations, and image preprocessing
2. **Autograd_and_Backpropagation.ipynb** - Automatic differentiation with torch.autograd and gradient computation
3. **MLP_Fashion_Mnist.ipynb** - Multi-layer perceptron implementation for Fashion-MNIST classification

### Data Organization
- `data/` - General data directory for downloaded datasets
- `F_MNIST_data/` - Fashion-MNIST dataset storage (automatically created by torchvision.datasets)
- `mnist_0.jpg`, `mnist_1.jpg` - Sample MNIST digit images for visualization examples

### Generated Files
- `grad_computation_graph.png` - Visualization of computational graphs created by torchviz
- `grad_computation_graph` - DOT file for graph visualization

## Development Workflow

### Running Notebooks
Launch Jupyter from the repository root:
```bash
jupyter notebook
# or
jupyter lab
```

The notebooks are designed to run sequentially and demonstrate progressively complex concepts.

### Working with PyTorch Datasets
- Datasets are automatically downloaded to appropriate subdirectories on first run
- Fashion-MNIST normalization is computed dynamically from the training set
- Mean and standard deviation values are calculated as: Mean: 0.2860, Std: 0.3530

### GPU Usage
The notebooks include CUDA availability checks and will automatically use GPU if available. The current environment includes CUDA 12.6 support.

## Key Learning Progression

1. **Tensor Operations**: Start with basic tensor creation, manipulation, and device management
2. **Autograd System**: Understanding automatic differentiation and computation graphs
3. **Dataset Handling**: Working with torchvision datasets, DataLoaders, and preprocessing
4. **Model Architecture**: Building multi-layer perceptrons with batch normalization and dropout
5. **Training Loop**: Implementing training/validation loops with proper metric tracking

## Code Patterns

### Dataset Normalization Pattern
The notebooks demonstrate computing dataset statistics dynamically:
```python
# Compute mean/std from raw dataset
all_pixels = torch.cat([img.view(-1) for img, _ in train_set_raw])
mean = all_pixels.mean().item()
std = all_pixels.std().item()
```

### Model Definition Pattern
Models follow PyTorch nn.Module conventions with:
- Initialization of layers in `__init__`
- Forward pass logic in `forward()`
- Use of functional API (F.relu, F.log_softmax) for activations
- Integration of batch normalization and dropout for regularization

### Training Configuration
Standard patterns include:
- Adam optimizer with learning rate 0.01
- Negative Log Likelihood Loss for classification
- Device-agnostic tensor operations
- Proper gradient zeroing and backpropagation steps
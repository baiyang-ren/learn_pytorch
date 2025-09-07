# MLP Fashion MNIST - Summary Class Notes

## Key Concepts Learned

### 1. Python Iteration Protocol
- **Map-style objects**: Use `__getitem__()` + `__len__()` (e.g., PyTorch datasets)
- **Iterable objects**: Use `__iter__()` method
- Python automatically falls back to sequence protocol when `__iter__()` is missing
- PyTorch datasets use `__getitem__()` sequentially, returning `(image, label)` tuples

### 2. Dataset Iteration Tools
```python
# Get single batch for testing
images, labels = next(iter(trainloader))

# iter() creates iterator, next() gets first item
# Commonly used for debugging/sampling
```

### 3. Neural Network Components

#### Dropout (p=0.3)
- **Purpose**: Regularization to prevent overfitting
- **Mechanism**: Randomly zeros 30% of neuron activations (NOT samples)
- **Training**: Active with scaling compensation
- **Inference**: Disabled completely

#### BatchNorm Parameters
- **γ (gamma)**: Learnable scale parameter per feature
- **β (beta)**: Learnable shift parameter per feature
- **Formula**: `output = γ * normalized_input + β`
- **Flexibility**: Can learn optimal mean/variance for each feature

### 4. Training vs Inference Modes

#### model.train() vs model.eval()
- **model.train()**: Sets `training=True`, enables dropout, uses batch statistics for BatchNorm
- **model.eval()**: Sets `training=False`, disables dropout, uses running statistics

#### torch.no_grad()
- **Purpose**: Disables gradient computation completely
- **Effect**: No computation graph built, saves memory
- **Scope**: Temporary within `with` block only

#### Best Practice
```python
# Inference
model.eval()           # Fix layer behaviors  
with torch.no_grad():  # Disable gradients
    outputs = model(x)
```

### 5. PyTorch Computation Graph

#### Graph Construction
- Built **incrementally** during forward pass
- Each operation creates nodes connected via `grad_fn`
- Loss tensor becomes the root of the complete graph

#### Graph Structure
```python
# Linked structure through grad_fn objects:
loss.grad_fn → outputs.grad_fn → fc4.grad_fn → ... → fc0.grad_fn
```

#### Backward Propagation
```python
loss.backward()  # Traverses graph backwards using chain rule
                # Populates param.grad for all parameters
```

### 6. Optimizer Relationship

#### Key Insight
- **Optimizer gets**: Parameter references via `model.parameters()`
- **Optimizer reads**: Gradients from `param.grad` after `loss.backward()`
- **Optimizer doesn't know**: The computation graph structure
- **Interface**: `param.grad` connects graph computation to parameter updates

#### Training Flow
```python
# 1. Forward pass builds graph
outputs = model(images)
loss = criterion(outputs, labels)

# 2. Backward pass computes gradients  
loss.backward()  # Writes to param.grad

# 3. Optimizer updates parameters
optimizer.step()  # Reads param.grad, updates param.data
optimizer.zero_grad()  # Clears gradients for next iteration
```

### 7. Validation vs Training

#### Validation Purpose
- **Evaluation only**: Monitors model performance
- **No learning**: No `backward()`, no `optimizer.step()`
- **Decision making**: Used for early stopping, model saving, hyperparameter tuning

#### Key Difference
- **Training**: Model learns and updates weights
- **Validation**: Model is tested but weights unchanged

## Important Patterns

### Context Managers (`with` statement)
- Creates **temporary effects** that auto-cleanup
- `torch.no_grad()` example: disables gradients temporarily
- Guarantees restoration even if exceptions occur

### Dataset Access Patterns  
```python
# Map-style: Random access by index
dataset[42]  # Direct access to 43rd item

# Sequential iteration (what for-loops do internally)
for i in range(len(dataset)):
    item = dataset[i]  # Calls __getitem__(i)
```

### Computation Graph Key Points
- **Built during**: Forward pass (every operation)
- **Used during**: Backward pass (gradient computation)
- **Stored in**: `grad_fn` attributes of tensors
- **Connected through**: Output tensors link loss to model parameters

### 8. Loss Functions and Numerical Stability

#### Log-Softmax Stability
- **Problem**: Regular softmax can underflow with large negative values
- **Solution**: `log_softmax(x) = x - log(sum(exp(x)))`
- **Max subtraction trick**: Subtract max value for numerical stability
- **Benefits**: No underflow, no division by zero, stable gradients

#### Negative Log Likelihood (NLL) Loss
```python
nll_loss = -log_probs[target_class]
```

**Key Properties:**
- **High confidence in correct class** → Low loss (e.g., P=0.9 → Loss=0.1)
- **Low confidence in correct class** → High loss (e.g., P=0.1 → Loss=2.3)
- **Perfect prediction** → Zero loss (P=1.0 → Loss=0.0)
- **Exponential penalty curve**: Small confidence decreases cause large loss increases

**Intuition**: "How surprised should I be that the model got it right?"
- Not surprised (high confidence) → Low loss
- Very surprised (low confidence) → High loss

## Summary
This conversation covered the fundamental mechanisms of PyTorch: how data flows through models, how gradients are computed and applied, the difference between training and inference modes, the separation of concerns between computation graphs and optimizers, and the mathematical foundations of loss functions. The key insight is that PyTorch creates a computational trace during forward passes that enables automatic differentiation during backward passes, while using numerically stable formulations like log-softmax and NLL loss to ensure reliable training.
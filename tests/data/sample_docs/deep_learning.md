# Deep Learning

## Introduction to Neural Networks

Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information using connectionist approaches.

## Basic Architecture

### Neurons (Perceptrons)

A single neuron computes:
```
output = activation(Σ(wᵢxᵢ) + b)
```

Where:
- xᵢ: inputs
- wᵢ: weights
- b: bias
- activation: non-linear function

### Layers

- **Input Layer**: Receives raw data
- **Hidden Layers**: Intermediate processing
- **Output Layer**: Produces final predictions

### Activation Functions

1. **Sigmoid**: σ(x) = 1/(1 + e^(-x))
   - Output: (0, 1)
   - Problem: vanishing gradients

2. **Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
   - Output: (-1, 1)
   - Zero-centered but still has vanishing gradient

3. **ReLU**: f(x) = max(0, x)
   - Most popular for hidden layers
   - Solves vanishing gradient
   - Problem: dying ReLU

4. **Leaky ReLU**: f(x) = max(αx, x)
   - Addresses dying ReLU problem

5. **Softmax**: For multi-class classification output

## Backpropagation

**Backpropagation** is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight.

### How Backpropagation Works

1. **Forward Pass**: Input propagates through network to produce output
2. **Compute Loss**: Compare output with target using loss function
3. **Backward Pass**: Compute gradients layer by layer using chain rule
4. **Update Weights**: Apply gradient descent to update weights

### The Chain Rule

For a composed function f(g(x)), the derivative is:
```
∂f/∂x = (∂f/∂g) × (∂g/∂x)
```

Backpropagation applies this recursively through all layers.

### Computational Graph

Modern frameworks (PyTorch, TensorFlow) build computational graphs that:
- Track operations during forward pass
- Enable automatic differentiation for backward pass

### Vanishing and Exploding Gradients

**Vanishing gradients**: Gradients become extremely small in early layers
- Causes: sigmoid/tanh activations, deep networks
- Solutions: ReLU, batch normalization, skip connections

**Exploding gradients**: Gradients become extremely large
- Solutions: gradient clipping, weight initialization (Xavier, He)

## Convolutional Neural Networks (CNN)

CNNs are specialized neural networks designed for processing grid-like data, particularly images.

### Why CNNs for Images?

Traditional neural networks:
- Too many parameters for images
- Don't exploit spatial structure
- Not translation invariant

CNNs address these through:
- Local connectivity
- Parameter sharing
- Pooling

### Convolutional Layer

The **convolutional layer** applies filters (kernels) that slide across the input to detect features.

Key concepts:
- **Filter/Kernel**: Small matrix (e.g., 3×3) that detects patterns
- **Stride**: Step size for sliding the filter
- **Padding**: Adding zeros around input to control output size
- **Feature Map**: Output of applying a filter

Convolution operation:
```
(I * K)[i,j] = Σₘ Σₙ I[i+m, j+n] × K[m, n]
```

### Pooling Layer

Pooling reduces spatial dimensions while retaining important information.

Types:
- **Max Pooling**: Takes maximum value in window
- **Average Pooling**: Takes average value in window

Benefits:
- Reduces computation
- Provides translation invariance
- Helps prevent overfitting

### CNN Architecture Pattern

Typical architecture:
```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

Famous architectures:
- **LeNet**: Pioneer CNN for digit recognition
- **AlexNet**: Won ImageNet 2012, popularized deep CNNs
- **VGGNet**: Simple architecture with small 3×3 filters
- **ResNet**: Introduced skip connections for very deep networks

## Recurrent Neural Networks (RNN)

RNNs are designed for sequential data where order matters (text, time series, audio).

### Basic RNN

At each timestep t:
```
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)
yₜ = Wₕᵧhₜ
```

The hidden state hₜ carries information from previous timesteps.

### Problems with Basic RNNs

- Vanishing gradients for long sequences
- Difficulty capturing long-range dependencies

### LSTM (Long Short-Term Memory)

LSTM uses gates to control information flow:
- **Forget Gate**: What to discard from cell state
- **Input Gate**: What new information to store
- **Output Gate**: What to output based on cell state

### GRU (Gated Recurrent Unit)

Simplified version of LSTM with fewer parameters:
- **Reset Gate**: Controls how much past information to forget
- **Update Gate**: Controls how much to update hidden state

## Transformers and Attention

### The Attention Mechanism

**Attention** allows models to focus on relevant parts of the input when producing output. It computes a weighted sum of values based on compatibility between query and keys.

The scaled dot-product attention:
```
Attention(Q, K, V) = softmax(QK^T / √dₖ)V
```

Where:
- **Q (Query)**: What we're looking for
- **K (Key)**: What we match against
- **V (Value)**: What we retrieve
- **dₖ**: Dimension of keys (for scaling)

### Self-Attention

In self-attention, Q, K, and V all come from the same sequence. This allows each position to attend to all other positions.

### Multi-Head Attention

Instead of single attention, use multiple parallel attention heads:
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᴼ
where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```

Benefits:
- Attends to information from different representation subspaces
- Different heads can capture different types of relationships

### Transformer Architecture

The Transformer uses attention without recurrence:
- **Encoder**: Processes input sequence
- **Decoder**: Generates output sequence

Key components:
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Positional encoding (since no inherent order)

### Why Transformers Dominate

- Parallelizable (unlike RNNs)
- Capture long-range dependencies effectively
- Scale well with data and compute
- Foundation for GPT, BERT, and modern LLMs

## Transfer Learning

**Transfer learning** uses knowledge learned from one task to improve performance on a related task.

### Why Transfer Learning?

- Requires less training data
- Faster training
- Better performance with limited data
- Leverages features learned from large datasets

### Common Approaches

1. **Feature Extraction**
   - Use pre-trained model as fixed feature extractor
   - Only train new classifier on top
   - Best when: limited data, similar domain

2. **Fine-tuning**
   - Start with pre-trained weights
   - Continue training on new task
   - Often freeze early layers, fine-tune later ones
   - Best when: more data available, different domain

### Transfer Learning in Practice

For images:
- Use ImageNet pre-trained models (ResNet, VGG, EfficientNet)
- Replace final classification layer
- Fine-tune if needed

For text:
- Use pre-trained language models (BERT, GPT)
- Fine-tune on downstream task
- Can achieve state-of-the-art with limited data

### When Transfer Learning Works Best

- Source and target tasks are related
- Source model trained on large, diverse dataset
- Target dataset is limited
- Low-level features (edges, textures) are transferable

# Machine Learning Fundamentals

## Introduction to Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions.

## Types of Learning

### Supervised Learning

Supervised learning uses **labeled data** to train models. The algorithm learns a mapping function from input variables (X) to output variables (Y). The goal is to approximate the mapping function well enough to predict Y for new X values.

Common supervised learning tasks:
- **Classification**: Predicting discrete labels (spam/not spam, cat/dog)
- **Regression**: Predicting continuous values (house prices, temperature)

Examples of supervised algorithms: Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines, Neural Networks.

### Unsupervised Learning

Unsupervised learning works with **unlabeled data**. The algorithm tries to find hidden patterns or intrinsic structures in the input data without any guidance.

Common unsupervised learning tasks:
- **Clustering**: Grouping similar data points (customer segmentation)
- **Dimensionality Reduction**: Reducing features while preserving information (PCA)
- **Association**: Finding rules that describe portions of data (market basket analysis)

### Semi-supervised and Reinforcement Learning

**Semi-supervised learning** combines labeled and unlabeled data, useful when labeling is expensive.

**Reinforcement learning** trains agents to make sequences of decisions by rewarding desired behaviors and punishing undesired ones.

## The Bias-Variance Tradeoff

The **bias-variance tradeoff** is a fundamental concept in machine learning that describes the balance between two sources of error:

### Bias

**Bias** is the error from overly simplistic assumptions in the learning algorithm. High bias causes the model to miss relevant relations between features and target outputs (underfitting).

- High bias = model is too simple
- Symptoms: poor training AND test performance
- Example: using linear regression for highly non-linear data

### Variance

**Variance** is the error from sensitivity to small fluctuations in the training set. High variance causes the model to model random noise in the training data (overfitting).

- High variance = model is too complex
- Symptoms: good training performance, poor test performance
- Example: deep decision tree that memorizes training data

### Finding the Balance

The total error = Bias² + Variance + Irreducible Error

The goal is to find the sweet spot:
- Simple models: high bias, low variance
- Complex models: low bias, high variance

Techniques to manage the tradeoff:
- Cross-validation for model selection
- Regularization (L1, L2)
- Ensemble methods
- Early stopping

## Overfitting and Underfitting

### What is Overfitting?

**Overfitting** occurs when a model learns the training data too well, including noise and outliers. The model performs excellently on training data but poorly on unseen data.

Signs of overfitting:
- Large gap between training and validation accuracy
- Model captures noise as if it were signal
- Too many parameters relative to observations

### Preventing Overfitting

1. **More training data**: Helps the model generalize better
2. **Feature selection**: Remove irrelevant features
3. **Regularization**: Add penalty for complexity (L1/Lasso, L2/Ridge)
4. **Cross-validation**: Use k-fold CV to validate performance
5. **Early stopping**: Stop training when validation error increases
6. **Dropout** (for neural networks): Randomly disable neurons during training
7. **Ensemble methods**: Combine multiple models (bagging, boosting)

### What is Underfitting?

**Underfitting** occurs when the model is too simple to capture the underlying pattern in the data.

Solutions:
- Use more complex model
- Add more features
- Reduce regularization
- Train longer (for neural networks)

## Gradient Descent Optimization

**Gradient descent** is the workhorse optimization algorithm for training machine learning models. It iteratively adjusts parameters to minimize a loss function.

### How Gradient Descent Works

1. Initialize parameters randomly
2. Calculate the gradient (partial derivatives) of the loss function
3. Update parameters in the opposite direction of the gradient
4. Repeat until convergence

The update rule:
```
θ = θ - α * ∇J(θ)
```

Where:
- θ = parameters
- α = learning rate
- ∇J(θ) = gradient of loss function

### Variants of Gradient Descent

**Batch Gradient Descent**: Uses entire dataset for each update
- Pros: Stable convergence
- Cons: Slow for large datasets, memory intensive

**Stochastic Gradient Descent (SGD)**: Uses one sample per update
- Pros: Fast, can escape local minima
- Cons: Noisy updates, may not converge smoothly

**Mini-batch Gradient Descent**: Uses small batches (typically 32-256 samples)
- Pros: Balance between batch and stochastic
- Cons: Requires tuning batch size

### Learning Rate

The learning rate α is crucial:
- Too high: May overshoot minimum, diverge
- Too low: Very slow convergence

Adaptive learning rate methods:
- **AdaGrad**: Adapts learning rate per parameter
- **RMSprop**: Uses moving average of squared gradients
- **Adam**: Combines momentum and RMSprop (most popular)

### Momentum

Momentum accelerates gradient descent by accumulating velocity:
```
v = βv - α∇J(θ)
θ = θ + v
```

Benefits:
- Faster convergence
- Dampens oscillations
- Helps escape shallow local minima

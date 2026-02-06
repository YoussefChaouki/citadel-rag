# Machine Learning Algorithms

## Decision Trees

Decision trees are versatile supervised learning algorithms used for both classification and regression tasks. They learn simple decision rules inferred from data features.

### How Decision Trees Work

1. Start at the root node with all data
2. Select the best feature to split data (using criteria like Gini impurity or information gain)
3. Create child nodes for each split
4. Recursively repeat until stopping criteria met (max depth, min samples, etc.)

### Advantages and Disadvantages

Advantages:
- Easy to interpret and visualize
- Handle both numerical and categorical data
- No feature scaling required
- Handle non-linear relationships

Disadvantages:
- Prone to overfitting (high variance)
- Sensitive to small data changes
- Can create biased trees with imbalanced data

## Random Forest

**Random Forest** is an ensemble learning method that constructs multiple decision trees and combines their predictions.

### How Random Forest Works

1. Create N bootstrap samples from training data (sampling with replacement)
2. For each sample, grow a decision tree:
   - At each node, randomly select m features from total M features
   - Split using the best feature among the m selected
3. Aggregate predictions:
   - Classification: majority vote
   - Regression: average

### Key Hyperparameters

- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of each tree
- **max_features**: Number of features to consider at each split
- **min_samples_split**: Minimum samples required to split a node
- **bootstrap**: Whether to use bootstrap sampling

### Why Random Forest Works

The power comes from two sources of randomness:
1. **Bootstrap aggregating (bagging)**: Each tree trained on different data subset
2. **Feature randomness**: Each split considers only subset of features

This reduces variance without increasing bias significantly, leading to better generalization.

### Out-of-Bag (OOB) Error

Each tree only sees ~63% of data due to bootstrap sampling. The remaining 37% (OOB samples) can be used for validation, providing a free cross-validation estimate.

## Support Vector Machines (SVM)

Support Vector Machines are powerful supervised learning models for classification and regression.

### Linear SVM

SVM finds the optimal hyperplane that maximizes the margin between classes. Support vectors are the data points closest to the decision boundary.

The optimization problem:
```
minimize: (1/2)||w||²
subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i
```

### Soft Margin SVM

For non-linearly separable data, soft margin SVM allows some misclassifications:
```
minimize: (1/2)||w||² + C Σξᵢ
```

Where:
- C: regularization parameter controlling trade-off
- ξᵢ: slack variables for misclassified points

### The Kernel Trick

The **kernel trick** allows SVM to find non-linear decision boundaries by implicitly mapping data to higher-dimensional space without explicitly computing the transformation.

A kernel function K(x, y) computes the dot product in the transformed space:
```
K(x, y) = φ(x) · φ(y)
```

### Common Kernels

1. **Linear Kernel**: K(x, y) = x · y
   - Use when data is linearly separable
   - Fast computation

2. **Polynomial Kernel**: K(x, y) = (γx · y + r)^d
   - Captures feature interactions
   - Parameters: degree d, coefficient r

3. **RBF (Gaussian) Kernel**: K(x, y) = exp(-γ||x - y||²)
   - Most popular for non-linear problems
   - Infinite-dimensional feature space
   - Parameter γ controls influence radius

4. **Sigmoid Kernel**: K(x, y) = tanh(γx · y + r)
   - Similar to neural network activation
   - Less commonly used

### Why Kernel Trick is Powerful

- Avoids explicit high-dimensional computation (can be infinite!)
- Computationally efficient (only need dot products)
- Enables learning complex decision boundaries
- Works well with small to medium datasets

## Gradient Boosting

Gradient Boosting builds an ensemble of weak learners (usually decision trees) sequentially, with each new model correcting errors of the combined ensemble.

### Algorithm Overview

1. Initialize model with constant value
2. For m = 1 to M:
   - Compute negative gradient (pseudo-residuals)
   - Fit a weak learner to pseudo-residuals
   - Find optimal step size
   - Update model
3. Final prediction is sum of all weak learners

### Popular Implementations

- **XGBoost**: Extreme Gradient Boosting with regularization
- **LightGBM**: Light Gradient Boosting Machine (faster, uses histogram-based algorithm)
- **CatBoost**: Handles categorical features natively

## K-Nearest Neighbors (KNN)

KNN is a simple, non-parametric algorithm that makes predictions based on the k closest training examples.

### How KNN Works

1. Store all training data
2. For a new point:
   - Calculate distance to all training points
   - Select k nearest neighbors
   - Classification: majority vote
   - Regression: average of neighbors

### Distance Metrics

- **Euclidean**: √(Σ(xᵢ - yᵢ)²)
- **Manhattan**: Σ|xᵢ - yᵢ|
- **Minkowski**: (Σ|xᵢ - yᵢ|^p)^(1/p)
- **Cosine similarity**: For text/sparse data

### Choosing K

- Small k: sensitive to noise, complex boundary
- Large k: smoother boundary, may miss local patterns
- Common practice: k = √n or cross-validation

### Considerations

- Feature scaling is crucial
- Curse of dimensionality in high dimensions
- Computational cost at prediction time
- Memory intensive (stores all data)

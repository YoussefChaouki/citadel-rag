# Machine Learning Best Practices

## Data Preprocessing

### Data Cleaning

Before training any model:

1. **Handle Missing Values**
   - Remove rows/columns with too many missing values
   - Imputation: mean, median, mode, or model-based
   - Create binary indicator for missingness

2. **Handle Outliers**
   - Identify using IQR or z-scores
   - Options: remove, cap, or transform

3. **Data Type Conversion**
   - Ensure correct types (numeric, categorical, datetime)
   - Parse dates, convert strings to categories

### Feature Scaling

Most ML algorithms benefit from scaled features:

**Standardization (Z-score normalization)**:
```
x_scaled = (x - mean) / std
```
- Results in mean=0, std=1
- Use for: SVM, neural networks, PCA

**Min-Max Normalization**:
```
x_scaled = (x - min) / (max - min)
```
- Scales to [0, 1] range
- Use for: neural networks, image data

**Robust Scaling**:
```
x_scaled = (x - median) / IQR
```
- Resistant to outliers

## Feature Engineering

**Feature engineering** is the process of creating new features from existing data to improve model performance.

### Techniques

1. **Mathematical Transformations**
   - Log transform for skewed distributions
   - Polynomial features for non-linear relationships
   - Interaction features: x₁ × x₂

2. **Encoding Categorical Variables**
   - One-hot encoding: Creates binary columns
   - Label encoding: For ordinal categories
   - Target encoding: Replace with target mean (careful of leakage)

3. **Binning/Discretization**
   - Convert continuous to categorical
   - Equal-width or equal-frequency bins

4. **Date/Time Features**
   - Extract: year, month, day, day of week, hour
   - Time since event
   - Cyclical encoding for periodic features

5. **Text Features**
   - Bag of words, TF-IDF
   - Word embeddings
   - Length, word count, special character count

### Feature Selection

Remove irrelevant or redundant features:

- **Filter Methods**: Statistical tests (correlation, chi-square)
- **Wrapper Methods**: Recursive feature elimination
- **Embedded Methods**: L1 regularization, tree feature importance

## Handling Imbalanced Datasets

Class **imbalance** occurs when one class significantly outnumbers others. This causes models to bias toward majority class.

### Detection

- Class distribution ratio (e.g., 95% vs 5%)
- Check if accuracy is misleading

### Solutions

1. **Resampling**
   - **Oversampling minority**: SMOTE (Synthetic Minority Oversampling)
   - **Undersampling majority**: Random or informed undersampling
   - **Combination**: SMOTE + Tomek links

2. **Algorithmic Approaches**
   - **Class weights**: Penalize misclassifying minority class more
   - **Cost-sensitive learning**: Different misclassification costs

3. **Ensemble Methods**
   - Balanced Random Forest
   - EasyEnsemble: Multiple undersampled subsets

4. **Evaluation Metrics**
   - Don't use accuracy!
   - Use precision, recall, F1, AUC-ROC, PR curve

### SMOTE (Synthetic Minority Oversampling Technique)

1. Select a minority class sample
2. Find its k nearest minority neighbors
3. Create synthetic samples along lines connecting to neighbors

## Model Evaluation

### Classification Metrics

**Confusion Matrix**:
```
                Predicted
              Pos    Neg
Actual Pos    TP     FN
       Neg    FP     TN
```

**Precision**: Of predicted positives, how many are correct?
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)**: Of actual positives, how many did we find?
```
Recall = TP / (TP + FN)
```

**F1 Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Specificity**: Of actual negatives, how many did we identify?
```
Specificity = TN / (TN + FP)
```

### When to Use Which Metric?

- **Precision**: When false positives are costly (spam detection)
- **Recall**: When false negatives are costly (cancer detection)
- **F1**: When you need balance between precision and recall
- **AUC-ROC**: For overall model ranking ability

### ROC and AUC

**ROC Curve**: Plots TPR vs FPR at various thresholds
- Diagonal line = random classifier
- Perfect classifier = top-left corner

**AUC**: Area Under ROC Curve
- 0.5 = random
- 1.0 = perfect
- Useful for comparing models

### Regression Metrics

- **MSE**: Mean Squared Error - penalizes large errors
- **RMSE**: Root MSE - same units as target
- **MAE**: Mean Absolute Error - robust to outliers
- **R²**: Proportion of variance explained

## Cross-Validation

**Cross-validation** is a technique to assess model generalization by testing on held-out data.

### Why Cross-Validation?

- More reliable than single train/test split
- Uses all data for both training and validation
- Provides variance estimate of performance
- Helps detect overfitting

### K-Fold Cross-Validation

1. Split data into K equal folds
2. For each fold:
   - Use fold as validation set
   - Train on remaining K-1 folds
   - Record performance
3. Average performance across all folds

Common choices: K = 5 or K = 10

### Variants

**Stratified K-Fold**: Maintains class distribution in each fold
- Essential for imbalanced classification

**Leave-One-Out (LOO)**: K = n (number of samples)
- Most thorough but computationally expensive

**Time Series Split**: Respects temporal order
- Never train on future data

**Group K-Fold**: Ensures groups don't span train/test
- Use when samples are not independent

### Cross-Validation Best Practices

1. Always use stratified CV for classification
2. Apply preprocessing within each fold to prevent leakage
3. Use time series split for temporal data
4. Report mean and standard deviation of metrics

## Hyperparameter Tuning

### Grid Search

Exhaustively try all combinations:
```python
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
```

- Pros: Thorough, finds best in grid
- Cons: Exponential complexity, may miss optimal between points

### Random Search

Randomly sample hyperparameter combinations:
- Pros: More efficient, explores more values
- Cons: May miss optimal

### Bayesian Optimization

Uses probabilistic model to guide search:
- Builds surrogate model of objective function
- Balances exploration and exploitation
- More efficient than grid/random for expensive evaluations

### Best Practices

1. Start with random search to identify promising regions
2. Use cross-validation for evaluation
3. Consider computational budget
4. Use early stopping when possible
5. Document all experiments

## Production Considerations

### Model Serialization

Save trained models for deployment:
- Pickle/Joblib for scikit-learn
- SavedModel for TensorFlow
- torch.save for PyTorch

### Monitoring

Track in production:
- Prediction latency
- Input data distribution (detect drift)
- Model performance over time
- Error rates and types

### Model Updates

- Retrain periodically or on trigger
- A/B test new models
- Maintain model versioning
- Enable rollback capability

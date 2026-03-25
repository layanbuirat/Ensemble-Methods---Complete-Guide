# Ensemble Methods - Complete Guide

## Table of Contents
1. [Introduction to Ensemble Methods](#introduction)
2. [Bias vs. Variance Tradeoff](#bias-variance)
3. [Bagging (Bootstrap Aggregating)](#bagging)
4. [Random Forests](#random-forests)
5. [Boosting with AdaBoost](#adaboost)
6. [AdaBoost Mathematics](#adaboost-math)
7. [Implementation with Scikit-learn](#implementation)
8. [Practice Exercises](#exercises)

---

## Introduction to Ensemble Methods {#introduction}

Ensemble methods combine multiple models (called **weak learners**) to create a more powerful and accurate model (called a **strong learner**). Think of it like getting opinions from multiple experts instead of relying on just one.

```python
# Basic concept: Combining multiple models
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create an ensemble of different models
ensemble = VotingClassifier([
    ('tree', DecisionTreeClassifier()),
    ('logistic', LogisticRegression()),
    ('svm', SVC())
])
```

---

## Bias vs. Variance Tradeoff {#bias-variance}

### Understanding Bias and Variance

| Term | Definition | Example | Characteristics |
|------|------------|---------|-----------------|
| **High Bias** | Model doesn't bend well to the data | Linear Regression | Underfitting, too simple |
| **Low Bias** | Model bends well to the data | Deep Decision Tree | Flexible, complex |
| **High Variance** | Changes drastically for each data point | Deep Decision Tree | Overfitting, too sensitive |
| **Low Variance** | Stable across different datasets | Linear Regression | Consistent, rigid |

```python
# Visualizing bias vs variance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# High Bias model (underfitting)
high_bias_model = LinearRegression()
high_bias_model.fit(X, y)

# High Variance model (overfitting)
high_variance_model = DecisionTreeRegressor(max_depth=None)
high_variance_model.fit(X, y)

# The ensemble (Random Forest) finds the sweet spot
from sklearn.ensemble import RandomForestRegressor
ensemble_model = RandomForestRegressor(n_estimators=100)
ensemble_model.fit(X, y)
```

### Key Insight
> By combining algorithms, we can build models that perform better by meeting in the middle in terms of bias and variance.

---

## Bagging (Bootstrap Aggregating) {#bagging}

### What is Bagging?

**Bagging** = **B**ootstrap **Agg**regating

The process:
1. Create random subsets of the original data (with replacement)
2. Train a weak learner on each subset
3. Combine predictions through **voting** (classification) or **averaging** (regression)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create bagging ensemble
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,           # Number of weak learners
    max_samples=0.8,           # Bootstrap 80% of data
    bootstrap=True,            # Sample with replacement
    random_state=42
)

bagging_model.fit(X_train, y_train)
predictions = bagging_model.predict(X_test)
```

### Why Bagging Works
- Reduces **variance** without increasing bias
- Each model sees different data, so they make different errors
- Combining them cancels out individual errors

---

## Random Forests {#random-forests}

### Key Randomization Techniques

Random Forests add **two layers of randomness**:

1. **Bootstrap the data** - Sample data with replacement
2. **Subset the features** - Use random subset of features for each split

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest with controlled randomness
rf_model = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_features='sqrt',        # √n features per split
    max_depth=None,             # Full depth
    bootstrap=True,             # Bootstrap samples
    random_state=42
)

rf_model.fit(X_train, y_train)
```

### Why Random Forests Beat Single Decision Trees

| Problem | Single Decision Tree | Random Forest |
|---------|---------------------|---------------|
| Overfitting | High | Low (controlled) |
| Generalization | Poor | Excellent |
| Feature dependency | High | Reduced |

---

## Boosting with AdaBoost {#adaboost}

### The AdaBoost Algorithm Steps

1. **Step 1**: Fit first learner to maximize accuracy (minimize errors)
2. **Step 2**: Identify misclassified points and increase their weights
3. **Step 3**: Fit next learner focusing more on misclassified points
4. **Repeat**: Continue for specified number of estimators
5. **Combine**: Weighted voting based on each learner's accuracy

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# AdaBoost with decision tree weak learners
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Stump
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME.R'
)

adaboost.fit(X_train, y_train)
```

### Weight Update Formula

The weight for each weak learner is:

$$\text{weight} = \ln\left(\frac{\text{accuracy}}{1 - \text{accuracy}}\right)$$

```python
def calculate_learner_weight(accuracy):
    """
    Calculate weight for a weak learner based on its accuracy
    
    Returns:
        Positive weight for accurate models
        Zero for random models (accuracy = 0.5)
        Negative weight for models worse than random
    """
    if accuracy <= 0 or accuracy >= 1:
        return float('inf') if accuracy == 1 else float('-inf')
    
    return np.log(accuracy / (1 - accuracy))

# Example
print(calculate_learner_weight(0.95))  # ~2.94 (very positive)
print(calculate_learner_weight(0.50))  # 0.0 (useless)
print(calculate_learner_weight(0.05))  # ~-2.94 (very negative)
```

---

## AdaBoost Mathematics {#adaboost-math}

### Step-by-Step Example

**Initial weights**: Each point gets weight = 1

**First Learner**:
- Correctly classified: sum(weights) = 7
- Incorrectly classified: sum(weights) = 3
- Weight = ln(7/3) = 0.84

**Second Learner** (after reweighting):
- Correctly classified: sum(weights) = 11
- Incorrectly classified: sum(weights) = 3
- Weight = ln(11/3) = 1.30

**Combining Models**:
```python
def combine_predictions(predictions, weights):
    """
    Combine weak learner predictions with their weights
    
    For each region:
        Positive regions: add weight
        Negative regions: subtract weight
        Final sign determines prediction
    """
    final_score = sum(pred * weight for pred, weight in zip(predictions, weights))
    return 1 if final_score > 0 else -1  # or 0 for tie
```

---

## Implementation with Scikit-learn {#implementation}

### Complete Example: Spam Classification

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data (example with spam dataset)
# df = pd.read_csv('spam.csv')

# Assuming X, y are prepared
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# 2. AdaBoost
ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
ada.fit(X_train_scaled, y_train)
ada_pred = ada.predict(X_test_scaled)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, ada_pred):.4f}")

# 3. Bagging
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bag.fit(X_train_scaled, y_train)
bag_pred = bag.predict(X_test_scaled)
print(f"Bagging Accuracy: {accuracy_score(y_test, bag_pred):.4f}")

# 4. Compare all models
from sklearn.metrics import confusion_matrix, roc_auc_score

models = {
    'Random Forest': rf,
    'AdaBoost': ada,
    'Bagging': bag
}

for name, model in models.items():
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, proba):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, pred)}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Tuning Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='accuracy'
)
grid_rf.fit(X_train_scaled, y_train)
print(f"Best RF parameters: {grid_rf.best_params_}")
print(f"Best RF score: {grid_rf.best_score_:.4f}")

# Tuning AdaBoost
param_grid_ada = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.5, 1.0],
    'base_estimator__max_depth': [1, 2, 3]
}

grid_ada = GridSearchCV(
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=42),
    param_grid_ada,
    cv=5,
    scoring='accuracy'
)
grid_ada.fit(X_train_scaled, y_train)
print(f"Best AdaBoost parameters: {grid_ada.best_params_}")
print(f"Best AdaBoost score: {grid_ada.best_score_:.4f}")
```

---

## Practice Exercises {#exercises}

### Exercise 1: Bias-Variance Analysis

```python
# Question: Which model has high bias? Which has high variance?
# Your answer: 
# High Bias: Linear Regression
# High Variance: Deep Decision Tree

# Code to verify:
from sklearn.metrics import mean_squared_error

def evaluate_bias_variance(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)
    
    bias = train_error
    variance = test_error - train_error
    
    return bias, variance
```

### Exercise 2: Build Your Own Ensemble

```python
class SimpleEnsemble:
    """
    Create a simple voting ensemble
    """
    def __init__(self, models, voting='hard'):
        self.models = models
        self.voting = voting
        
    def fit(self, X, y):
        for name, model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for _, model in self.models])
        if self.voting == 'hard':
            # Majority voting
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, 
                arr=predictions
            )
        elif self.voting == 'soft':
            # Weighted probability voting
            probas = np.array([model.predict_proba(X) for _, model in self.models])
            return np.argmax(np.mean(probas, axis=0), axis=1)
```

### Exercise 3: AdaBoost from Scratch

```python
class AdaBoostFromScratch:
    """
    Implement AdaBoost algorithm from scratch for understanding
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
    
    def fit(self, X, y):
        # Convert y to -1, 1
        y = np.where(y == 0, -1, 1)
        n_samples = X.shape[0]
        # Initialize weights
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Fit weak learner with current weights
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=weights)
            predictions = model.predict(X)
            
            # Calculate weighted error
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            
            # Calculate alpha (learner weight)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            # Update weights
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)  # Normalize
            
            # Store model and alpha
            self.models.append(model)
            self.alphas.append(alpha)
    
    def predict(self, X):
        # Combine predictions with weights
        final_pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            pred = np.where(model.predict(X) == 0, -1, 1)
            final_pred += alpha * pred
        
        return np.where(final_pred >= 0, 1, 0)
```

---

## Key Concepts Summary

### Important Terms

| Term | Definition |
|------|------------|
| **Ensemble** | Combining multiple models to create a stronger predictor |
| **Bagging** | Bootstrap sampling + aggregation to reduce variance |
| **Boosting** | Sequential training focusing on previous mistakes |
| **Random Forest** | Bagging + feature subsetting with decision trees |
| **AdaBoost** | Adaptive boosting with weighted instances and learners |
| **Weak Learner** | Slightly better than random guessing (e.g., decision stump) |
| **Strong Learner** | Highly accurate model formed by combining weak learners |

### When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Bagging** | High variance models | Reduces overfitting, parallelizable | Can't correct systematic bias |
| **Random Forest** | Complex data with many features | Handles non-linearity, feature importance | Less interpretable |
| **AdaBoost** | When you need sequential improvement | Adaptive, focuses on hard examples | Sensitive to noisy data |

---

## Quick Reference Code Snippets

```python
# Import everything you need
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    AdaBoostClassifier, 
    AdaBoostRegressor,
    BaggingClassifier, 
    BaggingRegressor,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Quick ensemble pipeline
def build_ensemble_pipeline(X_train, y_train, X_test):
    # Create models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    bag = BaggingClassifier(n_estimators=50, random_state=42)
    
    # Create voting ensemble
    ensemble = VotingClassifier([
        ('rf', rf),
        ('ada', ada),
        ('bag', bag)
    ], voting='soft')
    
    # Fit and predict
    ensemble.fit(X_train, y_train)
    return ensemble.predict(X_test)
```

---

## Common Interview Questions

### Question 1: What's the difference between bagging and boosting?

**Answer**:
- **Bagging**: Trains models in parallel on bootstrap samples; reduces variance
- **Boosting**: Trains models sequentially, each focusing on previous mistakes; reduces bias

### Question 2: Why does AdaBoost weight misclassified points?

**Answer**: To make subsequent classifiers focus more on difficult examples, forcing them to learn from previous mistakes and improve overall performance.

### Question 3: Which model is worse: always correct, always wrong, or 50/50?

**Answer**: The 50/50 model is worst because it provides no useful information. The always wrong model can be inverted to become perfect, while the always correct model is already perfect.

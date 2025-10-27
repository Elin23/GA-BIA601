"""
Embedded Feature Selection using Random Forest Feature Importance

This script demonstrates how to use Random Forest feature importance for embedded feature selection.
Embedded methods perform feature selection as part of the model training process, unlike filter
methods (which are independent) or wrapper methods (which use a separate evaluation function).

The approach:
1. Train a Random Forest classifier on the full dataset
2. Extract feature importance scores from the trained model
3. Select the top-k most important features
4. Train a downstream model using only the selected features
5. Compare performance against a baseline model using all features

Author: Generated for BIA601 course
Date: 2024
"""

# Import required libraries
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation (though not used in this synthetic example)
from sklearn.model_selection import train_test_split  # For splitting data into train/validation sets
from sklearn.linear_model import LogisticRegression  # For downstream model evaluation
from sklearn.ensemble import RandomForestClassifier  # For feature importance calculation
from sklearn.metrics import accuracy_score  # For evaluating model performance

# ---------------------------
# 1) DATA PREPARATION
# ---------------------------
# This section loads and prepares the dataset for feature selection.
# In practice, you would load your actual dataset here.

# For demonstration purposes, we generate a synthetic binary classification dataset
# with known characteristics to test our feature selection method
from sklearn.datasets import make_classification

print("=" * 60)
print("GENERATING SYNTHETIC DATASET")
print("=" * 60)

# Generate synthetic dataset with specific characteristics:
# - 1000 samples: reasonable size for demonstration
# - 100 features: high-dimensional to make feature selection meaningful
# - 20 informative features: only these actually contribute to classification
# - 10 redundant features: linear combinations of informative features
# - 2 classes: binary classification problem
# - random_state=42: ensures reproducible results
X, y = make_classification(n_samples=1000, n_features=100, n_informative=20,
                           n_redundant=10, n_classes=2, random_state=42)

# Display dataset characteristics
print(f"✓ Dataset generated successfully!")
print(f"  - Samples: {X.shape[0]}")  # Number of data points
print(f"  - Features: {X.shape[1]}")  # Total number of features
print(f"  - Informative features: 20")  # Features that actually matter for classification
print(f"  - Redundant features: 10")  # Features that are linear combinations of informative ones
print(f"  - Classes: {len(np.unique(y))}")  # Number of target classes
print(f"  - Class distribution: {np.bincount(y)}")  # How many samples per class
print()

# NOTE: To use your own data, replace the above with:
# df = pd.read_csv('your_data.csv')
# X = df.drop(columns=['target_column'])  # Features
# y = df['target_column']  # Target variable

# ---------------------------
# 2) DATA SPLITTING
# ---------------------------
# Split the data into training and validation sets to properly evaluate
# the feature selection method without data leakage

print("=" * 60)
print("SPLITTING DATA INTO TRAIN/VALIDATION SETS")
print("=" * 60)

# Split data into 80% training and 20% validation
# - test_size=0.2: 20% of data for validation
# - random_state=42: ensures reproducible splits
# - stratify=y: maintains the same class distribution in both sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display split information
print(f"✓ Data split completed!")
print(f"  - Training set: {X_train.shape[0]} samples")  # 800 samples for training
print(f"  - Validation set: {X_val.shape[0]} samples")  # 200 samples for validation
print(f"  - Training class distribution: {np.bincount(y_train)}")  # Class balance in training
print(f"  - Validation class distribution: {np.bincount(y_val)}")  # Class balance in validation
print()


# ---------------------------
# 3) EMBEDDED FEATURE SELECTION: RANDOM FOREST
# ---------------------------
# This section implements embedded feature selection using Random Forest.
# Random Forest naturally provides feature importance scores based on how much
# each feature contributes to reducing impurity across all trees in the forest.

print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

# Step 1: Train Random Forest on the full dataset
# Random Forest is robust to different scales, so we use unscaled data
print("Training Random Forest for feature importance...")
rf = RandomForestClassifier(
    n_estimators=200,    # Number of trees in the forest (more trees = more stable importance)
    random_state=42,     # For reproducible results
    n_jobs=-1           # Use all available CPU cores for faster training
)
rf.fit(X_train, y_train)  # Train on training data only
print("✓ Random Forest trained successfully!")

# Step 2: Extract feature importance scores
# Random Forest calculates importance as the mean decrease in impurity
# when a feature is used for splitting across all trees
importances = rf.feature_importances_
print(f"✓ Feature importances calculated!")
print(f"  - Mean importance: {np.mean(importances):.4f}")  # Average importance across all features
print(f"  - Max importance: {np.max(importances):.4f}")    # Highest importance score
print(f"  - Min importance: {np.min(importances):.4f}")    # Lowest importance score
print()

# Step 3: Select top-k most important features
# We choose the top 20 features (20% of total features)
# This is a reasonable choice for this dataset since we know 20 are informative
top_k = 20

# Get indices of top-k features (sorted by importance in descending order)
# np.argsort(importances)[-top_k:] gets the last top_k indices from sorted array
# [::-1] reverses to get them in descending order of importance
idx_topk = np.argsort(importances)[-top_k:][::-1]

print(f"✓ Top-{top_k} features selected!")
print(f"  - Selected features: {len(idx_topk)} / {X.shape[1]} ({len(idx_topk)/X.shape[1]*100:.1f}%)")
print(f"  - Selected feature indices: {idx_topk}")
print(f"  - Feature importances: {importances[idx_topk]}")
print()

# Step 4: Evaluate performance using only selected features
# This is the key step: we train a downstream model using only the selected features
# to see if we can maintain good performance with fewer features
print("Evaluating RF-selected features with downstream model...")

# Use Logistic Regression as the downstream model (any classifier would work)
# We use raw features without scaling since Random Forest doesn't require it
model_rf_subset = LogisticRegression(max_iter=1000, random_state=42)

# Train on the subset of features selected by Random Forest
model_rf_subset.fit(X_train[:, idx_topk], y_train)

# Make predictions on validation set using only selected features
preds_rf = model_rf_subset.predict(X_val[:, idx_topk])

# Calculate accuracy
acc_rf = accuracy_score(y_val, preds_rf)
print(f"✓ RF evaluation completed!")
print(f"  - Accuracy with RF-top-{top_k} features: {acc_rf:.4f}")
print()

# ---------------------------
# 4) EVALUATION & COMPARISON
# ---------------------------
# This section evaluates the effectiveness of our feature selection by comparing
# the performance of the model with selected features against a baseline model
# that uses all features.

print("=" * 60)
print("FINAL COMPARISON & SUMMARY")
print("=" * 60)

# Step 1: Calculate baseline performance using all features
# This gives us the best possible performance we can achieve with the full dataset
print("Calculating baseline accuracy with all features...")
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)  # Train on all features
baseline_preds = baseline_model.predict(X_val)  # Predict on all features
baseline_acc = accuracy_score(y_val, baseline_preds)
print(f"✓ Baseline evaluation completed!")
print(f"  - Accuracy with all features: {baseline_acc:.4f}")
print()

# Step 2: Display detailed results comparison
print("FEATURE SELECTION RESULTS:")
print("-" * 40)
print(f" Random Forest (Selected Features):")
print(f"   • Accuracy: {acc_rf:.4f}")
print(f"   • Features selected: {top_k} / {X.shape[1]} ({top_k/X.shape[1]*100:.1f}%)")
print(f"   • Feature reduction: {X.shape[1] - top_k} features removed")
print()
print(f" Baseline (All Features):")
print(f"   • Accuracy: {baseline_acc:.4f}")
print(f"   • Features used: {X.shape[1]} / {X.shape[1]} (100.0%)")
print()

# Step 3: Performance analysis
# We consider the feature selection successful if it maintains at least 95% of baseline performance
print(" PERFORMANCE COMPARISON:")
print("-" * 40)
if acc_rf >= baseline_acc * 0.95:
    print("Random Forest method: Maintains >95% of baseline performance")
    print("   → Feature selection is successful! We can reduce features without significant loss.")
else:
    print(" Random Forest method: Performance below 95% of baseline")
    print("   → Consider selecting more features or using a different selection method.")

# Step 4: Detailed performance comparison
print(f"\n Performance vs Baseline: ", end="")
if acc_rf > baseline_acc:
    print(f"Random Forest outperforms baseline by {acc_rf - baseline_acc:.4f}")
    print("   → This is unusual but possible due to noise reduction from feature selection.")
elif acc_rf < baseline_acc:
    print(f"Random Forest underperforms baseline by {baseline_acc - acc_rf:.4f}")
    print("   → Some important features may have been excluded.")
else:
    print("Random Forest matches baseline performance")
    print("   → Perfect feature selection - no performance loss!")

# Optional: If you have a GA baseline result, print it here as well
# print(f"- GA baseline: <your GA result>")

# ---------------------------
# 5) EXTENSIONS & MODIFICATIONS
# ---------------------------
# This section provides suggestions for extending and modifying the code
# for different scenarios and requirements.

print("\n" + "=" * 60)

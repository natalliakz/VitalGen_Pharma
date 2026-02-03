"""
Train clinical trial enrollment prediction model.

This script trains a machine learning model to predict the likelihood
of successful patient enrollment at clinical trial sites.

Model predicts: Probability that a site will meet its enrollment target.

Uses synthetic data for demonstration purposes only.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path

from model_utils import load_and_prepare_data, get_feature_importance_names


def train_model():
    """Train and evaluate enrollment prediction model."""

    print("=" * 70)
    print("Clinical Trial Enrollment Prediction Model Training")
    print("=" * 70)

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    X, y, feature_names, encoders, df = load_and_prepare_data()
    print(f"   Loaded {len(X)} site-trial combinations")
    print(f"   Features: {len(feature_names)}")
    print(f"   Positive class (success) rate: {y.mean():.2%}")

    # Split data
    print("\n2. Splitting data (70% train, 15% validation, 15% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
    )

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model - using Random Forest (simple, interpretable)
    print("\n4. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)
    print("   ✓ Model training complete")

    # Cross-validation
    print("\n5. Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"   Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Evaluate on validation set
    print("\n6. Evaluating on validation set...")
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)

    print(f"   Accuracy:  {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall:    {val_recall:.4f}")
    print(f"   F1 Score:  {val_f1:.4f}")
    print(f"   ROC-AUC:   {val_roc_auc:.4f}")

    # Final evaluation on test set
    print("\n7. Final evaluation on test set...")
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1 Score:  {test_f1:.4f}")
    print(f"   ROC-AUC:   {test_roc_auc:.4f}")

    print("\n   Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Not Met", "Met Target"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("   Confusion Matrix:")
    print(f"   True Negatives:  {cm[0, 0]:>4}  |  False Positives: {cm[0, 1]:>4}")
    print(f"   False Negatives: {cm[1, 0]:>4}  |  True Positives:  {cm[1, 1]:>4}")

    # Feature importance
    print("\n8. Analyzing feature importance...")
    feature_importance = model.feature_importances_
    readable_names = get_feature_importance_names()

    importance_dict = {name: float(imp) for name, imp in zip(readable_names, feature_importance)}
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n   Top 10 Most Important Features:")
    for i, (name, importance) in enumerate(sorted_importance[:10], 1):
        print(f"   {i:>2}. {name:<40} {importance:>6.4f}")

    # Save model artifacts
    print("\n9. Saving model artifacts...")
    Path("ml").mkdir(exist_ok=True)

    joblib.dump(model, "ml/enrollment_model.joblib")
    print("   ✓ ml/enrollment_model.joblib")

    joblib.dump(scaler, "ml/scaler.joblib")
    print("   ✓ ml/scaler.joblib")

    joblib.dump(encoders, "ml/encoders.joblib")
    print("   ✓ ml/encoders.joblib")

    # Save model metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "features": feature_names,
        "feature_names_readable": readable_names,
        "training_samples": len(X_train),
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "feature_importance": importance_dict,
        "target_variable": "enrollment_success",
        "created_date": str(np.datetime64("today"))
    }

    with open("ml/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   ✓ ml/model_metadata.json")

    print("\n" + "=" * 70)
    print("✓ Model training complete!")
    print("=" * 70)
    print("\nModel can be used to predict enrollment success probability")
    print("for new clinical trial sites based on site and trial characteristics.")
    print("\nNOTE: Model trained on synthetic data for demonstration purposes only.")


if __name__ == "__main__":
    train_model()

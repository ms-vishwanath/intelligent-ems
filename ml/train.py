import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from pathlib import Path


def load_and_prepare_data(csv_path: str = "ml/sample_data.csv") -> tuple:
    """
    Load sample data and prepare features and labels.
    
    Returns:
        Tuple of (X, y) where X is feature array and y is target labels
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Prepare features similar to features.py
    features_list = []
    
    for _, row in df.iterrows():
        age = float(row.get("patient_age", 0))
        gender = str(row.get("patient_gender", "Other"))
        symptoms = str(row.get("reported_symptoms", "")).lower()
        incident_type = str(row.get("incident_type", "medical")).lower()
        
        # Encode gender
        gender_m = 1.0 if gender == "M" else 0.0
        gender_f = 1.0 if gender == "F" else 0.0
        gender_other = 1.0 if gender not in ["M", "F"] else 0.0
        
        # Age features
        age_normalized = age / 100.0
        is_elderly = 1.0 if age >= 65 else 0.0
        is_child = 1.0 if age < 18 else 0.0
        
        # Symptom features
        critical_keywords = ["chest pain", "unconscious", "cardiac", "stroke", "seizure", 
                            "difficulty breathing", "choking", "severe", "critical", "emergency"]
        moderate_keywords = ["pain", "fever", "nausea", "dizziness", "weakness", "injury"]
        
        has_critical_symptom = 1.0 if any(keyword in symptoms for keyword in critical_keywords) else 0.0
        has_moderate_symptom = 1.0 if any(keyword in symptoms for keyword in moderate_keywords) else 0.0
        symptom_length = min(len(symptoms) / 200.0, 1.0)
        
        # Incident type
        incident_medical = 1.0 if incident_type == "medical" else 0.0
        incident_trauma = 1.0 if incident_type == "trauma" else 0.0
        incident_other = 1.0 if incident_type not in ["medical", "trauma"] else 0.0
        
        # Time features (if available in data)
        hour_of_day = float(row.get("hour_of_day", 12)) / 24.0 if "hour_of_day" in df.columns else 12.0 / 24.0
        is_weekend = float(row.get("is_weekend", 0)) if "is_weekend" in df.columns else 0.0
        
        features = [
            age_normalized,
            gender_m,
            gender_f,
            gender_other,
            is_elderly,
            is_child,
            has_critical_symptom,
            has_moderate_symptom,
            symptom_length,
            incident_medical,
            incident_trauma,
            incident_other,
            hour_of_day,
            is_weekend,
        ]
        
        features_list.append(features)
    
    X = np.array(features_list, dtype=np.float32)
    
    # Get target labels (severity)
    if "severity" in df.columns:
        y = df["severity"].values
    elif "severity_score" in df.columns:
        # Convert continuous score to categorical
        y_continuous = df["severity_score"].values
        y = np.where(y_continuous < 0.25, 0, 
                    np.where(y_continuous < 0.5, 1,
                            np.where(y_continuous < 0.75, 2, 3)))
    else:
        # Generate synthetic labels based on features
        y = np.zeros(len(X))
        for i, feat in enumerate(X):
            if feat[6] > 0.5:  # has_critical_symptom
                y[i] = 3  # critical
            elif feat[7] > 0.5:  # has_moderate_symptom
                y[i] = 2  # high
            elif feat[4] > 0.5 or feat[5] > 0.5:  # is_elderly or is_child
                y[i] = 1  # medium
            else:
                y[i] = 0  # low
    
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, model_path: str = "ml/model.pkl"):
    """
    Train RandomForestClassifier and save to file.
    
    Args:
        X: Feature array
        y: Target labels
        model_path: Path to save the model
    """
    # Check if stratified split is possible (each class needs at least 2 samples)
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = np.min(counts)
    can_stratify = min_class_count >= 2
    
    if can_stratify:
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Using stratified train-test split")
    else:
        # Split data without stratification (some classes have too few samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Warning: Using non-stratified split (minimum class count: {min_class_count})")
    
    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training RandomForestClassifier...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Get all unique classes from both y_test and y_pred to handle missing classes
    all_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = ["low", "medium", "high", "critical"]
    # Only include names for classes that exist
    labels_to_use = [i for i in range(4) if i in all_classes]
    names_to_use = [target_names[i] for i in labels_to_use]
    print(classification_report(y_test, y_pred, 
                              labels=labels_to_use,
                              target_names=names_to_use,
                              zero_division=0))
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to {model_path}")
    return model


def main():
    """Main training function"""
    csv_path = "ml/sample_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please create sample data first.")
        return
    
    print(f"Loading data from {csv_path}...")
    X, y = load_and_prepare_data(csv_path)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    model = train_model(X, y)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()


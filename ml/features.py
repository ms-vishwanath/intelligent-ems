import numpy as np
from typing import Dict, Any


def build_features(event_data: Dict[str, Any]) -> np.ndarray:
    """
    Convert raw event data to model-ready feature array.
    
    Args:
        event_data: Dictionary containing event information with keys:
            - patient_age: int
            - patient_gender: str (M, F, Other)
            - location_lat: float
            - location_lon: float
            - reported_symptoms: str
            - incident_type: str
    
    Returns:
        numpy array of features ready for model prediction
    """
    # Extract base features
    age = float(event_data.get("patient_age", 0))
    gender = event_data.get("patient_gender", "Other")
    symptoms = event_data.get("reported_symptoms", "").lower()
    incident_type = event_data.get("incident_type", "medical").lower()
    
    # Encode gender (one-hot encoding: M=1,0,0; F=0,1,0; Other=0,0,1)
    gender_m = 1.0 if gender == "M" else 0.0
    gender_f = 1.0 if gender == "F" else 0.0
    gender_other = 1.0 if gender not in ["M", "F"] else 0.0
    
    # Age-based features
    age_normalized = age / 100.0  # Normalize to 0-1 range
    is_elderly = 1.0 if age >= 65 else 0.0
    is_child = 1.0 if age < 18 else 0.0
    
    # Symptom-based features (keyword detection)
    critical_keywords = ["chest pain", "unconscious", "cardiac", "stroke", "seizure", 
                        "difficulty breathing", "choking", "severe", "critical", "emergency"]
    moderate_keywords = ["pain", "fever", "nausea", "dizziness", "weakness", "injury"]
    
    has_critical_symptom = 1.0 if any(keyword in symptoms for keyword in critical_keywords) else 0.0
    has_moderate_symptom = 1.0 if any(keyword in symptoms for keyword in moderate_keywords) else 0.0
    symptom_length = min(len(symptoms) / 200.0, 1.0)  # Normalize symptom description length
    
    # Incident type encoding
    incident_medical = 1.0 if incident_type == "medical" else 0.0
    incident_trauma = 1.0 if incident_type == "trauma" else 0.0
    incident_other = 1.0 if incident_type not in ["medical", "trauma"] else 0.0
    
    # Time-based features (if available)
    # For now, we'll use default values - can be enhanced with actual timestamp
    hour_of_day = 12.0 / 24.0  # Normalized to 0-1 (default to noon)
    is_weekend = 0.0  # Default to weekday
    
    # Combine all features into array
    # Feature order must match training data
    features = np.array([
        age_normalized,           # 0
        gender_m,                 # 1
        gender_f,                 # 2
        gender_other,             # 3
        is_elderly,               # 4
        is_child,                 # 5
        has_critical_symptom,     # 6
        has_moderate_symptom,     # 7
        symptom_length,           # 8
        incident_medical,         # 9
        incident_trauma,          # 10
        incident_other,           # 11
        hour_of_day,              # 12
        is_weekend,               # 13
    ], dtype=np.float32)
    
    return features.reshape(1, -1)  # Reshape to (1, n_features) for single prediction


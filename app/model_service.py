import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ml.features import build_features


class ModelService:
    """Service for loading and using the ML severity prediction model"""
    
    def __init__(self, model_path: str = "ml/model.pkl"):
        base_dir = Path(__file__).resolve().parent.parent  # project root (two levels up from app/)
        path_obj = Path(model_path)
        if not path_obj.is_absolute():
            path_obj = base_dir / path_obj
        self.model_path = path_obj
        self.model: Optional[Any] = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from file"""
        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}. "
                      f"Run 'python ml/train.py' to train the model.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict_severity(self, event_data: Dict[str, Any]) -> Tuple[float, str, bool]:
        """
        Predict severity score and category for an event.
        
        Args:
            event_data: Dictionary containing event information
        
        Returns:
            Tuple of (severity_score, severity_category)
            severity_score: float between 0 and 1
            severity_category: str (low, medium, high, critical)
        """
        if self.model is None:
            severity_score, category = self._fallback_prediction(event_data)
            return severity_score, category, True
        
        try:
            # Build features
            features = build_features(event_data)
            
            # Predict
            prediction = self.model.predict_proba(features)[0]
            
            # Get severity score (weighted average of class probabilities)
            # Classes: 0=low, 1=medium, 2=high, 3=critical
            if len(prediction) == 4:
                severity_score = (
                    prediction[0] * 0.125 +  # low: 0.0-0.25
                    prediction[1] * 0.375 +  # medium: 0.25-0.5
                    prediction[2] * 0.75 +    # high: 0.5-0.75
                    prediction[3] * 0.95      # critical: 0.75-1.0
                )
            else:
                # Fallback if model structure is different
                severity_score = float(np.max(prediction))
            
            # Determine category
            if severity_score < 0.25:
                category = "low"
            elif severity_score < 0.5:
                category = "medium"
            elif severity_score < 0.75:
                category = "high"
            else:
                category = "critical"
            
            return float(severity_score), category, False
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            severity_score, category = self._fallback_prediction(event_data)
            return severity_score, category, True
    
    def _fallback_prediction(self, event_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Fallback prediction using heuristics when model is unavailable.
        
        Args:
            event_data: Dictionary containing event information
        
        Returns:
            Tuple of (severity_score, severity_category)
        """
        age = event_data.get("patient_age", 50)
        symptoms = str(event_data.get("reported_symptoms", "")).lower()
        incident_type = str(event_data.get("incident_type", "medical")).lower()
        
        severity_score = 0.3  # Default to medium-low
        
        # Age factors
        if age >= 75 or age < 5:
            severity_score += 0.2
        
        # Symptom keywords
        critical_keywords = ["chest pain", "unconscious", "cardiac", "stroke", "seizure",
                           "difficulty breathing", "choking", "severe", "critical", "emergency"]
        moderate_keywords = ["pain", "fever", "nausea", "dizziness", "weakness", "injury"]
        
        if any(keyword in symptoms for keyword in critical_keywords):
            severity_score += 0.4
        elif any(keyword in symptoms for keyword in moderate_keywords):
            severity_score += 0.2
        
        # Incident type
        if incident_type == "trauma":
            severity_score += 0.1
        
        # Clamp to 0-1
        severity_score = min(max(severity_score, 0.0), 1.0)
        
        # Determine category
        if severity_score < 0.25:
            category = "low"
        elif severity_score < 0.5:
            category = "medium"
        elif severity_score < 0.75:
            category = "high"
        else:
            category = "critical"
        
        return severity_score, category
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


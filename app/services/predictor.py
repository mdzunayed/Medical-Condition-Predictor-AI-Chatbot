import joblib
import logging
from pathlib import Path
from typing import List

from app.config import MODEL_PATH, CLASS_NAMES, FEATURE_RANGES
from app.schemas import PredictionResponse

logger = logging.getLogger(__name__)


class Predictor:
    """Load and use ML model for predictions"""

    def __init__(self, model_path: str = None):
        """
        Initialize predictor with model path.

        Args:
            model_path: Path to model file (default: config.MODEL_PATH)
        """
        if model_path is None:
            model_path = MODEL_PATH

        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model from disk using joblib"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
            logger.info(f"   Model type: {type(self.model).__name__}")
            logger.info(f"   Expected features: {self.model.n_features_in_}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def predict(self, feature_vector: List[float]) -> PredictionResponse:
        """
        Make prediction on feature vector.

        Args:
            feature_vector: List of 16 floats in correct order

        Returns:
            PredictionResponse with prediction, probability, risk level

        Raises:
            RuntimeError: If model not loaded
            ValueError: If feature vector wrong size
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if len(feature_vector) != 16:
            raise ValueError(f"Expected 16 features, got {len(feature_vector)}")

        try:
            # Make prediction on single sample
            # Note: sklearn expects 2D array [n_samples, n_features]
            prediction_class = int(self.model.predict([feature_vector])[0])

            # Get probability/confidence array for all classes
            proba = self.model.predict_proba([feature_vector])[0]
            probability = float(max(proba))

            # SMART LOGIC: If top prediction is "Other/Unknown" (class 7), use second-best
            OTHER_UNKNOWN_CLASS = 7
            if prediction_class == OTHER_UNKNOWN_CLASS:
                # Find second-highest confidence
                top_two_indices = (-proba).argsort()[:2]  # Get top 2 class indices
                prediction_class = int(top_two_indices[1])  # Use second-best class
                probability = float(proba[prediction_class])  # Get its confidence
                logger.info(f"⚠️  Top prediction was Other/Unknown, using second-best: {CLASS_NAMES[prediction_class]}")

            # Map class number to class name
            class_name = CLASS_NAMES[prediction_class] if prediction_class < len(CLASS_NAMES) else "Unknown"

            # Determine risk level based on probability
            if probability >= 0.8:
                risk_level = "High"
            elif probability >= 0.6:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            # Create friendly explanation based on the predicted condition
            # Focus on actionable advice without technical model details
            if class_name == "Healthy":
                explanation = "Keep up a healthy lifestyle!"
            elif class_name == "Arthritis":
                explanation = "Consider low-impact exercises and consult with your doctor about pain management options."
            elif class_name == "Asthma":
                explanation = "Work with your healthcare provider on an asthma action plan and manage triggers."
            elif class_name == "Cancer":
                explanation = "Please consult with an oncologist immediately for proper evaluation and care."
            elif class_name == "Diabetes":
                explanation = "Monitor your blood sugar levels and work with your healthcare provider on a diabetes management plan."
            elif class_name == "Hypertension":
                explanation = "Monitor your blood pressure regularly and follow your doctor's guidance on medication and lifestyle changes."
            elif class_name == "Obesity":
                explanation = "Consider speaking with a nutritionist or healthcare provider about a healthy weight management plan."
            elif class_name == "Other/Unknown":
                explanation = "Please consult with a healthcare provider for proper evaluation and personalized advice."
            else:
                explanation = "Please consult with a healthcare professional for proper diagnosis and treatment."

            result = PredictionResponse(
                prediction=prediction_class,
                probability=probability,
                risk_level=risk_level,
                explanation=explanation
            )

            logger.info(f"✅ Prediction: {class_name} (class {prediction_class}, confidence: {probability*100:.1f}%)")
            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise


# Global predictor instance (loaded once)
_predictor_instance = None


def get_predictor() -> Predictor:
    """
    Get or create global predictor instance (lazy loading).

    Returns:
        Predictor instance

    Raises:
        FileNotFoundError: If model file not found
    """
    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = Predictor()

    return _predictor_instance


def reload_predictor() -> Predictor:
    """
    Force reload of predictor (useful for testing).

    Returns:
        New Predictor instance
    """
    global _predictor_instance
    _predictor_instance = Predictor()
    return _predictor_instance

import logging
from typing import List, Dict, Any
from app.config import FEATURE_RANGES, DEFAULT_MODEL_FEATURES

logger = logging.getLogger(__name__)


def validate_feature(name: str, value: Any) -> Any:
    """
    Validate a single feature value against its constraints.

    Args:
        name: Feature name
        value: Value to validate

    Returns:
        Validated value or None if invalid
    """
    if value is None:
        return None

    if name not in FEATURE_RANGES:
        return None

    min_val, max_val, expected_type = FEATURE_RANGES[name]

    # Convert to expected type
    try:
        converted = expected_type(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {name}={value} to {expected_type}")
        return None

    # Check range
    if not (min_val <= converted <= max_val):
        logger.warning(f"{name}={converted} is outside valid range [{min_val}, {max_val}]")
        return None

    return converted


def prepare_features_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and prepare features dictionary.

    Args:
        state: Dictionary of features from memory

    Returns:
        Dictionary with validated features
    """
    validated = {}

    for feature in DEFAULT_MODEL_FEATURES:
        value = state.get(feature)
        validated[feature] = validate_feature(feature, value)

    return validated


def prepare_feature_vector(state: Dict[str, Any]) -> List[float]:
    """
    Convert state dict to feature vector for ML model.

    The vector must have features in the exact order expected by the model.
    Features are ordered as in DEFAULT_MODEL_FEATURES.
    Missing values are filled with 0.0 (neutral value).

    Args:
        state: Dictionary with feature values from memory

    Returns:
        List of 16 floats ready for ML model prediction
    """
    # Define feature order (MUST match training data order)
    feature_order = DEFAULT_MODEL_FEATURES

    vector = []

    for feature_name in feature_order:
        value = state.get(feature_name)

        if value is not None:
            try:
                vector.append(float(value))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {feature_name}={value} to float, using 0.0")
                vector.append(0.0)
        else:
            # Use 0.0 for missing values (neutral/default)
            vector.append(0.0)

    assert len(vector) == 16, f"Expected 16 features, got {len(vector)}"
    return vector


def count_collected_features(state: Dict[str, Any]) -> int:
    """
    Count how many features have been collected (non-null).

    Args:
        state: Dictionary with feature values

    Returns:
        Count of non-null features
    """
    return sum(1 for v in state.values() if v is not None)


def is_ready_for_prediction(state: Dict[str, Any], min_features: int = 14) -> bool:
    """
    Check if enough features collected for reliable prediction.

    Args:
        state: Dictionary with feature values
        min_features: Minimum features needed (default 14/16)

    Returns:
        True if ready for prediction
    """
    collected = count_collected_features(state)
    return collected >= min_features

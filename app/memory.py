import logging
from typing import Dict, Any

from app.config import DEFAULT_MODEL_FEATURES, FEATURE_RANGES

logger = logging.getLogger(__name__)


def initialize_state() -> Dict[str, Any]:
    """
    Initialize conversation state with all 16 features set to None.

    Returns:
        Dictionary with all features initialized to None
    """
    state = {feature: None for feature in DEFAULT_MODEL_FEATURES}
    logger.debug(f"✅ State initialized with {len(state)} features")
    return state


def update_state(state: dict, new_data: dict) -> dict:
    """
    Merge newly extracted values into memory.
    Only updates non-null values (preserves existing data).
    Validates values are within acceptable ranges.

    Args:
        state: Current state dictionary
        new_data: Dictionary with newly extracted features

    Returns:
        Updated state dictionary
    """
    updated_count = 0
    for key, value in new_data.items():
        if value is not None:
            # Validate value is in acceptable range
            if key in FEATURE_RANGES:
                min_val, max_val, expected_type = FEATURE_RANGES[key]
                try:
                    converted = expected_type(value)
                    # Check range
                    if not (min_val <= converted <= max_val):
                        logger.debug(f"   SKIPPED {key}={value} (out of range [{min_val}, {max_val}])")
                        continue
                    value = converted
                except (ValueError, TypeError):
                    logger.debug(f"   SKIPPED {key}={value} (invalid type)")
                    continue

            old_value = state.get(key)
            state[key] = value
            if old_value != value:
                logger.debug(f"   Updated {key}: {old_value} → {value}")
                updated_count += 1

    if updated_count > 0:
        logger.debug(f"✅ State updated: {updated_count} features changed")
    return state


def get_missing_features(state: dict) -> list:
    """
    Return list of features that are still missing (None).

    Args:
        state: Current state dictionary

    Returns:
        List of feature names with None values
    """
    missing = [k for k, v in state.items() if v is None]
    logger.debug(f"❓ Missing features: {len(missing)}/16 - {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return missing


def get_state_summary(state: dict) -> Dict[str, Any]:
    """
    Get a summary of current state.

    Args:
        state: Current state dictionary

    Returns:
        Summary with counts and status
    """
    total = len(state)
    collected = sum(1 for v in state.values() if v is not None)
    missing = total - collected

    return {
        "total_features": total,
        "collected": collected,
        "missing": missing,
        "percentage": (collected / total * 100) if total > 0 else 0,
        "state": state.copy()
    }
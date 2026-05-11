"""
FastAPI REST API layer wrapping the existing medical chatbot logic.
Converts the Gradio interface to a REST API for React frontend integration.
"""

import hashlib
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json

from app.main import (
    chat_fn,
    _load_state,
    _save_state,
    _parse_response,
    _get_question_hint,
)
from app.services.llm_extractor import extract_features_from_text
from app.services.feature_builder import count_collected_features, is_ready_for_prediction, prepare_feature_vector
from app.services.predictor import get_predictor
from app.memory import initialize_state, update_state, get_missing_features
from app.config import DEFAULT_MODEL_FEATURES, MIN_FEATURES_FOR_PREDICTION, CLASS_NAMES
from app.utils.helpers import generate_question, prioritize_features
from app.services.session_manager import get_session_manager

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Medical Diagnosis AI", description="REST API for medical health prediction")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session manager
session_manager = get_session_manager()


# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    """Chat message request"""
    session_id: Optional[str] = None
    message: str
    history: list = []


class ChatResponse(BaseModel):
    """Chat response with features and state"""
    session_id: str
    message: str
    features: Dict[str, Any]
    collected_count: int
    total_features: int = 16
    is_complete: bool
    prediction: Optional[Dict[str, Any]] = None
    hint: str = ""


class ResetRequest(BaseModel):
    """Reset session request"""
    session_id: str


class SessionStateResponse(BaseModel):
    """Session state response"""
    session_id: str
    features: Dict[str, Any]
    collected_count: int
    total_features: int = 16


# ===== Helper Functions =====

def _extract_feature_from_question(question: str) -> str:
    """Extract the feature name from a generated question string"""
    # Questions are formatted as "**FeatureName:** question text"
    import re
    match = re.search(r'\*\*([^*]+)\*\*:', question)
    if match:
        return match.group(1)
    return None


def _is_greeting_or_introduction(user_message: str, extracted_features: dict) -> bool:
    """
    Detect if the user sent a greeting or introduction without medical metrics.
    Returns True if it's a greeting/intro, False if it contains medical data.
    """
    # Check if any medical features were extracted
    has_medical_data = any(value is not None for value in extracted_features.values())
    if has_medical_data:
        return False

    # Check for greeting/intro keywords
    greeting_keywords = [
        'hi', 'hello', 'hey', 'greetings', 'intro', 'introduction',
        'my name is', 'i am', "i'm", 'call me', 'you can call me',
        'nice to meet', 'pleased to meet', 'good morning', 'good afternoon',
        'good evening', 'how are you', 'what\'s up', 'howdy', 'welcome'
    ]

    text_lower = user_message.lower()
    return any(keyword in text_lower for keyword in greeting_keywords)


def _extract_name_from_greeting(user_message: str) -> str:
    """Extract user's name from greeting if provided"""
    import re
    text_lower = user_message.lower()

    # Try patterns in order of specificity (most specific first)
    patterns = [
        # "my name is [Name]" or "my name's [Name]"
        r'my\s+name\s+(?:is|\'s)\s+([a-zA-Z]+)',

        # "i am [Name]" or "i'm [Name]" - but only if it's at the end or followed by period/comma
        r'i\s+(?:am|\'m)\s+([a-zA-Z]+)(?:\s|,|\.|$)',

        # "this is [Name]" or "this is my name [Name]"
        r'this\s+is\s+(?:my\s+)?(?:name\s+)?([a-zA-Z]+)(?:\s|,|\.|$)',

        # "call me [Name]"
        r'call\s+me\s+([a-zA-Z]+)',

        # "you can call me [Name]"
        r'you\s+can\s+call\s+me\s+([a-zA-Z]+)',

        # Simple "hello [Name]" or "hi [Name]" - but only a single word after greeting
        # This is last resort and only matches if nothing else worked
        r'(?:hello|hi|hey)\s+(?:there\s+)?([a-zA-Z]+)(?:\s|,|\.|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).capitalize()
            return name

    return None


def _build_acknowledgment(extracted: dict) -> str:
    """Build acknowledgment message from extracted features"""
    extracted_items = []
    for feature, value in extracted.items():
        if value is not None and feature in DEFAULT_MODEL_FEATURES:
            extracted_items.append(f"{feature}: {value}")

    if not extracted_items:
        return ""

    # Format all extracted items with proper grammar
    if len(extracted_items) == 1:
        return f"✓ Got your {extracted_items[0]}"
    elif len(extracted_items) == 2:
        return f"✓ Got your {extracted_items[0]} and {extracted_items[1]}"
    else:
        # Join all but last with commas, then add "and" before the last one
        all_but_last = ', '.join(extracted_items[:-1])
        return f"✓ Got your {all_but_last}, and {extracted_items[-1]}"


# ===== API Endpoints =====

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Medical Diagnosis AI"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Send a message and get AI response with updated features.

    Handles:
    - Session ID generation if not provided
    - Feature extraction from user message
    - State persistence
    - Prediction when all 16 features collected
    """
    try:
        # Generate or use session ID
        if req.session_id:
            session_id = req.session_id
        else:
            # Generate from first message hash
            session_id = "sess_" + hashlib.md5(req.message.encode()).hexdigest()[:8]
            logger.info(f"🔐 Created new session: {session_id}")

        # Load persisted state
        state = _load_state(session_id)

        # Get the last asked feature (for contextual number extraction)
        # This is stored as a special key in the state
        pending_feature = state.pop("__pending_feature__", None)

        # ===== CONTEXT-AWARE FALLBACK (Step 1) =====
        # Before running heavy regex/LLM extraction, intelligently extract based on pending feature type
        extracted = {feature: None for feature in DEFAULT_MODEL_FEATURES}

        if pending_feature:
            # Binary features: Smoking, Alcohol, Family History
            binary_features = ["Smoking", "Alcohol", "Family History"]

            # Feature-specific valid ranges for numeric validation
            feature_ranges = {
                "Age": (18, 100),
                "Glucose": (70, 400),
                "HbA1c": (3, 15),
                "BMI": (10, 60),
                "Cholesterol": (100, 400),
                "Triglycerides": (20, 500),
                "Blood Pressure": (60, 200),
                "Physical Activity": (0, 24),
                "Sleep Hours": (0, 24),
                "Stress Level": (1, 10),
                "Oxygen Saturation": (80, 100),
                "LengthOfStay": (0, 365),
                "Diet Score": (1, 10),
            }

            # ===== HANDLE BINARY FEATURES (YES/NO) =====
            if pending_feature in binary_features:
                text_lower = req.message.lower()
                # Check for explicit "yes" or affirmative phrases
                if re.search(r'\b(?:yes|yeah|yep|sure|true|positive|have|i do|i am|smoker|drink|drinker|smoking|drink)', text_lower):
                    extracted[pending_feature] = 1
                    logger.info(f"✅ Context-Aware Fallback (Binary): {pending_feature} = 1 (from input: '{req.message}')")
                # Check for explicit "no" or negative phrases
                elif re.search(r'\b(?:no|nope|false|negative|don\'t|dont|don\'?t|never|stopped|quit|non[- ]smok)', text_lower):
                    extracted[pending_feature] = 0
                    logger.info(f"✅ Context-Aware Fallback (Binary): {pending_feature} = 0 (from input: '{req.message}')")

            # ===== HANDLE NUMERIC FEATURES (NUMBERS) =====
            elif pending_feature in feature_ranges:
                # Try to extract a simple number from the input
                number_match = re.search(r'(\d+(?:\.\d{1,2})?)', req.message)
                if number_match:
                    try:
                        number = float(number_match.group(1))
                        min_val, max_val = feature_ranges[pending_feature]

                        # If the number is in valid range, use it!
                        if min_val <= number <= max_val:
                            extracted[pending_feature] = number
                            logger.info(f"✅ Context-Aware Fallback (Numeric): {pending_feature} = {number} (from input: '{req.message}')")
                    except (ValueError, IndexError):
                        pass

        # If contextual fallback didn't extract anything, run full regex/LLM extraction
        if not any(v is not None for v in extracted.values()):
            extracted = extract_features_from_text(req.message, pending_feature=pending_feature)
            logger.info(f"🔄 Full extraction used (contextual fallback didn't match)")

        # Check if this is just a greeting/introduction without medical data
        if _is_greeting_or_introduction(req.message, extracted):
            # Handle greeting conversationally
            name = _extract_name_from_greeting(req.message)
            if name:
                greeting_response = f"Hello {name}! This is your medical assistant. I'm here to help assess your health. Please tell me about your health or share some of your medical metrics so we can begin."
            else:
                greeting_response = "Hello! This is your medical assistant. I'm here to help assess your health. Please tell me about your health or share some of your medical metrics so we can begin."

            return ChatResponse(
                session_id=session_id,
                message=greeting_response,
                features=state,
                collected_count=0,
                is_complete=False,
                hint=""
            )

        # Update state with extracted features
        state = update_state(state, extracted)

        # Count collected features
        collected = count_collected_features(state)
        missing = get_missing_features(state)

        # Check if ready for prediction
        if is_ready_for_prediction(state, MIN_FEATURES_FOR_PREDICTION):
            # All 16 features collected - make prediction
            feature_vector = prepare_feature_vector(state)
            predictor = get_predictor()
            pred_result = predictor.predict(feature_vector)

            # Remove internal tracking keys from state
            clean_state = {k: v for k, v in state.items() if not k.startswith("__")}

            pred_data = {
                "prediction_class": int(pred_result.prediction),
                "prediction_name": CLASS_NAMES[int(pred_result.prediction)],
                "confidence": float(pred_result.probability),
                "risk_level": pred_result.risk_level,
                "explanation": pred_result.explanation,
                "features": clean_state
            }

            # Simple completion message - full details now in DiagnosisCard component
            response_msg = "✅ Assessment Complete! Your diagnosis is ready below."

            _save_state(session_id, state)

            return ChatResponse(
                session_id=session_id,
                message=response_msg,
                features=state,
                collected_count=collected,
                is_complete=True,
                prediction=pred_data,
                hint=""
            )

        # Not complete yet - ask for next missing feature
        prioritized_missing = prioritize_features(missing)
        next_question = generate_question(prioritized_missing[:1])
        ack = _build_acknowledgment(extracted)
        remaining = 16 - collected

        response_msg = f"""{ack}

{next_question}

**{remaining} more pieces of information needed.**""" if ack else f"""{next_question}

**{remaining} more pieces of information needed.**"""

        # Get hint for the next question
        hint = _get_question_hint(next_question)

        # Store the pending feature for the next extraction (contextual number support)
        next_feature = _extract_feature_from_question(next_question)
        if next_feature:
            state["__pending_feature__"] = next_feature

        _save_state(session_id, state)

        # Remove internal tracking keys before returning to frontend
        clean_state = {k: v for k, v in state.items() if not k.startswith("__")}

        return ChatResponse(
            session_id=session_id,
            message=response_msg,
            features=clean_state,
            collected_count=collected,
            is_complete=False,
            hint=hint
        )

    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
def reset_endpoint(req: ResetRequest):
    """Reset a session - clear all features and start fresh"""
    try:
        session_manager = get_session_manager()
        success = session_manager.reset_session(req.session_id)

        if success:
            logger.info(f"✅ Reset session {req.session_id}")
            return {
                "success": True,
                "message": "Session reset successfully",
                "session_id": req.session_id
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except Exception as e:
        logger.error(f"❌ Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}", response_model=SessionStateResponse)
def get_session_endpoint(session_id: str):
    """Get current session state"""
    try:
        state = _load_state(session_id)
        collected = count_collected_features(state)

        return SessionStateResponse(
            session_id=session_id,
            features=state,
            collected_count=collected
        )

    except Exception as e:
        logger.error(f"❌ Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features")
def get_features_list():
    """Get list of all 16 features with their metadata"""
    from app.config import FEATURE_RANGES

    features_info = {}
    for feature in DEFAULT_MODEL_FEATURES:
        if feature in FEATURE_RANGES:
            min_val, max_val, _ = FEATURE_RANGES[feature]
            features_info[feature] = {
                "min": min_val,
                "max": max_val,
                "type": "numeric" if feature not in ["Smoking", "Alcohol", "Family History"] else "binary"
            }
        else:
            features_info[feature] = {"min": None, "max": None, "type": "unknown"}

    return {
        "total": len(DEFAULT_MODEL_FEATURES),
        "features": DEFAULT_MODEL_FEATURES,
        "metadata": features_info
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

import gradio as gr
import logging
import json
import re
import hashlib
from pathlib import Path
from typing import List, Tuple

from .services.llm_extractor import extract_features_from_text
from .services.feature_builder import (
    prepare_feature_vector,
    count_collected_features,
    is_ready_for_prediction
)
from .services.predictor import get_predictor
from .services.session_manager import get_session_manager
from .memory import initialize_state, update_state, get_missing_features
from .utils.helpers import generate_question, prioritize_features
from .config import DEFAULT_MODEL_FEATURES, MIN_FEATURES_FOR_PREDICTION, CLASS_NAMES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Session state storage
SESSION_DIR = Path("/tmp/deepsense_sessions")
SESSION_DIR.mkdir(exist_ok=True)

# Session manager
session_manager = get_session_manager()

def _get_session_id(history: List) -> str:
    """Generate session ID consistently across all turns

    Key insight: In Gradio ChatInterface:
    - Turn 1: history is empty
    - Turn 2+: history contains all prior messages, starting with first user message

    Strategy: When history is empty, use a temporary default and let it be overridden
    on the next turn when we have the actual first message.
    """
    try:
        first_msg = None

        # If we have history, always use the first message (most reliable)
        if history:
            # Gradio 6 format: list of dicts with 'role' and 'content'
            if isinstance(history[0], dict):
                first_msg = history[0].get("content", "")
            # Fallback for tuple format (Gradio 4 or mixed formats)
            elif isinstance(history[0], (tuple, list)):
                first_msg = history[0][0] if len(history[0]) > 0 else ""
            else:
                first_msg = str(history[0])

        # Handle case where first_msg is a list (Gradio 6 content can be list)
        if first_msg and isinstance(first_msg, list):
            first_msg = str(first_msg[0]) if first_msg else ""

        first_msg = str(first_msg) if first_msg else "default"

        # Generate session ID from first message hash
        session_id = "sess_" + hashlib.md5(first_msg.encode()).hexdigest()[:8]
        logger.debug(f"🔐 Session ID: {session_id} (from message: {first_msg[:50]}...)")
        return session_id

    except Exception as e:
        logger.warning(f"⚠️  Error generating session ID: {e}")
        return "sess_error"

def _load_state(session_id: str) -> dict:
    """Load state from file with detailed logging and error handling"""
    try:
        path = SESSION_DIR / f"{session_id}.json"
        if path.exists():
            with open(path, 'r') as f:
                state_data = json.load(f)
                collected = sum(1 for v in state_data.values() if v is not None)
                logger.info(f"📂 Loaded session {session_id}: {collected}/16 features")
                return state_data
        else:
            logger.info(f"📂 New session {session_id} (no prior file)")
            return initialize_state()
    except Exception as e:
        logger.error(f"❌ Error loading state: {e} - creating fresh state")
        return initialize_state()

def _save_state(session_id: str, state: dict):
    """Save state to file with detailed logging and proper error handling"""
    try:
        path = SESSION_DIR / f"{session_id}.json"
        with open(path, 'w') as f:
            json.dump(state, f)
        collected = sum(1 for v in state.values() if v is not None)
        logger.info(f"💾 Saved session {session_id}: {collected}/16 features to {path}")
    except Exception as e:
        logger.error(f"❌ Error saving state to {path}: {e}")


def _handle_simple_response(message: str, feature: str) -> dict:
    """
    Handle simple yes/no responses for binary features and numeric responses.
    Maps responses like 'yes', 'no', '45', '10 days', etc. to feature values.

    Args:
        message: User's message
        feature: Feature name being asked about

    Returns:
        Dictionary with feature mapped to value, or empty dict if not extractable
    """
    import re

    msg_lower = message.lower().strip()

    # Binary features: Smoking, Alcohol, Family History
    if feature in ["Smoking", "Alcohol", "Family History"]:
        if any(word in msg_lower for word in ["yes", "yep", "yeah", "yea", "true", "i have", "i do", "positive"]):
            return {feature: 1}
        elif any(word in msg_lower for word in ["no", "nope", "nah", "false", "i don't", "i do not", "negative", "neither"]):
            return {feature: 0}

    # Numeric features - extract first number from message
    numeric_features = [
        "Age", "Glucose", "HbA1c", "BMI", "Cholesterol", "Triglycerides",
        "Blood Pressure", "Physical Activity", "Sleep Hours", "Stress Level",
        "Diet Score", "LengthOfStay", "Oxygen Saturation"
    ]

    if feature in numeric_features:
        # Try to extract a number from the message
        numbers = re.findall(r"[-+]?\d*\.?\d+", message)
        if numbers:
            try:
                value = float(numbers[0]) if "." in numbers[0] else int(numbers[0])
                logger.debug(f"✅ Extracted {feature} = {value} from simple response")
                return {feature: value}
            except (ValueError, IndexError):
                pass

    return {}


def _extract_previously_asked_features(history: List[Tuple[str, str]]) -> set:
    """
    Extract what features have already been asked about in the conversation.
    This prevents repetition.

    Args:
        history: Conversation history (list of tuples)

    Returns:
        Set of features that have been asked about
    """
    asked_features = set()
    if not history:
        return asked_features

    question_keywords = {
        "Age": ["age", "years old"],
        "BloodPressure": ["blood pressure", "bp"],
        "Glucose": ["glucose", "blood sugar"],
        "BMI": ["bmi", "body mass"],
        "Cholesterol": ["cholesterol"],
        "HbA1c": ["hba1c", "hemoglobin"],
        "Triglycerides": ["triglycerides"],
        "Smoking": ["smoke", "smoking"],
        "Alcohol": ["drink", "alcohol"],
        "PhysicalActivity": ["exercise", "activity", "workout"],
        "SleepHours": ["sleep", "hours of sleep"],
        "StressLevel": ["stress"],
        "DietScore": ["diet", "nutrition"],
        "FamilyHistory": ["family history", "disease history"],
        "LengthOfStay": ["hospital", "stay", "days"],
        "OxygenSaturation": ["oxygen", "saturation"],
    }

    try:
        # Check bot's previous questions (Gradio 6 format)
        for item in history:
            try:
                # Gradio 6: dict with 'role' and 'content'
                if isinstance(item, dict):
                    if item.get("role") == "assistant":
                        bot_response = item.get("content", "")
                    else:
                        continue
                # Fallback for tuple format
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    _, bot_response = item[0], item[1]
                else:
                    continue

                if bot_response:
                    response_lower = str(bot_response).lower()
                    for feature, keywords in question_keywords.items():
                        for keyword in keywords:
                            if keyword in response_lower:
                                asked_features.add(feature)
                                break
            except (ValueError, IndexError, TypeError):
                # Skip malformed history items
                continue

        return asked_features
    except Exception as e:
        logger.warning(f"⚠️  Error extracting asked features: {e}")
        return set()


def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Main chat function for Gradio ChatInterface.

    Orchestrates the complete conversation flow:
    1. Extract features from user input (Groq LLM)
    2. For simple yes/no to binary features, use pattern matching
    3. Update memory with extracted features
    4. Check for missing features
    5. If all features ready, make prediction
    6. Otherwise, ask for next missing feature

    Args:
        message: User's current message
        history: Conversation history (not used but required by Gradio)

    Returns:
        Bot response (question or prediction)
    """
    try:
        # Get/create session and load persisted state
        # If history is empty (Turn 1), use current message for consistent session ID
        if not history:
            # Generate session_id from first user message to ensure consistency across turns
            session_id = "sess_" + hashlib.md5(message.encode()).hexdigest()[:8]
            logger.debug(f"🔐 Turn 1 Session ID: {session_id} (from current message)")
        else:
            # Turn 2+: Use first message from history
            session_id = _get_session_id(history)

        state = _load_state(session_id)

        # Normalize message
        msg_lower = message.lower().strip()

        # Check if conversation has started (by checking if state has any data)
        collected = count_collected_features(state)
        is_first_message = collected == 0  # True only if no features collected yet
        conversation_active = collected > 0
        previously_asked = _extract_previously_asked_features(history) if history else set()

        # Track last asked feature for simple response matching
        last_asked_feature = None

        # First message greeting (ONLY if truly first message AND looks like greeting)
        if is_first_message and len(msg_lower.split()) <= 6:  # Short messages are likely greetings
            logger.info("🎯 Starting new conversation")

            # Check if it's a simple greeting
            greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "name is", "i am", "i'm"]
            if any(word in msg_lower for word in greeting_words):
                # Extract name if mentioned (letters only, not numbers)
                name_match = re.search(r"(?:i'm|i am|my name is|name's)\s+([a-z]+)", msg_lower)
                user_name = name_match.group(1).capitalize() if name_match else "there"

                greeting_response = f"Hi {user_name}! Nice to meet you. 😊\n\n"
                greeting_response += """I'm here to help build your health profile. We'll gather 16 key health metrics in a natural, conversational way.

**To get started: What is your age?**"""
                _save_state(session_id, state)
                return _format_response(state, greeting_response)

        # Handle mid-conversation greetings (don't reset, reference last topic)
        if conversation_active:
            greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
            if any(word in msg_lower for word in greeting_words) and len(msg_lower.split()) <= 3:
                # It's a simple greeting mid-conversation, don't reset
                last_bot_message = history[-1][1] if history else ""
                context = "your health profile"
                if "age" in last_bot_message.lower():
                    context = "your age"
                elif "blood pressure" in last_bot_message.lower():
                    context = "your blood pressure"
                elif "glucose" in last_bot_message.lower():
                    context = "your glucose level"

                continuation = f"Hey! We were just talking about {context}. Do you have those numbers for me?"
                _save_state(session_id, state)
                return _format_response(state, continuation)

        # Handle reset command
        if msg_lower in ["reset", "clear", "new", "start over", "/reset"]:
            state = initialize_state()  # Fresh state
            _save_state(session_id, state)
            reset_msg = """Let's start fresh! 🌟

I'm ready to help build your health profile again. Just share your health information naturally, like:
- "I'm 45 years old and my glucose is 150"
- "I smoke and my stress level is 8"
- "My BMI is 28 and I exercise 5 hours per week"

What would you like to share first?"""
            return _format_response(state, reset_msg)

        # Step 1: Extract features (with safety)
        logger.info(f"👤 User: {message}")
        try:
            extracted = extract_features_from_text(message)
            if not extracted:
                extracted = {f: None for f in DEFAULT_MODEL_FEATURES}
        except Exception as e:
            logger.warning(f"Extraction failed: {e}, using empty")
            extracted = {f: None for f in DEFAULT_MODEL_FEATURES}

        logger.info(f"📊 Extracted: {[k for k,v in extracted.items() if v]}")

        # Step 1.5: Check if any features were extracted
        features_extracted = sum(1 for v in extracted.values() if v is not None)
        if features_extracted == 0:
            logger.debug(f"⚠️  No features extracted from: {message}")

        # Step 1b: If extraction didn't work and we asked about a feature, use simple pattern matching
        if last_asked_feature and all(v is None for v in extracted.values()):
            simple_response = _handle_simple_response(message, last_asked_feature)
            if simple_response:
                logger.info(f"✅ Matched simple response for {last_asked_feature}")
                extracted.update(simple_response)

        # Step 2: Update memory with extracted features (safe)
        try:
            state = update_state(state, extracted)
            collected = count_collected_features(state)
            logger.info(f"💾 Collected: {collected}/16")

            # Verify state integrity
            if state is None:
                logger.error("❌ State became None after update!")
                state = initialize_state()
            if not isinstance(state, dict):
                logger.error(f"❌ State is not a dict: {type(state)}")
                state = initialize_state()

        except Exception as e:
            logger.error(f"State update failed: {e}")
            state = initialize_state()
            collected = 0

        # Step 3: Check for missing features
        missing = get_missing_features(state)

        # Step 3.5: Smart handling - if nothing extracted and no features collected yet
        if features_extracted == 0 and collected == 0:
            logger.info("⚠️  User input is non-medical. Prompting for medical information with positive frame.")
            non_medical_msg = """I'm here to help build your health profile. To get started, could you tell me your age or any health details you'd like to share?

For example: "I'm 45 years old" or "My glucose is 150" — anything you're comfortable sharing helps!"""
            return _format_response(state, non_medical_msg)

        # Step 4: Decision - Can we make a prediction?
        if len(missing) == 0 or is_ready_for_prediction(state, MIN_FEATURES_FOR_PREDICTION):
            logger.info("🎯 All features collected! Making prediction...")
            last_asked_feature = None
            return _make_prediction(state, session_id)

        # Step 5: Not ready yet - ask for next missing features (1-2 at a time)
        else:
            if missing:
                last_asked_feature = missing[0]
            question_text = _ask_next_question(missing, len(missing), extracted)
            _save_state(session_id, state)  # Save state before returning
            return _format_response(state, "", question_text)

    except Exception as e:
        logger.error(f"❌ Error in chat: {e}", exc_info=True)
        logger.error(f"   History type: {type(history)}, length: {len(history) if history else 0}")
        logger.error(f"   State type: {type(state)}")
        logger.error(f"   Message: {message}")

        # Return user-friendly error message
        error_msg = "I encountered an issue processing your message. Could you try again or rephrase?"
        return _format_response(state or initialize_state(), error_msg)


def _make_prediction(state: dict, session_id: str = None) -> str:
    """
    Make ML prediction once all features are collected.

    Args:
        state: Dictionary with all medical features
        session_id: Session ID for saving final state

    Returns:
        Formatted prediction result message
    """
    try:
        # Prepare feature vector for ML model
        feature_vector = prepare_feature_vector(state)
        logger.info(f"🔢 Feature vector prepared: {feature_vector}")

        # Get predictor and make prediction
        predictor = get_predictor()
        result = predictor.predict(feature_vector)

        # Format response
        class_name = CLASS_NAMES[result.prediction] if result.prediction < len(CLASS_NAMES) else "Unknown"

        response = f"""
✅ **All Information Collected!**

---

### 🏥 Prediction Results

**Predicted Condition:** {class_name}
**Confidence Level:** {result.probability*100:.1f}%
**Risk Assessment:** **{result.risk_level}**

**Details:** {result.explanation}

---

### 📋 Features Used for Prediction:
"""
        # Add feature summary
        for feature in DEFAULT_MODEL_FEATURES:
            value = state.get(feature)
            response += f"\n• **{feature}:** {value if value is not None else 'Not provided'}"

        response += """

---

### ⚠️ Important Disclaimer
**This is a demonstration tool only.** The prediction should never be used as a substitute for professional medical advice. Please consult with a qualified healthcare professional to discuss these results and your health concerns.

---

**Thank you for providing this information. Your health matters!**
"""

        # Add JSON output for backend processing
        json_output = {
            "FINAL_DATA": {
                "prediction_class": result.prediction,
                "prediction_name": class_name,
                "confidence": round(result.probability * 100, 1),
                "risk_level": result.risk_level,
                "features": state.copy()
            }
        }

        # Add JSON to response (for backend to capture)
        response += f"\n\n```json\n{json.dumps(json_output, indent=2)}\n```"

        logger.info("✅ Prediction completed successfully")
        if session_id:
            _save_state(session_id, state)  # Save final state
        return _format_response(state, response)

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return f"❌ Error making prediction: {str(e)}. Please try again."


def _format_response(current_state: dict, acknowledgment: str = "", question: str = "") -> str:
    """
    Format response: ALWAYS show DATA_STATE JSON + RESPONSE text
    """
    # Build JSON state
    data_state = {}
    for feature in DEFAULT_MODEL_FEATURES:
        value = current_state.get(feature)
        data_state[feature] = value

    json_state = json.dumps(data_state)

    # Build response text
    response_text = acknowledgment
    if question:
        if acknowledgment:
            response_text += f"\n\n{question}"
        else:
            response_text = question

    # ALWAYS show both DATA_STATE and RESPONSE
    formatted = f"""DATA_STATE: {json_state}

RESPONSE: {response_text}"""

    return formatted


def _ask_next_question(missing_features: list, count_missing: int, last_extracted: dict = None) -> str:
    """
    Generate professional response asking for next missing features (1-2 at a time).
    Follows the Medical Data Analyst protocol with structured output.

    Args:
        missing_features: List of missing feature names
        count_missing: Count of missing features
        last_extracted: Dictionary of features just extracted

    Returns:
        Formatted response with internal state and assistant message
    """
    try:
        # Build acknowledgment of what was received
        acknowledgment = ""
        if last_extracted and any(v is not None for v in last_extracted.values()):
            extracted_items = []
            for k, v in last_extracted.items():
                if v is not None:
                    # Format value nicely
                    if v in [0, 1]:
                        display_val = "Yes" if v == 1 else "No"
                    else:
                        display_val = f"{v}"
                    extracted_items.append(f"{k} of {display_val}")

            if extracted_items:
                acknowledgment = f"✓ I've noted your {' and '.join(extracted_items)}."

        # Prioritize next questions
        prioritized_missing = prioritize_features(missing_features)
        next_questions = generate_question(prioritized_missing[:2])

        progress = f"**{count_missing} more pieces of information needed to complete your profile.**"

        full_question = f"{next_questions}\n\n{progress}"

        logger.info(f"❓ Asking for: {prioritized_missing[0] if prioritized_missing else 'unknown'}")
        return full_question

    except Exception as e:
        logger.error(f"❌ Error generating question: {e}", exc_info=True)
        return "❌ Error generating question. Please try again."


def reset_session(history: List) -> List:
    """
    Reset current session: clear all data and start fresh

    Args:
        history: Chat history (used to get session ID)

    Returns:
        New history with reset confirmation message
    """
    try:
        session_id = _get_session_id(history)

        # Reset session in manager
        success = session_manager.reset_session(session_id)

        if success:
            reset_msg = """🔄 **Session Reset Successfully!**

Your health profile has been cleared. Let's start fresh with a new assessment.

Just introduce yourself and tell me about your health:
- "Hi, I'm John, 45 years old, my glucose is 150"
- "I'm Sarah, smoke occasionally, stress level is 8"
- Or any other health information you'd like to share"""

            logger.info(f"✅ Session {session_id} reset by user")

            # Return new history with reset message (Gradio 6 format)
            return [{"role": "assistant", "content": reset_msg}]
        else:
            return [{"role": "assistant", "content": "❌ Error resetting session. Please try again."}]

    except Exception as e:
        logger.error(f"❌ Error in reset_session: {e}")
        return [{"role": "assistant", "content": f"❌ Error: {str(e)}"}]


def _parse_response(raw: str) -> tuple:
    """
    Parse the raw chat_fn output to extract clean response and data state.

    Input format: "DATA_STATE: {...}\n\nRESPONSE: <text>"
    Returns: (clean_response_text, data_state_dict)
    """
    data_state = {}
    response = raw

    if "DATA_STATE:" in raw and "RESPONSE:" in raw:
        try:
            parts = raw.split("RESPONSE:", 1)
            data_part = parts[0].replace("DATA_STATE:", "").strip()
            response = parts[1].strip()
            data_state = json.loads(data_part)
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Could not parse DATA_STATE from response")
            data_state = {f: None for f in DEFAULT_MODEL_FEATURES}

    return response, data_state


def _get_question_hint(response: str) -> str:
    """
    Extract hint text based on which feature is being asked in the bot's response.
    Matches feature names in the response text.
    """
    hints = {
        "Blood Pressure": "Systolic pressure (e.g., 120). Normal: 90-120 mmHg.",
        "Glucose": "Blood glucose in mg/dL (e.g., 100). Normal: 70-140.",
        "HbA1c": "HbA1c percentage (e.g., 5.5%). Normal: 4-6%.",
        "BMI": "Body Mass Index (e.g., 25). Normal: 18.5-24.9.",
        "Age": "Age in years (e.g., 34). Range: 18-100.",
        "Cholesterol": "Total cholesterol in mg/dL (e.g., 180). Normal: <200.",
        "Triglycerides": "Triglycerides in mg/dL (e.g., 150). Normal: <150.",
        "Oxygen Saturation": "SpO2 percentage (example: 98). Normal: 95-100%.",
        "Sleep Hours": "Hours of sleep per night (e.g., 7). Normal: 6-9.",
        "Stress Level": "Stress level 1-10 (e.g., 5). 1=very low, 10=very high.",
        "Physical Activity": "Exercise hours per week (e.g., 5).",
        "Diet Score": "Diet quality 1-10 (e.g., 7). 1=poor, 10=excellent.",
        "Smoking": "Do you smoke? Answer yes or no.",
        "Alcohol": "Do you drink alcohol? Answer yes or no.",
        "Family History": "Family history of disease? Answer yes or no.",
        "LengthOfStay": "Hospital stay in days (e.g., 3). Enter 0 if not hospitalized.",
    }

    response_lower = response.lower()
    for feature, hint in hints.items():
        if feature.lower() in response_lower:
            return hint

    return "Type your response naturally or just enter a number."


# Create Gradio UI with modern medical intake form design
css_styling = """
/* Base container */
.gradio-container {
    background: linear-gradient(135deg, #F5F3FF 0%, #FAF9FF 100%) !important;
    max-width: 900px !important;
    margin: auto !important;
    padding: 1rem !important;
}

/* Header card */
.header-card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(124, 58, 237, 0.08);
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.app-title {
    color: #7C3AED;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    margin-bottom: 0.5rem;
}

.app-subtitle {
    color: #999;
    font-size: 0.95rem;
    margin: 0;
}

/* Main form card */
.form-card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(124, 58, 237, 0.1);
    padding: 2rem;
}

/* Question section header */
.question-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #F0E7FF;
}

.header-label {
    color: #7C3AED;
    font-weight: 600;
    font-size: 0.95rem;
}

.progress-label {
    color: #7C3AED;
    font-weight: 600;
    font-size: 0.95rem;
    text-align: right;
}

/* Progress bar */
.progress-bar-container {
    background: #EDE9FE;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.progress-bar-fill {
    height: 100%;
    background: #7C3AED;
    transition: width 0.3s ease;
}

/* Question text */
.question-text {
    font-size: 1.1rem;
    color: #1F2937;
    margin-bottom: 0.8rem;
    line-height: 1.6;
}

/* Hint text */
.hint-text {
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    padding: 0.75rem 1rem;
    background: #F9F7FF;
    border-left: 3px solid #DDD6FE;
    border-radius: 4px;
}

/* Input and buttons */
.input-group {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.5rem;
}

.input-box {
    flex: 1;
}

.btn-new {
    background: white !important;
    border: 1px solid #DDD !important;
    border-radius: 8px !important;
    color: #555 !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-new:hover {
    border-color: #7C3AED !important;
    color: #7C3AED !important;
}

.btn-submit {
    background: #7C3AED !important;
    border: 0 !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-submit:hover {
    background: #6D28D9 !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

/* Disclaimer */
.disclaimer {
    text-align: center;
    color: #999;
    font-size: 0.85rem;
    margin-top: 1rem;
}

/* Gradio textbox customization */
.textbox-input input {
    border-radius: 8px !important;
    border: 1px solid #DDD !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
}

.textbox-input input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
}
"""

with gr.Blocks() as demo:
    # State management
    chat_history = gr.State([])

    # Header section
    gr.HTML("""
    <div class="header-card">
        <h1 class="app-title">Medical Diagnosis AI</h1>
        <p class="app-subtitle">Clinical intake assessment powered by machine learning</p>
    </div>
    """)

    # Main form card
    with gr.Column(elem_classes="form-card"):
        # Question header with progress
        with gr.Row(elem_classes="question-header"):
            header_label = gr.HTML('<div class="header-label">Medical Intake Assistant<br>Question 1 of 16</div>')
            progress_label = gr.HTML('<div class="progress-label">0 / 16 questions</div>')

        # Progress bar
        gr.HTML('<div class="progress-bar-container"><div class="progress-bar-fill" id="progress-fill" style="width:0%"></div></div>')

        # Question display
        question_display = gr.Markdown(
            "**Hi! I'm your Medical Intake Assistant. Let's begin your health assessment.**\n\n**What is your age?**",
            elem_classes="question-text"
        )

        # Hint text
        hint_display = gr.Markdown(
            "Age in years (example: 34). Normal range: 18-100.",
            elem_classes="hint-text"
        )

        # Input row
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Enter your response...",
                show_label=False,
                lines=1,
                elem_classes="textbox-input",
                scale=8
            )
            new_btn = gr.Button("New", scale=1, elem_classes="btn-new")
            submit_btn = gr.Button("▶", scale=1, elem_classes="btn-submit")

        # Hidden chatbot for state management
        chatbot = gr.Chatbot(visible=False)

        # Disclaimer
        gr.HTML('<div class="disclaimer">⚠️ This is a demonstration tool. Always consult with healthcare professionals for medical decisions.</div>')

    # Event handler for submission
    def handle_submission(message, history):
        """Process user input and return updated UI elements"""
        if not message or not message.strip():
            return (
                history,  # Updated history
                "Please enter a response.",  # Question display
                "Try entering your answer.",  # Hint
                "",  # Clear input
                "0 / 16"  # Progress
            )

        # Call the existing chat function
        raw_response = chat_fn(message, history)
        clean_response, data_state = _parse_response(raw_response)

        # Update history
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": clean_response}
        ]

        # Calculate progress
        collected = sum(1 for v in data_state.values() if v is not None)
        progress_text = f"{collected} / 16"

        # Get hint for next question
        hint_text = _get_question_hint(clean_response)

        logger.info(f"📊 Progress: {collected}/16")

        return (
            new_history,
            clean_response,
            hint_text,
            "",  # Clear input for next message
            progress_text
        )

    # Handle Enter key and submit button
    submit_btn.click(
        fn=handle_submission,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, question_display, hint_display, msg_input, progress_label],
        queue=True
    )

    msg_input.submit(
        fn=handle_submission,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, question_display, hint_display, msg_input, progress_label],
        queue=True
    )

    # Reset button handler
    def reset_conversation():
        """Reset the conversation"""
        reset_history = reset_session([])
        return (
            reset_history,  # Reset history
            "**Hi! I'm your Medical Intake Assistant. Let's begin your health assessment.**\n\n**What is your age?**",
            "Age in years (example: 34). Normal range: 18-100.",
            "",
            "0 / 16"
        )

    new_btn.click(
        fn=reset_conversation,
        outputs=[chatbot, question_display, hint_display, msg_input, progress_label],
        queue=False
    )

if __name__ == "__main__":
    logger.info("🚀 Starting Medical Predictor Chatbot...")
    demo.launch()

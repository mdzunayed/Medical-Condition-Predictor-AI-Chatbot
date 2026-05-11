import json
import logging
import re
from llama_cpp import Llama

from app.config import LOCAL_LLM_PATH, LOCAL_LLM_N_CTX, LOCAL_LLM_N_GPU_LAYERS, LOCAL_LLM_TEMPERATURE, DEFAULT_MODEL_FEATURES

logger = logging.getLogger(__name__)

# Initialize local GGUF model (GLOBAL)
try:
    client = Llama(
        model_path=str(LOCAL_LLM_PATH),
        n_ctx=LOCAL_LLM_N_CTX,
        n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS,
        verbose=False,
    )
    logger.info("✅ Local GGUF model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load local GGUF model: {e}")
    client = None


def _extract_with_regex(user_input: str, aggressive: bool = True, pending_feature: str = None) -> dict:
    """
    SMART regex-based extraction with flexible patterns.
    Handles: verb tenses (is/was/being), connecting words (a/up to/hit),
    corrections (put 0/actually/not X), and multiple features in one sentence.

    Args:
        user_input: User's text input
        pending_feature: Optional feature name that was just asked for (for contextual number extraction)

    Returns:
        Dictionary with extracted features
    """
    result = {feature: None for feature in DEFAULT_MODEL_FEATURES}
    text = user_input.lower()

    # Helper function to extract a number safely with ultra-flexible matching
    def extract_number(text, keyword_pattern, value_type=int, min_val=None, max_val=None, allow_float=False):
        """Extract number with MAXIMUM flexibility for medical terminology.
        Handles: all verb tenses, multiple connectors, adverbs, typos, and word order variations."""

        # Comprehensive verb tenses and action words
        verbs = r'(?:is|was|were|being|am|are|equals|measure|measured|measures|rate|rated|rates|hit|reached|have|has|say|saying|said|indicate|indicates)'

        # Connector words for maximum flexibility (now includes = symbol)
        connectors = r'(?:up\s+to|actually|exactly|around|approximately|about|of|in|as|like|nearly|roughly|=)\s*'

        # Optional article before number
        article = r'(?:a|an|the)?\s*'

        # Modifier words that can appear between keyword and verb
        modifiers = r'(?:\s+(?:level|score|quality|quantity|range|reading|measurement|value))?'

        # All possible patterns, ordered by specificity (simplest first for better coverage!)
        patterns = [
            # Pattern 0: SIMPLEST - keyword + optional spaces/equals + number (e.g., "glucose 150", "glucose=150")
            rf'{keyword_pattern}\s*(?:=\s*)?(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 1: keyword + [level/score/etc] + verb + connectors + article + number (MOST FLEXIBLE!)
            rf'{keyword_pattern}{modifiers}\s+(?:{verbs})\s+(?:{connectors})?{article}(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 2: keyword + verb + connectors + article + number
            rf'{keyword_pattern}\s+(?:{verbs})\s+(?:{connectors})?{article}(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 3: keyword + connector words (without verb) + article + number
            rf'{keyword_pattern}\s+(?:{connectors}){article}(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 4: prefix + keyword + [level] + verb + connectors + article + number
            rf'(?:my|his|her|the|your)\s+{keyword_pattern}{modifiers}\s+(?:{verbs})\s+(?:{connectors})?{article}(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 5: prefix + keyword + optional verb + number
            rf'(?:my|his|her|the|your)\s+{keyword_pattern}\s+(?:{verbs})?\s*(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 6: keyword + optional verb + number (handles "glucose is 150")
            rf'{keyword_pattern}\s+(?:{verbs})?\s*(\d{{1,3}}(?:\.\d{{1,2}})?)',

            # Pattern 7: "let's say number" or "say number" pattern
            rf'(?:let\'s\s+)?say\s+(?:an?\s+)?(\d{{1,3}}(?:\.\d{{1,2}})?)\s+(?:out\s+of\s+10|for\s+{keyword_pattern})?',
        ]

        for pattern in patterns:
            try:
                match = re.search(pattern, text)
                if match:
                    val_str = match.group(1)
                    val = float(val_str) if (allow_float or '.' in val_str) else int(val_str)
                    if min_val is not None and max_val is not None:
                        if min_val <= val <= max_val:
                            return val
                    elif min_val is not None and val >= min_val:
                        return val
                    elif max_val is not None and val <= max_val:
                        return val
                    else:
                        return val if not (min_val or max_val) else None
            except (ValueError, IndexError, AttributeError, TypeError):
                continue
        return None

    try:
        # ===== AGE =====
        # Handle multiple formats: "i'm 45", "age 45", "45 years old", "45-year-old", "45 year old"
        age_patterns = [
            # "45 years old" or "45 year old"
            r'(\d{1,3})\s+years?\s+old',
            # "45-year-old" or "45year old"
            r'(\d{1,3})\s*-?\s*years?\s*-?\s*old',
            # Using the flexible extractor for other formats
            r'(?:age|i\s+(?:am|\'m))\s+(\d{1,3})',
            # "I'm 45" or "I am 45"
            r'(?:i\'m|i\s+am)\s+(\d{1,3})',
        ]
        age_val = None
        for age_pat in age_patterns:
            try:
                age_match = re.search(age_pat, text)
                if age_match:
                    age_candidate = int(age_match.group(1))
                    if 18 <= age_candidate <= 100:
                        age_val = float(age_candidate)
                        break
            except (IndexError, AttributeError, ValueError):
                continue
        result["Age"] = age_val

        # ===== GLUCOSE =====
        glucose_val = extract_number(text, r'(?:glucose|blood\s+sugar)', min_val=70, max_val=400)
        result["Glucose"] = float(glucose_val) if glucose_val else None

        # ===== HBA1C =====
        # Handles: "HbA1c was exactly 5.1%", "HbA1c up to 6.5", "my HbA1c is 5%", "my hba1c level is 10%"
        hba1c_patterns = [
            # "my hba1c level is 10%" or "hba1c level is 6%"
            r'(?:my\s+)?hba1c\s+(?:level)?\s+(?:is|was|being)?\s+(?:exactly|around|approximately)?\s+(?:up\s+to\s+)?(\d{1,2}(?:\.\d{1,2})?)',
            # "HbA1c was exactly 5.1%"
            r'hba1c\s+(?:is|was|being)?\s+(?:exactly|around|approximately)?\s+(?:up\s+to\s+)?(\d{1,2}(?:\.\d{1,2})?)',
            # "my HbA1c is 5.1"
            r'my\s+hba1c\s+(?:is|was)?\s+(?:exactly\s+)?(\d{1,2}(?:\.\d{1,2})?)',
            # Direct "HbA1c 5.1"
            r'hba1c\s+(\d{1,2}(?:\.\d{1,2})?)',
            # Just "5.1%" or "6%" after context (when HbA1c is implied)
            r'(?:hba1c\s+)?(?:level\s+)?(?:of\s+)?(\d{1,2}(?:\.\d{1,2})?)%',
        ]
        for hba_pat in hba1c_patterns:
            try:
                hba_match = re.search(hba_pat, text)
                if hba_match:
                    hba = float(hba_match.group(1))
                    if 3 <= hba <= 15:
                        result["HbA1c"] = float(hba)
                        break
            except (IndexError, AttributeError):
                continue

        # ===== BMI =====
        # Updated patterns to handle trailing punctuation (commas, periods)
        bmi_patterns = [
            r'bmi\s+(?:is\s+)?(\d{1,2})(?:\s|,|\.|$)',
            r'my\s+bmi\s+(?:is\s+)?(\d{1,2})(?:\s|,|\.|$)',
            r'body\s+mass\s+index\s+(?:is\s+)?(\d{1,2})(?:\s|,|\.|$)',
            # Also handle = symbol: "bmi = 29"
            r'bmi\s*=\s*(\d{1,2})',
            r'my\s+bmi\s*=\s*(\d{1,2})',
        ]
        for bmi_pat in bmi_patterns:
            bmi_match = re.search(bmi_pat, text)
            if bmi_match:
                bmi = int(bmi_match.group(1))
                if 10 <= bmi <= 60:
                    result["BMI"] = float(bmi)
                    break

        # ===== CHOLESTEROL =====
        chol_val = extract_number(text, r'cholesterol', min_val=100, max_val=400)
        result["Cholesterol"] = float(chol_val) if chol_val else None

        # ===== TRIGLYCERIDES =====
        trig_val = extract_number(text, r'triglycerides', min_val=20, max_val=500)
        result["Triglycerides"] = float(trig_val) if trig_val else None

        # ===== BLOOD PRESSURE =====
        # Special handling for BP: "120/80" or just systolic "160"
        # Simplified patterns to handle basic phrases like "blood pressure 130"
        bp_patterns = [
            # Systolic/diastolic format first (most specific)
            r'(?:blood\s+pressure|bp)\s*(?:is|was|=)?\s*(\d{2,3})\s*/\s*(\d{2,3})',

            # Simple: "blood pressure 130" or "bp 130"
            r'(?:blood\s+pressure|bp)\s+(\d{2,3})',

            # With equals: "blood pressure = 130"
            r'(?:blood\s+pressure|bp)\s*=\s*(\d{2,3})',

            # With verbs and modifiers
            r'(?:blood\s+pressure|bp)\s+(?:is|was|being|measures?|measured|hit|reached)?\s*(?:around|about|approximately|up\s+to)?\s*(\d{2,3})',

            # "my blood pressure 115" or "my bp is 130"
            r'my\s+(?:blood\s+pressure|bp)\s+(?:is|was)?\s*(\d{2,3})',

            # Just "bp" followed by number
            r'bp\s+(\d{2,3})',
        ]
        for bp_pat in bp_patterns:
            try:
                bp_match = re.search(bp_pat, text)
                if bp_match:
                    bp_val = int(bp_match.group(1))
                    if 60 <= bp_val <= 200:
                        result["Blood Pressure"] = float(bp_val)
                        break
            except (IndexError, AttributeError):
                continue

        # ===== PHYSICAL ACTIVITY =====
        # Handles: "exercise 5 hours", "play football for 4 hours", "walk to work 15 hours", "activity is 15", "walking 3 hours"
        activity_patterns = [
            r'(?:physical\s+)?activity\s+(?:is|was)?\s+(?:for\s+)?(?:about\s+)?(\d{1,2})\s*(?:hours?)?',
            r'(?:exercise|activity|workout|sport|play|football|basketball|swimming|running|walk|biking)\s+(?:for\s+)?(?:about\s+|around\s+)?(\d{1,2})\s*(?:hours?)?',
            r'(?:walk|exercise|activity|play)ing\s+(?:for\s+)?(?:about\s+)?(\d{1,2})\s*(?:hours?)?',
            r'(\d{1,2})\s*(?:hours?)\s+(?:of\s+)?(?:exercise|activity|walking|playing|workout)',
        ]
        for act_pat in activity_patterns:
            activity_match = re.search(act_pat, text)
            if activity_match:
                activity = int(activity_match.group(1))
                if 0 <= activity <= 24:
                    result["Physical Activity"] = float(activity)
                    break

        # ===== SLEEP HOURS =====
        # Handles: "sleep 8 hours", "sleeping 7 hours", "sleep 7", "get 6 hours of sleep", "sleep for 5 hours"
        # Also handles: "sleep hours 5", "sleeping hours is 5", "sleep hours is 7"
        sleep_patterns = [
            # NEW: "sleep hours 5" or "sleeping hours is 5" (hours BEFORE number)
            r'(?:sleep|sleeping)\s+(?:time|hours?)\s+(?:is|was|are|equals)?\s*(\d{1,2})',

            # "sleep for 5 hours" or "sleep 5 hours"
            r'(?:sleep|sleeping|get|getting)\s+(?:for\s+)?(?:about\s+|around\s+)?(\d{1,2})\s*(?:hours?)?',

            # "5 hours of sleep" or "5 hours sleep"
            r'(\d{1,2})\s*(?:hours?)\s+(?:of\s+)?sleep',
        ]
        for sleep_pat in sleep_patterns:
            sleep_match = re.search(sleep_pat, text)
            if sleep_match:
                sleep = int(sleep_match.group(1))
                if 0 <= sleep <= 24:
                    result["Sleep Hours"] = float(sleep)
                    break

        # ===== STRESS LEVEL =====
        # Handles: "stress 8", "stress level is 5", "I'd rate my stress level as a 3"
        stress_patterns = [
            # "I'd rate my stress level as a 3 out of 10"
            r'(?:rate|rated)\s+(?:my\s+)?stress\s+(?:level)?\s+(?:as|like)\s+(?:a|an)?\s*(\d{1,2})',
            # "stress level is/as a 5"
            r'stress\s+(?:level)?\s+(?:is|was|as)?\s+(?:a|an)?\s*(\d{1,2})',
            # "my stress level in 4" (typo handling - "in" for "is")
            r'(?:my\s+)?stress\s+(?:level)?\s+(?:in|is)\s+(?:a|an)?\s*(\d{1,2})',
            # Direct pattern
            r'(?:my\s+)?stress\s+(?:level)?\s+(?:a|an)?\s*(\d{1,2})',
        ]
        for stress_pat in stress_patterns:
            try:
                stress_match = re.search(stress_pat, text)
                if stress_match:
                    stress = int(stress_match.group(1))
                    if 1 <= stress <= 10:
                        result["Stress Level"] = float(stress)
                        break
            except (IndexError, AttributeError):
                continue

        # ===== OXYGEN SATURATION =====
        # Handles: "oxygen saturation 90%", "oxygen saturation level is 95%", "oxygen was 96%", "o2 96"
        oxygen_patterns = [
            # "oxygen saturation level is 95%" or "oxygen saturation is 95%"
            r'(?:oxygen|o2|o₂)\s+(?:saturation\s+)?(?:level\s+)?(?:is|was)?\s+(?:around\s+)?(\d{1,3})(?:%)?',
            # "oxygen saturation 90%"
            r'(?:oxygen|o2|o₂)\s+saturation\s+(?:level\s+)?(\d{1,3})(?:%)?',
            # "oxygen was 96%" or "o2 is 95"
            r'(?:oxygen|o2|o₂)\s+(?:was|is|being|level)?\s+(\d{1,3})(?:%)?',
            # "my oxygen was 90%"
            r'my\s+oxygen\s+(?:was|is)?\s+(\d{1,3})(?:%)?',
            # "SpO2 95" or "spo2 is 92%"
            r'(?:spo2|spo\s*2)\s+(?:is\s+)?(\d{1,3})(?:%)?',
            # Direct "o2 97" or "oxygen 95" (needs to be number at end)
            r'(?:oxygen|o2|o₂)\s+(\d{1,3})$',
        ]
        for o2_pat in oxygen_patterns:
            o2_match = re.search(o2_pat, text)
            if o2_match:
                o2 = int(o2_match.group(1))
                if 80 <= o2 <= 100:
                    result["Oxygen Saturation"] = float(o2)
                    break

        # ===== LENGTH OF STAY =====
        # Handles: "hospitalized for 2 days", "2 day hospital stay", "stay 4 days", "length of stay is 10", "i was in hospital for 5 days"
        stay_patterns = [
            # "length of stay is 10"
            r'length\s+of\s+stay\s+(?:is\s+)?(\d{1,3})',
            # "my length of stay is 10"
            r'my\s+length\s+of\s+stay\s+(?:is\s+)?(\d{1,3})',
            # "hospitalized for 4 days" or "hospitali... for 2 days"
            r'(?:hospitali[z]?e?d?|in\s+hospital)\s+(?:for\s+)?(\d{1,3})\s*(?:days?)?',
            # "stay for 5 days" or "hospital stay 4 days"
            r'(?:hospital\s+)?stay\s+(?:for\s+)?(\d{1,3})\s*(?:days?)?',
            # "for/in 2 days hospital/stay"
            r'(?:for|in)\s+(\d{1,3})\s*(?:days?)\s+(?:hospital|stay)',
            # "2 day hospital stay" or "5 days in hospital"
            r'(\d{1,3})\s*(?:days?)\s+(?:hospital|stay|in\s+hospital)',
        ]
        if 'hospital' in text or 'stay' in text or 'length' in text:
            for stay_pat in stay_patterns:
                try:
                    stay_match = re.search(stay_pat, text)
                    if stay_match:
                        stay = int(stay_match.group(1))
                        if 0 <= stay <= 365:
                            result["LengthOfStay"] = float(stay)
                            break
                except (AttributeError, IndexError):
                    continue

        # ===== SMOKING =====
        # Explicit corrections: "put 0 for smoking", "i stopped smoking", "put 0 for that"
        if re.search(r'(?:put|mark|set)\s+0\s+(?:for\s+)?(?:smoking|smok)', text):
            result["Smoking"] = 0
        elif re.search(r'(?:stopped|quit|don\'t)\s+(?:smok|smoke)', text):
            result["Smoking"] = 0
        elif re.search(r'non[- ]smok|nonsmoker|don\'t\s+smok|no\s+smok', text):
            result["Smoking"] = 0
        elif re.search(r'(?:smok|smoke|smoking|smoker)(?!\s+before)', text):
            result["Smoking"] = 1

        # ===== ALCOHOL =====
        # Explicit: "put 0 for alcohol", "don't drink", "wine daily" (means drinking)
        if re.search(r'(?:put|mark|set)\s+0\s+(?:for\s+)?(?:alcohol|drink)', text):
            result["Alcohol"] = 0
        elif re.search(r'(?:don\'t|no|don\'?t)\s+(?:drink|alcohol)', text):
            result["Alcohol"] = 0
        elif re.search(r'(?:wine|beer|alcohol|drink|drinking|glass|daily)', text):
            result["Alcohol"] = 1

        # ===== FAMILY HISTORY =====
        # Only match when context is clear - require "family", "history", "disease", or "condition"
        if re.search(r'(?:no|don\'t|don\'?t)\s+(?:family\s+)?(?:history|disease|condition)', text):
            result["Family History"] = 0
        elif re.search(r'(?:family\s+)?(?:history|disease|condition)', text):
            # Only match if family/history/disease/condition is explicitly mentioned
            # Don't match standalone "yes" - that's handled by Context-Aware Fallback for pending feature
            result["Family History"] = 1

        # ===== DIET SCORE =====
        # Handles: "diet score is 8", "diet quality is 5", "eat healthy so say 8"
        diet_patterns = [
            # "diet score is/quality is 8"
            r'diet\s+(?:score|quality|nutritional?\s+value)\s+(?:is|was)?\s+(?:an?\s+)?(\d{1,2})',
            # "nutrition score 8"
            r'nutrition\s+(?:score|quality)?\s+(?:is|was)?\s+(?:an?\s+)?(\d{1,2})',
            # "say an 8" pattern for diet (contextual)
            r'(?:diet|nutrition|healthy)\s+[^.]*\bsay\s+(?:an?\s+)?(\d{1,2})(?:\s+out\s+of\s+10)?',
            # "let's say an 8"
            r'let\'s\s+say\s+(?:an?\s+)?(\d{1,2})\s+(?:for\s+diet|for\s+nutrition|out\s+of\s+10)',
            # Direct "diet 8"
            r'diet\s+(\d{1,2})',
        ]
        for diet_pat in diet_patterns:
            try:
                diet_match = re.search(diet_pat, text)
                if diet_match:
                    diet = int(diet_match.group(1))
                    if 1 <= diet <= 10:
                        result["Diet Score"] = float(diet)
                        break
            except (IndexError, AttributeError):
                continue

        # ===== BUG FIX #1: CONTEXTUAL NUMBER EXTRACTION =====
        # If no features extracted yet and we have a pending feature, try standalone number
        if pending_feature and result[pending_feature] is None and not any(v is not None for v in result.values()):
            # Try to extract a standalone number for the pending feature
            standalone_number_match = re.search(r'\b(\d+(?:\.\d{1,2})?)\b', text)
            if standalone_number_match:
                number = float(standalone_number_match.group(1))

                # Validate the number based on the feature type
                feature_ranges = {
                    "Age": (18, 100),
                    "Glucose": (70, 400),
                    "HbA1c": (3, 15),
                    "BMI": (10, 60),
                    "Cholesterol": (100, 400),
                    "Triglycerides": (20, 500),
                    "Blood Pressure": (30, 220),
                    "Physical Activity": (0, 24),
                    "Sleep Hours": (0, 24),
                    "Stress Level": (1, 10),
                    "Oxygen Saturation": (80, 100),
                    "LengthOfStay": (0, 365),
                    "Diet Score": (1, 10),
                }

                if pending_feature in feature_ranges:
                    min_val, max_val = feature_ranges[pending_feature]
                    if min_val <= number <= max_val:
                        result[pending_feature] = float(number)
                        logger.info(f"✅ Contextual extraction: {pending_feature} = {number} (from standalone number)")

        extracted_keys = {k: v for k, v in result.items() if v is not None}
        logger.debug(f"✅ Regex extraction found: {list(extracted_keys.keys())}")
        logger.info(f"🔍 Regex result: {extracted_keys if extracted_keys else 'EMPTY'}")
        return result

    except Exception as e:
        logger.error(f"Regex extraction error: {e}")
        return {feature: None for feature in DEFAULT_MODEL_FEATURES}


def extract_features_from_text(user_input: str, pending_feature: str = None) -> dict:
    """
    Uses SMART extraction: Regex first (fast), then LLM (comprehensive), with smart fallbacks.
    Returns dictionary with all required keys (values are null if not extracted).

    Args:
        user_input: User's free-form text input
        pending_feature: Optional feature name that was just asked for (for contextual number extraction)

    Returns:
        Dictionary with all 16 features (values null if not found)
    """

    if not user_input or not user_input.strip():
        logger.warning("Empty user input received")
        return {feature: None for feature in DEFAULT_MODEL_FEATURES}

    # STEP 1: TRY REGEX FIRST (fast, reliable for common patterns)
    regex_result = _extract_with_regex(user_input, aggressive=True, pending_feature=pending_feature)
    extracted_keys = {k: v for k, v in regex_result.items() if v is not None}
    logger.info(f"🔍 Regex result: {extracted_keys if extracted_keys else 'EMPTY'}")

    if any(v is not None for v in regex_result.values()):
        logger.info(f"✅ Regex extraction succeeded: {list(extracted_keys.keys())}")
        return regex_result

    # STEP 2: If regex didn't work, try LLM (only if client available)
    if client is None:
        logger.warning("⚠️  Groq client not initialized - returning regex results (empty)")
        return regex_result

    prompt = f"""
You are a medical information extraction system. Extract health data from ANY mention in the text.

Features with VALID RANGES:
- Age (18-100): "I am 30", "30 years old", "age 45" → extract the number
- Glucose (70-400): any number mentioned with glucose/blood sugar
- HbA1c (3-15): any number with HbA1c/hemoglobin
- BMI (10-60): any number with BMI/body mass
- Cholesterol (100-400): any number with cholesterol
- Triglycerides (20-500): any number with triglycerides
- Blood Pressure (60-200): any number with BP/blood pressure
- Physical Activity (0-24): any number with exercise/activity/workout hours
- Sleep Hours (0-24): any number with sleep/hours slept
- Stress Level (1-10): any number (1-10) with stress
- Diet Score (1-10): any number (1-10) with diet/nutrition
- Smoking (0 or 1): "smoke/smoking/smoker" → 1, "don't/no smoking/non-smoker" → 0, "sometimes/occasionally" → 1
- Alcohol (0 or 1): "drink/alcohol/drinking" → 1, "don't drink/no alcohol" → 0, "sometimes" → 1
- Family History (0 or 1): "family history/disease history" → 1, "no family history" → 0
- LengthOfStay (0-365): any number with hospital/stay/days
- Oxygen Saturation (80-100): any number with oxygen/O2/saturation

EXTRACTION RULES:
1. Return ONLY valid JSON. No markdown, no text, no explanation
2. Use ALL 16 keys exactly as shown
3. Extract ANY number from text (age, measurements, etc)
4. For binary features (Smoking/Alcohol/Family History): ALWAYS return 0 or 1, never null if mentioned
5. For "sometimes/occasionally/rarely" with Smoking/Alcohol → use 1 (yes, they do it)
6. If value outside range, use null
7. If feature not mentioned at all, use null

Examples:
"I'm 30 years old and I smoke sometimes" → {{"Age": 30, "Smoking": 1, ...other null...}}
"my name is John, I don't drink" → {{"Alcohol": 0, ...other null...}}
"45 with diabetes" → {{"Age": 45, ...}}
"I exercise 5 hours" → {{"Physical Activity": 5, ...}}

User Input: "{user_input}"

Return ONLY JSON with 16 keys:"""

    try:
        # STEP 2A: Call local GGUF model with error handling
        response = client.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a strict JSON generator. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=LOCAL_LLM_TEMPERATURE,
            max_tokens=1000
        )

        content = response["choices"][0]["message"]["content"].strip()

        # Try to parse JSON
        try:
            data = json.loads(content)
            logger.debug(f"✅ Successfully extracted via LLM: {[k for k,v in data.items() if v]}")
        except json.JSONDecodeError:
            # Try removing markdown code blocks
            logger.debug(f"First parse failed, trying fallback parsing")
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:].strip()

            try:
                data = json.loads(content)
                logger.debug(f"✅ Fallback JSON parse succeeded")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Could not parse JSON: {e}")
                logger.error(f"   Content was: {content[:300]}")
                logger.warning(f"⚠️  LLM JSON parsing failed, falling back to regex results")
                data = {feature: None for feature in DEFAULT_MODEL_FEATURES}

        result = {feature: None for feature in DEFAULT_MODEL_FEATURES}
        result.update(data)

        # STEP 2B: Use regex as enhancement to LLM (fill gaps)
        regex_data = _extract_with_regex(user_input, aggressive=True)

        # Merge: LLM data takes priority, regex fills gaps
        llm_extracted = sum(1 for v in result.values() if v is not None)
        for feature, value in regex_data.items():
            if value is not None and result[feature] is None:
                result[feature] = value

        regex_filled = sum(1 for v in result.values() if v is not None) - llm_extracted
        if regex_filled > 0:
            logger.info(f"📝 LLM + Regex merge: LLM found {llm_extracted}, Regex filled {regex_filled} gaps")

        return result

    except Exception as e:
        error_msg = str(e)
        if "model_path" in error_msg.lower() or "no such file" in error_msg.lower():
            logger.error(f"❌ Local LLM Model Error: Model file not found. Ensure {LOCAL_LLM_PATH} exists.")
        elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
            logger.warning(f"⚠️  GPU inference error, falling back to regex-only. Check GPU availability or set n_gpu_layers=0.")
        else:
            logger.error(f"❌ Local LLM Error: {type(e).__name__}: {error_msg[:200]}")

        # Fallback: Return regex results if LLM fails (don't return empty)
        logger.info("⚠️  Falling back to regex-only extraction")
        regex_fallback = _extract_with_regex(user_input, aggressive=True)
        return regex_fallback
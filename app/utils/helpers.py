def prioritize_features(missing_features: list) -> list:
    """
    Prioritize missing features for asking in natural order.

    Priority: Critical Basics → Key Metrics → Lifestyle → Wellness → Medical
    """
    priority_order = [
        "Age", "Blood Pressure", "Glucose",  
        "BMI", "Cholesterol", "HbA1c",  
        "Smoking", "Alcohol", "Physical Activity", 
        "Sleep Hours", "Stress Level", "Diet Score",
        "Triglycerides", "Family History", "Oxygen Saturation", "LengthOfStay"
    ]

    sorted_features = sorted(missing_features,
                            key=lambda x: priority_order.index(x) if x in priority_order else 999)
    return sorted_features


def generate_question(missing_features: list) -> str:
    """
    Generate warm, empathetic questions for ONE missing feature at a time.
    This keeps the user focused on one health metric at a time, improving UX.

    Args:
        missing_features: List of missing feature names (typically just 1)

    Returns:
        Formatted question string asking for the first missing item
    """
    questions = {
        "Age": "Could you tell me your age?",
        "Glucose": "What's your typical glucose level (in mg/dL)?",
        "Smoking": "Do you smoke, or have you smoked in the past? (yes/no)",
        "Family History": "Is there a family history of disease or health conditions? (yes/no)",
        "HbA1c": "What's your HbA1c level (%)? (This measures average blood sugar over 3 months)",
        "Diet Score": "How would you rate your overall diet quality on a scale of 1-10?",
        "Alcohol": "Do you consume alcohol regularly? (yes/no)",
        "Physical Activity": "How many hours per week do you typically exercise or stay physically active?",
        "Blood Pressure": "What's your blood pressure reading? (e.g., 120)",
        "BMI": "What's your BMI (Body Mass Index)?",
        "Cholesterol": "What's your cholesterol level (in mg/dL)?",
        "Sleep Hours": "How many hours of sleep do you typically get per night?",
        "Stress Level": "How would you rate your current stress level on a scale of 1-10?",
        "Triglycerides": "What's your triglycerides level (in mg/dL)?",
        "Oxygen Saturation": "What's your oxygen saturation level (%)? (Normal is 95-100%)",
        "LengthOfStay": "How many days were you hospitalized, if applicable?"
    }

    if not missing_features:
        return ""

    feature = missing_features[0]
    return f"**{feature}:** {questions.get(feature, f'Please provide {feature}')}"
from typing import Optional
from pydantic import BaseModel, field_validator


class MedicalFeatures(BaseModel):
    """All 16 required medical features with validation"""

    Age: Optional[float] = None
    Glucose: Optional[float] = None
    HbA1c: Optional[float] = None
    BMI: Optional[float] = None
    Cholesterol: Optional[float] = None
    Triglycerides: Optional[float] = None
    Blood_Pressure: Optional[float] = None
    Physical_Activity: Optional[float] = None
    Sleep_Hours: Optional[float] = None
    Stress_Level: Optional[float] = None
    Diet_Score: Optional[float] = None
    Smoking: Optional[int] = None
    Alcohol: Optional[int] = None
    Family_History: Optional[int] = None
    LengthOfStay: Optional[int] = None
    Oxygen_Saturation: Optional[float] = None

    @field_validator("Age")
    @classmethod
    def validate_age(cls, v):
        if v is not None and not (0 <= v <= 150):
            raise ValueError("Age must be between 0 and 150")
        return v

    @field_validator("Glucose")
    @classmethod
    def validate_glucose(cls, v):
        if v is not None and not (70 <= v <= 400):
            raise ValueError("Glucose must be between 70 and 400")
        return v

    @field_validator("HbA1c")
    @classmethod
    def validate_hba1c(cls, v):
        if v is not None and not (3 <= v <= 15):
            raise ValueError("HbA1c must be between 3 and 15")
        return v

    @field_validator("BMI")
    @classmethod
    def validate_bmi(cls, v):
        if v is not None and not (10 <= v <= 60):
            raise ValueError("BMI must be between 10 and 60")
        return v

    @field_validator("Cholesterol")
    @classmethod
    def validate_cholesterol(cls, v):
        if v is not None and not (100 <= v <= 400):
            raise ValueError("Cholesterol must be between 100 and 400")
        return v

    @field_validator("Triglycerides")
    @classmethod
    def validate_triglycerides(cls, v):
        if v is not None and not (20 <= v <= 500):
            raise ValueError("Triglycerides must be between 20 and 500")
        return v

    @field_validator("Blood_Pressure")
    @classmethod
    def validate_blood_pressure(cls, v):
        if v is not None and not (60 <= v <= 200):
            raise ValueError("Blood Pressure must be between 60 and 200")
        return v

    @field_validator("Physical_Activity")
    @classmethod
    def validate_physical_activity(cls, v):
        if v is not None and not (0 <= v <= 24):
            raise ValueError("Physical Activity must be between 0 and 24 hours/week")
        return v

    @field_validator("Sleep_Hours")
    @classmethod
    def validate_sleep_hours(cls, v):
        if v is not None and not (0 <= v <= 24):
            raise ValueError("Sleep Hours must be between 0 and 24")
        return v

    @field_validator("Stress_Level")
    @classmethod
    def validate_stress_level(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Stress Level must be between 1 and 10")
        return v

    @field_validator("Diet_Score")
    @classmethod
    def validate_diet_score(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Diet Score must be between 1 and 10")
        return v

    @field_validator("Smoking", "Alcohol", "Family_History")
    @classmethod
    def validate_binary(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("Binary values must be 0 or 1")
        return v

    @field_validator("LengthOfStay")
    @classmethod
    def validate_length_of_stay(cls, v):
        if v is not None and not (0 <= v <= 365):
            raise ValueError("Length of Stay must be between 0 and 365 days")
        return v

    @field_validator("Oxygen_Saturation")
    @classmethod
    def validate_oxygen_saturation(cls, v):
        if v is not None and not (80 <= v <= 100):
            raise ValueError("Oxygen Saturation must be between 80 and 100%")
        return v

    class Config:
        use_enum_values = True


class PredictionRequest(BaseModel):
    """Request for prediction"""

    features: MedicalFeatures


class PredictionResponse(BaseModel):
    """Prediction response with confidence"""

    prediction: int  # 0 or 1 (disease class)
    probability: float  # 0.0 to 1.0
    risk_level: str  # "Low", "Medium", "High"
    explanation: str


class ExtractionResponse(BaseModel):
    """LLM extraction response"""

    extracted_features: MedicalFeatures
    confidence: float

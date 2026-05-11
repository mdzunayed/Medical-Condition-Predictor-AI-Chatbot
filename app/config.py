from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Paths - robust for both local and HF Spaces deployment
BASE_DIR = Path(__file__).parent.parent

# Try multiple possible model locations
_possible_paths = [
    BASE_DIR / "models" / "GradientBoosting_model.pkl",  # Local development
    BASE_DIR / "model" / "GradientBoosting_model.pkl",   # If named 'model' instead
    Path("/app/models/GradientBoosting_model.pkl"),      # HF Spaces absolute path
    Path.cwd() / "models" / "GradientBoosting_model.pkl", # Current working directory
]

MODEL_PATH = None
for path in _possible_paths:
    if path.exists():
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    MODEL_PATH = _possible_paths[0]

# Local GGUF Model Configuration
LOCAL_LLM_PATH = BASE_DIR / "llm" / "meta-llama-3.1-8b-instruct.Q4_K_M.gguf"
LOCAL_LLM_N_CTX = 4096
LOCAL_LLM_N_GPU_LAYERS = 0  # Set to -1 to offload all layers to GPU if available
LOCAL_LLM_TEMPERATURE = 0
LOCAL_LLM_TIMEOUT = 120  # Local inference is slower; allow more time

DEFAULT_MODEL_FEATURES = [
    "LengthOfStay",
    "Smoking",
    "Family History",
    "HbA1c",
    "Glucose",
    "Age",
    "Diet Score",
    "Alcohol",
    "Physical Activity",
    "Blood Pressure",
    "BMI",
    "Cholesterol",
    "Sleep Hours",
    "Stress Level",
    "Triglycerides",
    "Oxygen Saturation"
]

# Feature validation ranges (min, max, expected type)
FEATURE_RANGES = {
    "Age": (0, 150, float),
    "Glucose": (30, 400, float),
    "HbA1c": (3, 15, float),
    "BMI": (10, 60, float),
    "Cholesterol": (100, 400, float),
    "Triglycerides": (20, 500, float),
    "Blood Pressure": (40, 250, float),
    "Physical Activity": (0, 24, float),
    "Sleep Hours": (0, 24, float),
    "Stress Level": (1, 10, float),
    "Diet Score": (1, 10, float),
    "Smoking": (0, 1, int),
    "Alcohol": (0, 1, int),
    "Family History": (0, 1, int),
    "LengthOfStay": (0, 365, int),
    "Oxygen Saturation": (50, 100, float),
}

# Model configuration
MIN_FEATURES_FOR_PREDICTION = 16  
MAX_RETRIES = 3

# Prediction classes
CLASS_NAMES = [
    "Arthritis",
    "Asthma",
    "Cancer",
    "Diabetes",
    "Healthy",
    "Hypertension",
    "Obesity",
    "Other/Unknown",
]

DEBUG_MODE = True
CONVERSATION_MAX_TURNS = 20

# System Architecture & Data Flow

## 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     HUGGING FACE SPACES                          │
│                    (Free CPU Tier)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GRADIO UI (Chat Interface)                              │  │
│  │  - Display chat messages                                 │  │
│  │  - Take user input                                       │  │
│  │  - Show predictions                                      │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                           │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │  app/main.py                                              │  │
│  │  (Chat Function & Orchestration)                          │  │
│  │  - Coordinates all services                              │  │
│  │  - Manages conversation flow                             │  │
│  └────┬──────────────────┬──────────────────┬───────────────┘  │
│       │                  │                  │                   │
│ ┌─────▼──────┐ ┌────────▼──────┐ ┌────────▼──────┐             │
│ │ LLM         │ │ Memory        │ │ Predictor     │             │
│ │ Extractor   │ │ Manager       │ │ Service       │             │
│ │             │ │               │ │               │             │
│ │ Input:      │ │ Input:        │ │ Input:        │             │
│ │ "I'm 45..." │ │ Extracted     │ │ Feature       │             │
│ │             │ │ features      │ │ vector        │             │
│ │ Output:     │ │               │ │               │             │
│ │ JSON with   │ │ Output:       │ │ Output:       │             │
│ │ features    │ │ Full state    │ │ Prediction    │             │
│ └─────┬──────┘ │ (16 features) │ │ + confidence  │             │
│       │        └────────┬──────┘ └───────┬───────┘             │
│       │                 │                │                      │
│ ┌─────▼─────┐ ┌────────▼──┐ ┌───────────▼───┐                │
│ │ GROQ       │ │ app/      │ │ Feature       │                │
│ │ Llama 3    │ │ memory.py │ │ Builder       │                │
│ │ (Free API) │ │           │ │               │                │
│ │            │ │ State:    │ │ Validates &   │                │
│ │ Cloud-     │ │ {         │ │ prepares      │                │
│ │ based      │ │  Age: 45, │ │ feature       │                │
│ │            │ │  Glucose: │ │ vector for    │                │
│ │            │ │  150,     │ │ ML model      │                │
│ │            │ │  ...      │ │               │                │
│ │            │ │  ...      │ │ (Validates    │                │
│ │            │ │ }         │ │ ranges)       │                │
│ │            │ │           │ │               │                │
│ └────────────┘ └───────────┘ └────────┬──────┘                │
│                                        │                        │
│                       ┌────────────────▼──────────┐             │
│                       │ Predictor Service         │             │
│                       │ (app/services/           │             │
│                       │  predictor.py)           │             │
│                       │                          │             │
│                       │ Loads:                   │             │
│                       │ GradientBoosting_       │             │
│                       │  model.pkl              │             │
│                       │                          │             │
│                       │ Inputs: [16 floats]     │             │
│                       │ Outputs: class + prob   │             │
│                       └────────────┬─────────────┘             │
│                                    │                            │
│                       ┌────────────▼──────────┐                │
│                       │ scikit-learn           │                │
│                       │ GradientBoosting       │                │
│                       │ Classifier             │                │
│                       │                        │                │
│                       │ (Runs locally on CPU)  │                │
│                       └────────────────────────┘                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Sequence

### Step 1: User Inputs Text
```
User: "I'm 45 years old and my glucose is 150"
                    ↓
            [Sent to main.py]
```

### Step 2: LLM Extraction (Groq API)
```
app/main.py
    ↓
call: extract_features_from_text(message)
    ↓
llm_extractor.py
    ↓
Groq API (Cloud)
    │ 
    └─→ llama-3.1-8b-instant
        (Processes: "I'm 45... glucose 150...")
        ↓
    Returns JSON: {
        "Age": 45,
        "Glucose": 150,
        "Smoking": null,
        "HbA1c": null,
        ... (rest null)
    }
    ↓
[Returns to main.py]
```

### Step 3: Update Memory
```
current_state = {
    "Age": null,
    "Glucose": null,
    ... (all null)
}
        ↓
extracted = {"Age": 45, "Glucose": 150, ...}
        ↓
memory.update_state(current_state, extracted)
        ↓
new_state = {
    "Age": 45,
    "Glucose": 150,
    "Smoking": null,
    ... (rest null)
}
```

### Step 4: Check Missing Features
```
missing_features = get_missing_features(state)
    ↓
Result: ["Smoking", "Family History", "HbA1c", ... (12 more)]
    ↓
len(missing) = 14 features still needed
    ↓
DECISION: Not ready for prediction yet
```

### Step 5: Generate Question
```
next_missing = missing_features[0]  # "Smoking"
    ↓
question = generate_question("Smoking")
    ↓
Result: "Do you smoke? (yes/no)"
    ↓
Send to user in chat
```

### Step 6: User Answers (Loop Back to Step 1)
```
User: "No, I don't smoke"
    ↓
[Loop back to Step 1]
    ↓
(Repeat until all 16 features collected)
```

### Step 7: All Features Collected - Ready for Prediction
```
state = {
    "Age": 45,
    "Glucose": 150,
    "Smoking": 0,
    "Family History": 1,
    ... (all 16 features filled)
}
    ↓
missing_features = []  # Empty!
    ↓
DECISION: Ready for prediction!
```

### Step 8: Feature Builder - Prepare for ML Model
```
feature_builder.prepare_feature_vector(state)
    ↓
Validation:
  - Check each value in valid range
  - Convert types to float
  - Handle missing with defaults
    ↓
Output: [45.0, 150.0, 0.0, 1.0, ... (16 floats total)]
    ↓
This vector is ready for ML model
```

### Step 9: Prediction
```
feature_vector = [45.0, 150.0, 0.0, 1.0, ...]
    ↓
predictor = get_predictor()  # Loads model.pkl
    ↓
result = predictor.predict(feature_vector)
    ↓
Model processes:
  - Input: 16 features
  - Runs through GradientBoosting
  - Output: class (0 or 1) + probability
    ↓
Returns: PredictionResponse {
    "prediction": 1,
    "probability": 0.85,
    "risk_level": "High",
    "explanation": "Model predicts class 1 with 85% confidence"
}
```

### Step 10: Display Results to User
```
Chatbot: "✅ All information collected!
         
         Prediction Results:
         - Prediction: Class 1
         - Confidence: 85.0%
         - Risk Level: High
         - Details: Model predicts class 1 with 85% confidence"
```

---

## 🔄 Complete Conversation Example

```
USER: "I'm 45 years old, my glucose is 150, I smoke, and my stress is high"

STEP 1 (Extract):
  Groq extracts: {Age: 45, Glucose: 150, Smoking: 1, StressLevel: null}
  
STEP 2 (Update Memory):
  state = {Age: 45, Glucose: 150, Smoking: 1, StressLevel: null, ...}
  
STEP 3 (Check Missing):
  missing = ["Family History", "HbA1c", "StressLevel", ... (12 more)]
  
STEP 4 (Ask Question):
  BOT: "Do you have a family history of disease? (yes/no)"

USER: "Yes"

STEP 1 (Extract):
  Groq extracts: {FamilyHistory: 1}
  
STEP 2 (Update Memory):
  state = {Age: 45, Glucose: 150, Smoking: 1, FamilyHistory: 1, ...}
  
STEP 3 (Check Missing):
  missing = ["HbA1c", "StressLevel", ... (12 more)]
  
STEP 4 (Ask Question):
  BOT: "What is your HbA1c level?"

... (repeat until all 16 features)

STEP 7 (All Collected):
  state = {Age: 45, Glucose: 150, Smoking: 1, FamilyHistory: 1, 
           HbA1c: 7.2, BMI: 28, ... (all 16 filled)}
  
STEP 8 (Prepare):
  feature_vector = [45.0, 150.0, 1.0, 1.0, 7.2, ... (16 values)]
  
STEP 9 (Predict):
  ML Model processes vector
  Returns: {prediction: 1, probability: 0.82, risk_level: "High"}
  
STEP 10 (Display):
  BOT: "✅ Prediction: Class 1 (82% confidence)"
```

---

## 📁 File Dependencies & Data Flow

```
┌─────────────────┐
│  app/main.py    │ ← ORCHESTRATOR (coordinates everything)
└────────┬────────┘
         │
    ┌────┼────┬────────────────┬──────────────┐
    │    │    │                │              │
    ▼    ▼    ▼                ▼              ▼
┌───────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐
│memory │ │llm_      │ │feature_      │ │predictor    │
│.py    │ │extractor │ │builder.py    │ │.py          │
│       │ │.py       │ │              │ │             │
│ ┌──┐  │ │ ┌──────┐ │ │ ┌─────────┐  │ │ ┌────────┐  │
│ │  │  │ │ │Groq  │ │ │ │Validate │  │ │ │Load    │  │
│ │  │  │ │ │API   │ │ │ │Features │  │ │ │Model   │  │
│ └──┘  │ │ └──────┘ │ │ │         │  │ │ │ pkl    │  │
│       │ │          │ │ │Prepare  │  │ │ │        │  │
│Tracks │ │Extracts  │ │ │Vector   │  │ │ │Predict │  │
│State  │ │Features  │ │ │         │  │ │ │Result  │  │
│       │ │(JSON)    │ │ │         │  │ │ │        │  │
└───────┘ └──────────┘ └─────────┬─┘  │ └───┬────┬─┘  │
                                │    │      │    │    │
                                │    │      │    └────┼──┐
                                │    └──────┼─────────┘  │
                                │           │            │
                                ▼           ▼            ▼
                          ┌──────────────┐ ┌─────────────────┐
                          │app/schemas.py│ │models/          │
                          │(Validation)  │ │GradientBoosting │
                          │              │ │_model.pkl       │
                          │ ┌──────────┐ │ │                 │
                          │ │Pydantic  │ │ │ (scikit-learn)  │
                          │ │Models    │ │ │ Binary Classifier
                          │ │(Types)   │ │ │ 16 Features     │
                          │ └──────────┘ │ │ Input → Output  │
                          └──────────────┘ └─────────────────┘

                          ┌──────────────────────────────┐
                          │app/config.py                 │
                          │(Constants & Configuration)   │
                          │ • Paths                      │
                          │ • API settings               │
                          │ • Feature ranges             │
                          │ • Feature list               │
                          └──────────────────────────────┘

                          ┌──────────────────────────────┐
                          │app/utils/helpers.py          │
                          │(Question Mapping)            │
                          │ Feature → Question lookup    │
                          └──────────────────────────────┘
```

---

## 🔌 Integration Points

### 1. **Groq API ↔ LLM Extractor**
```
Input:  User text (string)
Process: HTTP request to Groq cloud
Output: JSON with features
Error:  Timeout, invalid JSON, API errors
```

### 2. **LLM Extractor ↔ Memory**
```
Input:  JSON from Groq
Process: Merge into state dict
Output: Updated state with new values
Error:  Type mismatch, null values (OK)
```

### 3. **Memory ↔ Feature Builder**
```
Input:  State dict with all features
Process: Validate ranges, convert types
Output: Prepared feature vector
Error:  Out of range, type errors
```

### 4. **Feature Builder ↔ Predictor**
```
Input:  Feature vector [16 floats]
Process: Load model, make prediction
Output: Prediction class + probability
Error:  Model not found, predict error
```

### 5. **Predictor ↔ Main App**
```
Input:  Request for prediction
Process: Get result from model
Output: PredictionResponse object
Error:  Model errors, input errors
```

---

## 🎯 Key Design Decisions

### Why Separate Services?
- **llm_extractor.py**: Handles all Groq API logic
- **feature_builder.py**: Handles all validation logic
- **predictor.py**: Handles all ML model logic
- **memory.py**: Handles state management
- **helpers.py**: Handles UI text generation

**Benefits**: 
- Easy to test each independently
- Easy to modify without breaking others
- Clear separation of concerns
- Reusable components

### Why Pydantic Schemas?
- Type validation
- Automatic conversion
- Error messages
- Documentation
- IDE autocomplete

### Why Groq Instead of Local LLM?
- Free tier (very generous)
- Fast inference (cloud-based)
- No GPU needed
- No local setup required
- Easy to deploy on CPU-only Spaces

### Why scikit-learn Model?
- Lightweight (fast on CPU)
- Works on HF Spaces free tier
- Easy to load/save (joblib)
- No deep learning overhead
- Deterministic results

---

## 🚀 Performance Considerations

### Typical Response Times
| Step | Time | Notes |
|------|------|-------|
| Groq API call | 1-3s | Cloud-based, depends on load |
| Feature extraction | <0.1s | JSON parsing |
| Memory update | <0.01s | Dict operations |
| Feature validation | <0.01s | Simple checks |
| Prediction | <0.1s | scikit-learn inference |
| **Total** | **1-3s** | User sees response in 1-3 seconds |

### Scalability
- **Concurrent Users**: HF Spaces free CPU can handle ~10-20 concurrent users
- **API Rate**: Groq free tier: very generous (1000s of calls/day)
- **Model Size**: GradientBoosting small (<5MB)
- **Memory Usage**: ~200MB for app + model

---

## 🔒 Security Considerations

### Secrets Handling
- GROQ_API_KEY: Stored in .env locally, HF Spaces secrets in production
- Model file: Public (no sensitive info)
- User data: In-memory only (not persisted)

### Input Validation
- All user inputs validated via Pydantic
- Feature ranges checked
- Type conversion safe

### Privacy
- No data logged
- No external APIs called except Groq
- No user data persisted

---

## ⚡ Optimization Opportunities (Future)

1. **Caching**: Cache similar predictions
2. **Batching**: Process multiple users' requests together
3. **Model**: Use faster model variant
4. **LLM**: Use smaller Groq model for faster extraction
5. **Storage**: Add database for history (optional)

---

This architecture is designed for:
- ✅ Clarity & maintainability
- ✅ Testability
- ✅ Deployability on free Spaces
- ✅ Easy debugging
- ✅ Extensibility

Ready to implement? Follow the GUIDES.md file step-by-step!

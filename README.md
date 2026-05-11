# MediHelp - Medical Predictor Chatbot

An AI-powered health assessment chatbot that engages in natural conversations to collect health metrics and predict medical conditions using machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18+-61DAFB) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)

---

## Features

- 🤖 **AI Conversations** — Intelligent feature extraction using Llama 3.1 8B GGUF
- 🏥 **16 Health Metrics** — Age, Glucose, BMI, Blood Pressure, Diet Score, and more
- 📊 **ML Predictions** — Gradient Boosting classifier for 8 medical conditions
- 📱 **Responsive UI** — Mobile & desktop support with Tailwind CSS
- 💾 **Session Persistence** — Maintains conversation state
- ⚡ **Local Inference** — No cloud API required, runs locally
- 🔄 **Smart Extraction** — Regex (fast) → LLM (comprehensive) fallback

---

<p align="center">
  <img src="./assets/logo.png" width="700" alt="">
</p>

---

## Tech Stack

**Backend:**
- FastAPI (REST API)

**Frontend:**
- React 18 + Vite
- Tailwind CSS
- Lucide React

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- 5GB disk space

### Installation & Run

```bash
# Clone repository
git clone https://github.com/zunayed/medical-predictor-chatbot.git
cd medical-predictor-chatbot

# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

**Terminal 1 — Backend:**
```bash
python server.py
# Runs on http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend && npm run dev
# Runs on http://localhost:5173
```

Open `http://localhost:5173` in your browser.

---

## How It Works

```
User Message
    ↓
[Feature Extraction: Regex → LLM]
    ↓
[Update Session State & Progress]
    ↓
[Generate Next Question]
    ↓
[Repeat until 16 features collected]
    ↓
[ML Prediction: Gradient Boosting]
    ↓
[Display Diagnosis & Risk Level]
```

### 16 Health Metrics

Age, Glucose, HbA1c, BMI, Cholesterol, Triglycerides, Blood Pressure, Physical Activity, Sleep Hours, Stress Level, Diet Score, Smoking, Alcohol, Family History, LengthOfStay, Oxygen Saturation

### Prediction Classes

Healthy, Diabetes, Hypertension, Asthma, Obesity, Arthritis, Cancer, Other/Unknown

---

## Project Structure

```
medical-predictor-chatbot/
├── app/
│   ├── api.py              # REST endpoints
│   ├── config.py           # Configuration
│   ├── services/
│   │   ├── llm_extractor.py
│   │   ├── predictor.py
│   │   └── session_manager.py
│   └── schemas.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── hooks/useChat.js
│   │   └── components/
│   └── package.json
├── models/
│   └── GradientBoosting_model.pkl
├── server.py
└── requirements.txt
```

---

## API Endpoints

### `POST /api/chat`
Send message, get response with extracted features.

```json
Request:
{"message": "I'm 45 and my glucose is 150", "session_id": "abc123"}

Response:
{
  "response": "What's your blood pressure?",
  "extracted_features": {"Age": 45, "Glucose": 150},
  "collected_features": 2,
  "is_complete": false
}
```

### `POST /api/reset`
Clear session and start new assessment.

### `GET /api/features`
Get all 16 metrics with descriptions.

---

## Performance

- **Feature Extraction:** 95% accuracy (regex), 92% (LLM)
- **Prediction Latency:** <100ms
- **LLM Response:** 2-5 seconds (CPU), <500ms (GPU)
- **Average Conversation:** 8-12 turns

---

## Deployment

**Docker:**
```bash
docker build -t medihelp .
docker run -p 8000:8000 -p 5173:5173 medihelp
```

**HuggingFace Spaces:** Ready to deploy (set `SPACE_ID` env var)

---

## ⚠️ Disclaimer

**For educational purposes only.** Not a replacement for professional medical diagnosis. Always consult a healthcare provider.

---

## Future Improvements

- Vision LLM for medical charts
- Ensemble models (4-model voting)
- SHAP feature importance
- Model monitoring dashboard
- Multi-language support

---

## Author

**Zunayed** — [GitHub](https://github.com/zunayed) | [Email](mailto:zunayed02@gmail.com)

---

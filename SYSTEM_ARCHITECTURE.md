# Medical Predictor Chatbot - Complete System Architecture & Data Flow

**Document Version:** 1.0  
**Date:** 2026-04-14  
**Stack:** React 18 (Frontend) + FastAPI (Backend) + Llama 3.1 8B (Local LLM) + scikit-learn (ML Model)

---

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Backend System](#backend-system)
4. [Frontend System](#frontend-system)
5. [Data Flow (End-to-End)](#data-flow-end-to-end)
6. [Component Details](#component-details)
7. [Key Features](#key-features)
8. [Deployment](#deployment)

---

## 🎯 System Overview

The **Medical Predictor Chatbot** is an AI-powered health assessment tool that:
- Engages users in natural conversation to collect health metrics
- Extracts medical features from free-form text using regex + local LLM
- Tracks session state and validates data ranges
- Predicts 8 health conditions using a trained scikit-learn gradient boosting model
- Displays personalized diagnosis with confidence scores

**Target Conditions:** Arthritis, Asthma, Cancer, Diabetes, Healthy, Hypertension, Obesity, Other/Unknown

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER BROWSER                               │
│                    (React Frontend)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  App.jsx (State & Layout Management)                     │  │
│  │  ├─ ChatWindow (Messages Display + Loading Indicator)    │  │
│  │  ├─ FeatureSidebar (Health Metrics Tracker)              │  │
│  │  │  └─ MetricItem (Individual Metric Rows)               │  │
│  │  │     └─ DietScoreModal (Interactive Calculator)        │  │
│  │  ├─ InputBar (User Message Input)                        │  │
│  │  └─ DiagnosisCard (Final Prediction Display)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↕ HTTP                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
           ┌──────────────────────────────────────┐
           │     FastAPI REST API Server          │
           │  (http://localhost:8000/api)         │
           ├──────────────────────────────────────┤
           │                                      │
           │  POST /api/chat                      │
           │  POST /api/reset                     │
           │  GET  /api/session/{session_id}      │
           │  GET  /api/features                  │
           │  GET  /health                        │
           │                                      │
           └──────────────────────────────────────┘
                      ↓        ↓        ↓
        ┌─────────────┴────┬───┴────┬──┴──────────┐
        ↓                  ↓        ↓             ↓
    ┌────────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ LLM         │  │ Feature  │ │ Session  │ │ Predictor│
    │ Extractor   │  │ Builder  │ │ Manager  │ │ Model    │
    │             │  │          │ │          │ │          │
    │ Llama 3.1   │  │ Extract  │ │ State    │ │ scikit   │
    │ 8B GGUF     │  │ Features │ │ Persist  │ │ -learn   │
    │             │  │ Validate │ │ Reset    │ │ Gradient │
    │ Regex +     │  │ Count    │ │ Retrieve │ │ Boosting │
    │ JSON Output │  │          │ │          │ │          │
    └────────────┘  └──────────┘ └──────────┘ └──────────┘
        ↓                ↓             ↓            ↓
    ┌─────────────────────────────────────────────────────┐
    │         File System & State Storage                  │
    │  - llm/meta-llama-3.1-8b-instruct.Q4_K_M.gguf      │
    │  - models/GradientBoosting_model.pkl                 │
    │  - .sessions/ (Session state JSON files)             │
    │  - config.py (Feature definitions & ranges)          │
    └─────────────────────────────────────────────────────┘
```

---

## 🔧 Backend System

### Backend Architecture

The backend is a **FastAPI REST API** that orchestrates:
1. **Feature Extraction** - Regex + Local LLM
2. **State Management** - Session persistence
3. **Health Prediction** - Scikit-learn model inference
4. **API Endpoints** - RESTful interface for frontend

### Key Backend Files

#### 1. **server.py** - Entry Point
```
Purpose: Start FastAPI server
- Detects environment (local dev vs HF Spaces)
- Configures host, port, reload based on environment
- Mounts static React frontend files
- Sets up logging

Flow:
  server.py → gets_server_config() → uvicorn.run(app)
  └─ Serves on localhost:8000 (dev) or 0.0.0.0:7860 (HF Spaces)
```

#### 2. **app/api.py** - REST Endpoints
```
Purpose: Define REST API endpoints and request/response handling

Endpoints:

POST /api/chat
├─ Input: ChatRequest {session_id, message, history}
├─ Process:
│  ├─ Generate session ID if new (hash of first message)
│  ├─ Load persisted session state
│  ├─ Extract features from user message (regex + LLM)
│  ├─ Check if greeting/intro (no medical data)
│  ├─ Update state with extracted features
│  ├─ Count collected features
│  ├─ If 16 features collected:
│  │  ├─ Prepare feature vector
│  │  ├─ Run scikit-learn prediction
│  │  ├─ Return prediction with class, confidence, explanation
│  │  └─ Save final state
│  ├─ Else:
│  │  ├─ Generate next contextual question
│  │  ├─ Build acknowledgment message
│  │  ├─ Store pending feature for next extraction
│  │  └─ Return question + feature count
│  └─ Save state for session
└─ Output: ChatResponse {session_id, message, features, is_complete, prediction}

POST /api/reset
├─ Input: ResetRequest {session_id}
├─ Process: Clear session state file
└─ Output: {success: true, session_id}

GET /api/session/{session_id}
├─ Process: Load and return session state
└─ Output: SessionStateResponse {session_id, features, collected_count}

GET /api/features
├─ Process: Return metadata about all 16 features
└─ Output: Feature ranges, types, validation info

GET /health
└─ Output: {status: "ok"}
```

#### 3. **app/services/llm_extractor.py** - Feature Extraction
```
Purpose: Extract medical metrics from free-form text

Two-Stage Extraction Strategy:

Stage 1: REGEX EXTRACTION (Fast)
├─ Run comprehensive regex patterns for all 16 features
├─ Patterns handle:
│  ├─ Multiple verb tenses (is/was/being/have)
│  ├─ Connector words (up to/about/around/exactly)
│  ├─ Typos and word order variations
│  ├─ Binary features (yes/no for Smoking, Alcohol, Family History)
│  ├─ Numeric features with range validation
│  └─ Contextual fallback (standalone numbers for pending feature)
└─ Return extracted values or null

Stage 2: LOCAL LLM EXTRACTION (Comprehensive)
├─ If regex found nothing, use local Llama 3.1 8B model
├─ Send prompt: "Extract these 16 medical features from user text"
├─ Model returns JSON with all 16 features
├─ Merge: LLM results + regex fills gaps
└─ Return merged results

Fallback Strategy:
├─ Regex succeeds → return immediately
├─ Regex fails + LLM available → try LLM
├─ LLM fails → return regex results (empty)
└─ No extraction → return all nulls

Feature Extraction Examples:

User: "I'm 45 and glucose is 180"
├─ Regex extracts: Age=45, Glucose=180
└─ Returns immediately (regex succeeded)

User: "I weigh 85kg and am 6 feet tall"
├─ Regex finds nothing (no direct feature keywords)
├─ LLM extracts: BMI≈31 (calculates from height/weight)
└─ Returns LLM results

User: "250" (with pending feature = "Glucose")
├─ Contextual extraction: Glucose=250 (validates against range 70-400)
└─ Returns extracted value

Features Extracted (16 total):
1. Age (18-100) - Years old
2. Glucose (70-400) - Blood sugar in mg/dL
3. HbA1c (3-15) - 3-month average blood sugar %
4. BMI (10-60) - Body Mass Index
5. Cholesterol (100-400) - Total cholesterol mg/dL
6. Triglycerides (20-500) - Triglycerides mg/dL
7. Blood Pressure (60-200) - Systolic mmHg
8. Physical Activity (0-24) - Hours per week
9. Sleep Hours (0-24) - Hours per night
10. Stress Level (1-10) - Subjective 1-10 scale
11. Diet Score (1-10) - Subjective 1-10 scale
12. Smoking (0 or 1) - Binary yes/no
13. Alcohol (0 or 1) - Binary yes/no
14. Family History (0 or 1) - Binary yes/no
15. LengthOfStay (0-365) - Hospital days
16. Oxygen Saturation (80-100) - SpO2 %
```

#### 4. **app/config.py** - Configuration
```
Purpose: Centralized configuration and constants

Key Configs:
- LOCAL_LLM_PATH: Path to Llama 3.1 8B GGUF model
- LOCAL_LLM_N_CTX: Context window (4096 tokens)
- LOCAL_LLM_TEMPERATURE: LLM temperature (0 = deterministic)
- MIN_FEATURES_FOR_PREDICTION: 16 (all features required)
- MODEL_PATH: Scikit-learn gradient boosting model
- FEATURE_RANGES: Valid ranges for each feature
- CLASS_NAMES: 8 possible diagnoses
  ├─ Arthritis
  ├─ Asthma
  ├─ Cancer
  ├─ Diabetes
  ├─ Healthy
  ├─ Hypertension
  ├─ Obesity
  └─ Other/Unknown
```

### Backend Data Flow: User Message to Prediction

```
1. User sends message: "I'm 45 and glucose is 180"

2. Frontend sends POST /api/chat:
   {
     "session_id": "sess_abc123" (or null if new),
     "message": "I'm 45 and glucose is 180",
     "history": [...]
   }

3. Backend receives request in chat_endpoint():
   ├─ Generate/use session_id
   ├─ Load persisted state from disk
   ├─ Extract pending feature from stored context
   └─ Initialize extracted dict (all 16 keys = null)

4. CONTEXT-AWARE FALLBACK:
   ├─ Check if pending_feature set (from previous question)
   ├─ If pending = "Glucose":
   │  ├─ Check message for number in valid range (70-400)
   │  ├─ "180" found and in range
   │  ├─ extracted["Glucose"] = 180
   │  └─ Log: "✅ Context-Aware Fallback: Glucose = 180"
   └─ If nothing matched, continue to next stage

5. REGEX EXTRACTION:
   ├─ Run regex patterns on message
   ├─ Pattern 1: r'(\d{1,3})\s+years?\s+old' → Age=45
   ├─ Pattern 2: r'glucose.*?(\d+)' → Glucose=180
   ├─ Return: {Age: 45, Glucose: 180, ...rest: null}
   └─ Since found matches, skip LLM

6. GREETING CHECK:
   ├─ extracted_features has medical data (Age, Glucose)
   ├─ Not a greeting (greeting would have no metrics)
   └─ Continue to state update

7. UPDATE STATE:
   ├─ Load state: {Age: null, Glucose: null, ...}
   ├─ Merge with extracted: {Age: 45, Glucose: 180, ...rest: null}
   ├─ New state: {Age: 45, Glucose: 180, ...rest: null}
   ├─ Count collected: 2/16 features
   ├─ Missing: [HbA1c, BMI, Cholesterol, ...13 more]

8. DECISION:
   ├─ 2 features < 16 required
   ├─ Not ready for prediction yet

9. GENERATE NEXT QUESTION:
   ├─ Prioritize missing features (by importance)
   ├─ Pick top 1: [HbA1c]
   ├─ Generate question: "**HbA1c:** What is your HbA1c level?"
   ├─ Extract feature name from question: "HbA1c"
   ├─ Store in state: state["__pending_feature__"] = "HbA1c"

10. BUILD RESPONSE MESSAGE:
    ├─ Acknowledgment: "✓ Got your Age: 45 and Glucose: 180"
    ├─ Question: "**HbA1c:** What is your HbA1c level?"
    ├─ Count: "**14 more pieces of information needed.**"
    └─ Get hint: "HbA1c is your average blood sugar over 3 months"

11. SAVE STATE:
    ├─ Write to disk: .sessions/sess_abc123.json
    └─ Contains: {Age: 45, Glucose: 180, __pending_feature__: "HbA1c"}

12. RETURN RESPONSE:
    {
      "session_id": "sess_abc123",
      "message": "✓ Got your Age: 45 and Glucose: 180\n\n**HbA1c:** What is...",
      "features": {Age: 45, Glucose: 180, ...rest: null},
      "collected_count": 2,
      "is_complete": false,
      "prediction": null,
      "hint": "..."
    }

---

PREDICTION SCENARIO (16th feature collected):

User: "My oxygen is 95" (after collecting 15 features)

Backend:
├─ Load state (15 features filled)
├─ Extract: Oxygen Saturation = 95
├─ Update state (now 16/16)
├─ is_ready_for_prediction() = true

PREDICTION:
├─ Prepare feature vector [values in model training order]
├─ Load scikit-learn gradient boosting model
├─ model.predict(vector) → class index (e.g., 3 = Diabetes)
├─ model.predict_proba(vector) → confidence (e.g., 0.87 = 87%)
├─ Get CLASS_NAMES[3] = "Diabetes"
├─ Get explanation from predictor service

RETURN:
{
  "session_id": "sess_abc123",
  "message": "✅ Assessment Complete! Your diagnosis is ready below.",
  "features": {all 16 filled},
  "collected_count": 16,
  "is_complete": true,
  "prediction": {
    "prediction_class": 3,
    "prediction_name": "Diabetes",
    "confidence": 0.87,
    "risk_level": "High",
    "explanation": "Based on your glucose level of 180 mg/dL...",
    "features": {...}
  }
}
```

---

## 🎨 Frontend System

### Frontend Architecture

The frontend is a **React 18 application** with:
- Component-based UI (functional components with hooks)
- Centralized state management via `useChat` hook
- Tailwind CSS for responsive styling
- Portal-based modals (rendered outside DOM hierarchy)

### Frontend Files Structure

```
frontend/
├─ src/
│  ├─ App.jsx                    # Main layout & state orchestration
│  ├─ hooks/
│  │  └─ useChat.js              # Chat logic + API calls
│  └─ components/
│     ├─ ChatWindow.jsx          # Message display area
│     ├─ MessageBubble.jsx       # Individual message styling
│     ├─ InputBar.jsx            # Message input + send
│     ├─ FeatureSidebar.jsx      # Health metrics tracker
│     ├─ MetricItem.jsx          # Individual metric row
│     ├─ DietScoreModal.jsx      # Diet calculator modal
│     ├─ DiagnosisCard.jsx       # Final diagnosis display
│     ├─ LoadingIndicator.jsx    # Typing wave animation
│     └─ (others)
├─ public/
│  └─ index.html
├─ package.json
└─ vite.config.js
```

### Key Frontend Components

#### 1. **App.jsx** - Main Layout & State Orchestration
```
Purpose: Top-level component managing layout and state

State:
├─ useChat() hook:
│  ├─ messages: Array of {role, content}
│  ├─ features: {Age, Glucose, HbA1c, ...}
│  ├─ loading: boolean (waiting for response)
│  ├─ prediction: null or {class, name, confidence, explanation}
│  └─ sendMessage, resetChat functions
├─ showConfirm: boolean (reset confirmation modal)
├─ sidebarOpen: boolean (mobile sidebar open/close)
└─ inputValue: string (current message in input)

Layout (flex):
├─ Mobile Header (md:hidden)
│  ├─ "MediHelp" title
│  └─ Hamburger menu (open/close sidebar)
├─ Sidebar (fixed md:static)
│  ├─ Hidden on mobile by default
│  ├─ Slides in from left when opened (transform animation)
│  ├─ Overlay backdrop closes on click
│  └─ FeatureSidebar component
└─ Main Chat Area (flex-1)
   ├─ ChatWindow (messages + loading indicator)
   └─ InputBar (message input)

Responsive Design:
- Mobile (default): flex-col, sidebar hidden, hamburger visible
- Desktop (md+): flex-row, sidebar static, full layout visible
```

#### 2. **useChat.js** - Chat Hook
```
Purpose: Manage chat state and API communication

State:
├─ messages: []
├─ features: {}
├─ loading: false
├─ sessionId: null
├─ prediction: null

Functions:

sendMessage(text):
├─ Add user message immediately: {role: 'user', content: text}
├─ Set loading = true
├─ POST /api/chat with message
├─ Response:
│  ├─ Update sessionId
│  ├─ Update features
│  ├─ Add assistant message to messages
│  └─ If is_complete, store prediction
├─ Set loading = false
└─ Return response

resetChat():
├─ If no sessionId:
│  └─ Clear UI locally (no API call)
├─ Else:
│  ├─ POST /api/reset
│  └─ Clear all state
├─ Reset messages, features, sessionId, prediction
└─ Handle errors gracefully
```

#### 3. **ChatWindow.jsx** - Message Display
```
Purpose: Display chat messages and results

Props:
├─ messages: Array of {role, content}
├─ prediction: null or diagnosis object
└─ loading: boolean

Renders:
├─ Empty state (when no messages)
│  └─ Title + description
├─ Message list:
│  └─ For each message: MessageBubble
├─ Loading indicator (when loading=true):
│  └─ LoadingIndicator with 3-dot wave
├─ Diagnosis card (when prediction available):
│  └─ DiagnosisCard with full diagnosis
└─ Auto-scroll to bottom on new messages
```

#### 4. **InputBar.jsx** - User Input
```
Purpose: Collect and send user messages

Props:
├─ onSend: callback (message) → sends to API
├─ onReset: callback → resets session
├─ disabled: boolean (disable during loading)
├─ inputValue: string (from parent)
└─ onInputChange: callback (text) → updates parent

Features:
├─ Textarea with auto-resize (max 120px)
├─ Send on Enter (Shift+Enter = newline)
├─ Disabled state during loading
├─ "New" button to reset session
├─ Send button with icon
└─ Keyboard shortcuts tip

Behavior:
├─ User types → onInputChange callback updates parent
├─ User presses Enter → onSend callback
├─ Message sent → clear input, auto-focus
└─ During loading → buttons/input disabled
```

#### 5. **FeatureSidebar.jsx** - Health Metrics Tracker
```
Purpose: Display collected health metrics

State:
├─ collectedCount: sum of non-null features
├─ progressPercent: (collectedCount / 16) * 100

Renders:
├─ Header with progress bar
│  ├─ "X/16 collected"
│  ├─ "Y% complete"
│  └─ Animated progress bar (purple)
├─ Feature list (16 items):
│  └─ For each feature: MetricItem
│     ├─ Display label
│     ├─ Status icon (checked or empty circle)
│     ├─ Extracted value (green badge)
│     └─ Info icon → popover tooltip
└─ Help tip at bottom

Conditional Rendering:
├─ Diet Score feature gets special onFillInput callback
└─ Other features: onFillInput = undefined
```

#### 6. **MetricItem.jsx** - Individual Metric Row
```
Purpose: Display single health metric with interactive features

Props:
├─ feature: string (e.g., "Glucose")
├─ displayLabel: string (e.g., "Blood Glucose")
├─ definition: string (detailed explanation)
├─ isCollected: boolean (value extracted or not)
├─ value: number or null (extracted value)
└─ onFillInput: function or undefined (for Diet Score only)

State:
├─ showHint: boolean (popover visible)
├─ showCalculator: boolean (diet modal open)
├─ popoverPosition: {top, left} (calculated)

Features:

1. Status Icon & Label:
   ├─ CheckCircle (green) if collected
   └─ Circle (gray) if pending

2. Value Badge:
   ├─ Shows only if collected
   ├─ Formats value (binary → Yes/No, HbA1c → 1 decimal)
   └─ Green background

3. Hover Interactions:
   ├─ Row background highlights
   ├─ Info icon clickable
   └─ For Diet Score: "Don't know? Calculate" button appears

4. Popover:
   ├─ Positioned to right of info icon
   ├─ Shows definition text
   ├─ Semi-transparent green backdrop
   ├─ Close button (X)
   └─ Arrow pointer

5. Diet Score Modal:
   ├─ Shows only for Diet Score metric
   ├─ Triggered by "Don't know? Calculate" button
   ├─ Portal rendered (outside sidebar)
   └─ Calls onFillInput on submit

Responsive:
├─ Mobile: Text wraps, smaller badges
├─ Desktop: Full width layout
```

#### 7. **DietScoreModal.jsx** - Diet Calculator
```
Purpose: Interactive 3-question diet assessment

State:
├─ q1: null | 1 | 2 | 4 (Fruits & Veggies)
├─ q2: null | 0 | 1.5 | 3 (Processed Foods)
└─ q3: null | 1 | 2 | 3 (Grains & Proteins)

Questions:

Q1 - Fruits & Vegetables (max 4 points):
├─ 0-1 servings/day → 1 pt
├─ 2-3 servings/day → 2 pts
└─ 4+ servings/day → 4 pts

Q2 - Processed Foods (max 3 points):
├─ 4+ times/week → 0 pts
├─ 1-3 times/week → 1.5 pts
└─ Rarely/Never → 3 pts

Q3 - Grains & Proteins (max 3 points):
├─ Mostly refined/red meat → 1 pt
├─ Mix of both → 2 pts
└─ Mostly whole grains/lean → 3 pts

Scoring:
├─ Total = q1 + q2 + q3
├─ Range: 0-10 points
└─ Can include decimals (e.g., 7.5)

UI:
├─ Modal overlay (fixed, centered)
├─ Radio buttons (grouped by question)
├─ Score preview (when all answered)
├─ "Calculate & Add" button (enabled when complete)
├─ "Cancel" button
└─ Rendered via createPortal to document.body

Behavior:
├─ User selects all 3 answers
├─ "Calculate & Add" becomes enabled
├─ Click → calculate score
├─ Call onFillInput("Diet Score is 7.5")
├─ Close modal
├─ Input field auto-filled: "Diet Score is 7.5"
└─ User sends message
```

#### 8. **DiagnosisCard.jsx** - Final Diagnosis
```
Purpose: Display prediction results professionally

Props:
├─ diagnosis: {
│  ├─ prediction_name: "Diabetes"
│  ├─ confidence: 0.87
│  ├─ risk_level: "High"
│  ├─ explanation: "..."
│  └─ features: {all 16 collected}
│ }

Renders:

1. Header:
   └─ "Health Summary" title

2. Diagnosis Section:
   ├─ "Diagnosis" label
   └─ Large bold prediction name

3. Explanation:
   ├─ Background card
   ├─ Detailed explanation text
   └─ Addresses specific findings

4. Patient Data Report:
   ├─ Grid layout (1 col mobile → 2 col tablet → 4 col desktop)
   ├─ For each feature:
   │  ├─ Label (uppercase, smaller)
   │  └─ Value (bold, larger)
   └─ Shows all 16 collected metrics

Styling:
├─ Green border + background
├─ Professional card design
└─ Responsive grid
```

#### 9. **LoadingIndicator.jsx** - Typing Animation
```
Purpose: Show polished loading state with dynamic text

State:
├─ show: boolean (1-second delay before showing)
└─ phraseIndex: int (current phrase in rotation)

Features:

1. 1-Second Delay:
   ├─ useEffect with setTimeout
   ├─ Sets show = true after 1000ms
   └─ Returns null before delay (clean for quick responses)

2. Dynamic Text Cycling:
   ├─ Phrases array:
   │  ├─ "Recognizing metrics"
   │  ├─ "Running health models"
   │  ├─ "Cross-referencing data"
   │  ├─ "Synthesizing response"
   │  ├─ "Almost there"
   │  ├─ "Analyzing vital signs"
   │  ├─ "Evaluating risk factors"
   │  ├─ "Processing test results"
   │  ├─ "Correlating symptoms"
   │  ├─ "Assessing health indicators"
   │  ├─ "Computing probability scores"
   │  ├─ "Comparing diagnostic patterns"
   │  ├─ "Validating predictions"
   │  ├─ "Generating personalized insights"
   │  └─ "Finalizing diagnosis"
   ├─ useEffect with setInterval
   ├─ Updates phraseIndex every 1500ms
   ├─ Cycles through all phrases (modulo arithmetic)
   └─ Proper cleanup on unmount

3. 3-Dot Wave Animation:
   ├─ Three small dots (5px circles)
   ├─ Custom CSS keyframes:
   │  └─ @keyframes wave:
   │     ├─ 0%: translateY(0px)
   │     ├─ 50%: translateY(-8px)
   │     └─ 100%: translateY(0px)
   ├─ Animation duration: 1.4s
   ├─ Infinite loop with ease-in-out
   └─ Staggered delays (0s, 0.2s, 0.4s)

4. Styling:
   ├─ Flex container (items-center gap-2)
   ├─ Small italic text (text-sm italic)
   ├─ Subtle gray color (text-gray-400)
   ├─ Padding (py-4)
   └─ Inline CSS for animation

Cleanup:
├─ setTimeout returns cleanup function (clears timer)
├─ setInterval returns cleanup function (clears timer)
└─ All timers cleared on unmount or show change
```

#### 10. **MessageBubble.jsx** - Message Styling
```
Purpose: Style individual messages

Props:
├─ role: "user" or "assistant"
└─ content: string (message text)

Renders:

User Message:
├─ Right-aligned
├─ White background + border
├─ Gray text
├─ Max-width prose
└─ Whitespace preserved + word wrap

Assistant Message:
├─ Left-aligned
├─ Pink background (#FFD1DC)
├─ Gray text
├─ Max-width prose
└─ Whitespace preserved + word wrap
```

---

## 🔄 Data Flow (End-to-End)

### Complete User Journey

```
SCENARIO: User gets health assessment

STEP 1: PAGE LOAD
├─ React App mounts
├─ useChat hook initializes (empty state)
├─ ChatWindow renders empty state
├─ FeatureSidebar shows 0/16 progress
└─ InputBar ready for input

STEP 2: USER GREETS
User types: "Hello, I want to get checked"

STEP 3: SEND MESSAGE
├─ User clicks Send or presses Enter
├─ onSend callback triggered
├─ InputBar value → sendMessage(text)
├─ Input cleared immediately

STEP 4: FRONTEND → BACKEND
JavaScript:
  axios.POST('/api/chat', {
    session_id: null,
    message: "Hello, I want to get checked",
    history: []
  })

STEP 5: BACKEND PROCESSING
├─ api.py chat_endpoint() called
├─ session_id generated: "sess_abc123"
├─ extract_features_from_text() called
├─ Regex extracts: {all null} (no metrics)
├─ Check if greeting: YES
├─ Return greeting response (no metrics extracted)

Python response:
  ChatResponse(
    session_id: "sess_abc123",
    message: "Hello! This is your medical assistant...",
    features: {},
    collected_count: 0,
    is_complete: false,
    prediction: null
  )

STEP 6: FRONTEND UPDATE
├─ Response received
├─ sessionId set to "sess_abc123"
├─ features updated (empty)
├─ setLoading(false) → loading state ends
├─ Message added to messages array
├─ LoadingIndicator disappears (loading=false)
├─ ChatWindow re-renders with new message
├─ InputBar auto-focuses
├─ Progress bar still 0/16

STEP 7: USER PROVIDES METRICS
User types: "I'm 45 years old and my glucose is 180"

STEP 8: SEND MESSAGE (2nd)
├─ onSend("I'm 45 years old and my glucose is 180")
├─ Add user message to messages
├─ Set loading = true
└─ LoadingIndicator appears with typing animation

STEP 9: BACKEND PROCESSING
├─ chat_endpoint() called with session_id
├─ Load persisted state (empty)
├─ extract_features_from_text() called
├─ REGEX extraction:
│  ├─ Pattern "(\d+) years old" → Age = 45
│  ├─ Pattern "glucose (\d+)" → Glucose = 180
│  └─ Return {Age: 45, Glucose: 180, ...rest: null}
├─ Not a greeting (has medical data)
├─ Update state: {Age: 45, Glucose: 180, ...rest: null}
├─ Count: 2/16 features
├─ Missing: 14 features
├─ Generate next question: "HbA1c: What is your HbA1c level?"
├─ Store pending feature: state["__pending_feature__"] = "HbA1c"
├─ Save state to disk
└─ Return:
   {
     session_id: "sess_abc123",
     message: "✓ Got your Age: 45 and Glucose: 180\n\n**HbA1c:** What...",
     features: {Age: 45, Glucose: 180, ...rest: null},
     collected_count: 2,
     is_complete: false,
     prediction: null
   }

STEP 10: FRONTEND UPDATE
├─ loading = false (LoadingIndicator disappears)
├─ Features updated: {Age: 45, Glucose: 180, ...rest: null}
├─ Progress bar updates: 2/16 (12.5%)
├─ New assistant message displayed
├─ FeatureSidebar MetricItems update:
│  ├─ Age: CheckCircle (green) + badge "45"
│  ├─ Glucose: CheckCircle (green) + badge "180"
│  └─ Others: Circle (gray)

STEPS 11-15: REPEAT CONVERSATION
User continues answering questions one by one:

Message 3: "HbA1c is 6.8" → Backend extracts HbA1c=6.8 → 3/16
Message 4: "My BMI is 28" → Backend extracts BMI=28 → 4/16
Message 5: "Cholesterol 210" → Backend extracts Cholesterol=210 → 5/16
...
Message 19: "Oxygen saturation is 97" → Backend extracts O2=97 → 16/16 ✓

STEP 16: ALL FEATURES COLLECTED
Backend recognizes: collected_count = 16

├─ Prepare feature vector in model training order
├─ Load scikit-learn gradient boosting model
├─ model.predict([feature_vector]) → class index (e.g., 3)
├─ model.predict_proba([feature_vector]) → confidence (e.g., [0.05, 0.03, 0.02, 0.87, 0.02, 0.01, 0, 0])
├─ CLASS_NAMES[3] = "Diabetes"
├─ confidence = 0.87 (87%)
├─ get_explanation(Diabetes, features) → "Based on your..."
└─ Return:
   {
     session_id: "sess_abc123",
     message: "✅ Assessment Complete! Your diagnosis is ready below.",
     features: {all 16 collected},
     collected_count: 16,
     is_complete: true,
     prediction: {
       prediction_class: 3,
       prediction_name: "Diabetes",
       confidence: 0.87,
       risk_level: "High",
       explanation: "...",
       features: {all 16}
     }
   }

STEP 17: FINAL DISPLAY
Frontend:
├─ loading = false
├─ is_complete = true
├─ prediction received
├─ ChatWindow renders DiagnosisCard
├─ DiagnosisCard displays:
│  ├─ "Health Summary" header
│  ├─ Large "Diabetes" title
│  ├─ Explanation paragraph
│  └─ Patient Data Report grid (all 16 metrics)
├─ FeatureSidebar shows 16/16 (100%)
└─ Progress bar full (green)

STEP 18: USER RESETS (Optional)
User clicks "New" button:
├─ showConfirm modal appears
├─ User clicks "Start New"
├─ resetChat() called
├─ POST /api/reset with session_id
├─ Backend clears session file
├─ Frontend clears all state:
│  ├─ messages = []
│  ├─ features = {}
│  ├─ sessionId = null
│  ├─ prediction = null
│  └─ inputValue = ""
└─ Return to empty chat state
```

---

## 🔑 Key Features Explained

### 1. **Smart Feature Extraction**
- **Two-Stage**: Regex (fast) → LLM (comprehensive)
- **Contextual**: Pending feature awareness for standalone numbers
- **Fallback**: Works even if LLM unavailable
- **Range Validation**: Ensures extracted values are realistic

### 2. **Session Persistence**
- **Session ID**: Generated from first message hash
- **State Storage**: Persisted to `.sessions/` directory
- **Feature Tracking**: Knows which features collected, which pending
- **Reset**: Clears session completely

### 3. **Dynamic Question Generation**
- **Prioritization**: Asks most important missing features first
- **Context-Aware**: Remembers what was just asked
- **Hints**: Provides definition/help for each metric
- **Acknowledgment**: Confirms what was extracted

### 4. **Prediction Model**
- **Algorithm**: Scikit-learn Gradient Boosting Classifier
- **Features**: All 16 health metrics as input
- **Output**: Class prediction + confidence probability
- **Explanations**: Human-readable interpretation

### 5. **Interactive UI**
- **Sidebar**: Tracks collected metrics with progress bar
- **Hover Tooltips**: Definitions for each metric
- **Diet Calculator**: 3-question form for diet score
- **Loading Animation**: Polished 3-dot wave with cycling text
- **Responsive Design**: Mobile-first layout

### 6. **Mobile Responsiveness**
- **Mobile**: Full-width chat, hamburger sidebar toggle
- **Desktop**: Split layout (sidebar + chat)
- **Tablet**: Intermediate responsive sizing
- **Touch-friendly**: Large buttons, proper spacing

---

## 🚀 Deployment

### Local Development
```bash
# Backend
cd /path/to/project
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python server.py
→ Runs on http://localhost:8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
→ Runs on http://localhost:5173
→ Proxied to backend via vite.config.js
```

### Hugging Face Spaces
```
Environment: SPACE_ID set by HF
Port: 7860 (auto-detected)
Reload: Disabled (production mode)
Frontend: Built to dist/, mounted at /
```

### File Structure
```
project-root/
├─ server.py                 # Entry point
├─ app/
│  ├─ api.py                # REST endpoints
│  ├─ config.py             # Configuration
│  ├─ main.py               # Legacy Gradio app
│  ├─ memory.py             # State management
│  ├─ services/
│  │  ├─ llm_extractor.py   # Feature extraction
│  │  ├─ feature_builder.py # Feature validation
│  │  ├─ predictor.py       # Model inference
│  │  └─ session_manager.py # Session management
│  └─ utils/
│     └─ helpers.py         # Utility functions
├─ models/
│  └─ GradientBoosting_model.pkl
├─ llm/
│  └─ meta-llama-3.1-8b-instruct.Q4_K_M.gguf
├─ .sessions/               # Session state storage (created at runtime)
├─ frontend/
│  ├─ src/
│  │  ├─ App.jsx
│  │  ├─ hooks/
│  │  ├─ components/
│  │  └─ index.css
│  ├─ dist/                 # Built frontend
│  ├─ package.json
│  ├─ vite.config.js
│  └─ tailwind.config.js
├─ requirements.txt
└─ README.md
```

---

## 📊 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 | UI framework |
| | Tailwind CSS | Styling |
| | Axios | HTTP requests |
| | Lucide Icons | UI icons |
| **Backend** | FastAPI | REST API framework |
| | Uvicorn | ASGI server |
| | Pydantic | Data validation |
| **LLM** | Llama 3.1 8B GGUF | Feature extraction |
| | llama-cpp-python | Model inference |
| **ML** | scikit-learn | Prediction model |
| | Gradient Boosting | Classification algorithm |
| **State** | JSON files | Session persistence |

---

## 🎯 Summary

This medical predictor chatbot represents a complete full-stack application that:

1. **Engages** users naturally about their health
2. **Extracts** medical metrics using smart regex + local LLM
3. **Validates** data ranges and manages state
4. **Predicts** one of 8 health conditions using ML
5. **Displays** results with confidence and explanation
6. **Works** offline (no cloud dependencies after model download)
7. **Scales** from mobile to desktop with responsive design

The system is production-ready, deployable to Hugging Face Spaces, and can handle multiple concurrent users with independent sessions.

---

**End of Document**

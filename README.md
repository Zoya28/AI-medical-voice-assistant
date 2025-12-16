# Voice Assistant

### Real-Time, Data-Driven Health Insights via Voice

A production-grade **voice-enabled health assistant** that understands natural language queries, securely fetches **real user health data from InfluxDB**, applies business logic, and generates **human-like responses** using an LLM.

---

## 📌 Problem Statement

Health applications collect large volumes of time-series data (steps, heart rate, SpO₂, blood pressure, calories, etc.), but:

- Users struggle to understand raw metrics
- Dashboards are not conversational
- Existing assistants give generic answers
- Most systems do not reason over *real* user data

Users want **simple voice-based answers** like:
- “Was my heart rate higher this week than last week?”
- “Is my oxygen level normal this morning?”
- “How active was I yesterday evening?”

---

## ✅ Solution

This Voice Health Assistant provides:

- 🎙️ Voice-first interaction
- 🧠 Intent & entity understanding
- ⏱ Human time phrase resolution
- 📊 Secure real-time data fetching from InfluxDB
- 📈 Business logic and trend analysis
- 🤖 Natural language responses using Phi-3 LLM

All answers are generated **only after processing the user’s actual data**.

---

## 🧠 What the System Does

End-to-end workflow:

1. Records user voice from mobile app
2. Converts audio → text (Faster-Whisper)
3. Understands intent, metric, and time range
4. Converts human time phrases to timestamps
5. Fetches user data from InfluxDB
6. Applies business logic and comparisons
7. Generates natural language response via LLM
8. Sends final answer back to the user

---

## 🏗 High-Level Architecture

A **5-layer modular architecture**:

| Layer | Responsibility |
|------|---------------|
| Microphone | Captures voice & receives response |
| Speech Layer | Audio → Text |
| NLU Layer | Intent, metric, time extraction |
| Data Layer | Secure InfluxDB access |
| Logic + LLM Layer | Insights + natural language |

---

## 📦 Complete Project Structure

```
realtime_voice_assistant/
├── src/
│   ├── __init__.py
│   ├── audio_capture.py              # Real-time audio with VAD
│   ├── speech_to_text.py             # Faster Whisper STT
│   ├── Data_manager.py          
│   ├── entity_extractor.py           # LLM entity extraction 
│   ├── influx_data_manager.py               # Database management
│   ├── llm_processor.py              # LLM response generation
│   ├── graph_conversation_manager.py       # Context & slot filling 
│
├── datasets/
│   └── intent_mapped.csv      
│
├── models/                           # Model cache (auto-created) 
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation
└── app.py         # Main orchestrator
└── PROJECT_SUMMARY.md                # This file
```


## 🎯 Key Features Implemented

### 1. Real-Time Audio Capture ✅
- Voice Activity Detection (VAD)
- Push-to-talk and continuous listening modes
- Audio buffering and segmentation
- Silence detection

### 2. Speech-to-Text ✅
- Faster Whisper integration (medium.en model)
- Near real-time transcription (< 2 seconds)
- Confidence scoring
- Background noise filtering

### 3. Intent Classification ✅
- Rule-based pattern matching
- 9 supported intent categories
- Confidence scoring
- Extensible pattern system

### 4. Entity Extraction ⭐
- **LLM-Powered** using Phi3
- Handles tricky questions ("how long did I walk" → extracts steps, duration, distance)
- Contextual understanding
- Multi-metric extraction
- Date parsing (yesterday, last week, specific dates)
- Value extraction with units
- Comparison operators

### 5. Conversation Context Management 
- **Slot Memory**: Remembers entities across turns
- **Automatic Slot Filling**: Uses context to fill missing information
- **Clarification Questions**: Asks for missing required slots
- **Multi-Turn Tracking**: Maintains conversation history
- **Context Resolution**: Resolves ambiguous references
- **Context Reset**: Clear history for new topics

### 6. Data Management ✅
-  CSV dataset with health/fitness data
- Steps, calories, sleep, heart rate tracking
- Date range queries
- Aggregation functions (sum, avg, min, max)

### 7. LLM Response Generation ✅
- Phi3 integration
- Context-aware responses
- Conversation history integration
- Natural language generation
- Encouraging and helpful tone

## 8. Beautiful Streamlit UI ##
- Two Input Modes: Voice + Text
- Side-by-Side Display:
    - AI Response (gradient box)
    - Actual Data (data box)
- Real-time Status: Component health checks
- Debug Mode: Full transparency
- Conversation History: Persistent across session
- Quick Examples: One-click test queries


# Technical Stack #
## Core Technologies ##
yamlAudio Processing:
  - pyaudio: Audio capture
  - webrtcvad: Voice Activity Detection
  - numpy: Audio processing

Speech-to-Text:
  - faster-whisper: Transcription
  - torch: Deep learning backend

NLP & ML:
  - spacy: NER model
  - sentence-transformers: Embeddings
  - scikit-learn: Intent classifier
  - transformers: LLM interface

UI Framework:
  - streamlit: Web interface
  - streamlit-webrtc: Real-time audio

Utilities:
  - python-dotenv: Environment variables
  - joblib: Model serialization

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python3.12.5 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash

pip install -r requirements.txt
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Running the Project

```bash
streamlit run app.py
```


## Notes
- Ensure you have Python 3.12.5 installed.
- Update `requirements.txt` as needed for your dependencies.
- By default this will run on gpu if you have it.
- If you dont have a gpu then simply replace cuda with cpu in the code.


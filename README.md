# Voice Assistant

 Voice Health Assistant
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


# Voice Assistant

This project is a voice assistant powered by AI models.

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


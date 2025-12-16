"""
Streamlit Voice Assistant UI with InfluxDB Integration
"""

import streamlit as st
import os
import logging
import joblib
import time
from typing import Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
from src.audio_capture import AudioCapture
from src.speech_to_text import SpeechToText
from src.entity_extractor import EntityExtractor
from src.influxdb_data_manager import InfluxDataManager
from src.llm_processor import LLMProcessor
from src.graph_conversation_manager import LangGraphConversationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VoiceAssistant")

# Paths
BASE_DIR = os.path.dirname(__file__)
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "multi_intent_classifier.pkl")

# InfluxDB Configuration
INFLUX_USER_ID = os.getenv("INFLUX_USER_ID", "OF36GM")
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

# ==========================================================
# Component Initialization (Cached for Performance)
# ==========================================================

@st.cache_resource
def get_audio_capture():
    """Initialize audio capture (cached)"""
    try:
        logger.info("Initializing audio capture...")
        return AudioCapture(
            sample_rate=16000,
            channels=1,
            frame_ms=30,
            energy_threshold=300,
            dynamic_energy=True,
            max_silence_sec=0.8,
            min_segment_sec=0.25
        )
    except Exception as e:
        logger.error(f"Failed to initialize audio capture: {e}")
        return None

@st.cache_resource
def get_speech_to_text():
    """Initialize speech-to-text model (cached)"""
    try:
        logger.info("Loading speech-to-text model...")
        device = "cuda" if st.session_state.get('use_gpu', False) else "cpu"
        return SpeechToText(
            model_name="medium.en",
            device=device,
            sample_rate=16000
        )
    except Exception as e:
        logger.error(f"Failed to load STT model: {e}")
        return None

@st.cache_resource
def get_entity_extractor():
    """Initialize entity extractor (cached)"""
    try:
        logger.info("Initializing entity extractor...")
        use_llm = st.session_state.get('use_llm_for_entities', False)
        return EntityExtractor(use_llm=use_llm)
    except Exception as e:
        logger.error(f"Failed to initialize entity extractor: {e}")
        return None

@st.cache_resource
def get_data_manager():
    """Initialize InfluxDB data manager (cached)"""
    try:
        logger.info("Initializing InfluxDB data manager...")
        
        if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET]):
            logger.error("InfluxDB configuration incomplete. Check environment variables.")
            return None
        
        return InfluxDataManager(
            user_id=INFLUX_USER_ID,
            url=INFLUX_URL,
            token=INFLUX_TOKEN,
            org=INFLUX_ORG,
            bucket=INFLUX_BUCKET
        )
    except Exception as e:
        logger.error(f"Failed to initialize InfluxDB data manager: {e}")
        return None

@st.cache_resource
def get_llm_processor():
    """Initialize LLM processor (cached)"""
    try:
        logger.info("Loading LLM processor...")
        return LLMProcessor()
    except Exception as e:
        logger.error(f"Failed to load LLM processor: {e}")
        return None

@st.cache_resource
def get_intent_classifier():
    """Load intent classifier (cached)"""
    try:
        logger.info(f"Loading intent classifier from {CLASSIFIER_PATH}")
        if not os.path.exists(CLASSIFIER_PATH):
            logger.error(f"Classifier not found: {CLASSIFIER_PATH}")
            return None
        return joblib.load(CLASSIFIER_PATH)
    except Exception as e:
        logger.error(f"Failed to load intent classifier: {e}")
        return None

@st.cache_resource
def get_embedder():
    """Load sentence embedder (cached)"""
    try:
        logger.info("Loading sentence embedder...")
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        return None

@st.cache_resource
def get_conversation_manager():
    """Initialize LangGraph conversation manager (cached)"""
    try:
        logger.info("Initializing LangGraph conversation manager...")
        return LangGraphConversationManager(
            max_history=10,
            inference_window=4,
            session_timeout=600
        )
    except Exception as e:
        logger.error(f"Failed to initialize conversation manager: {e}")
        return None

def get_session_id():
    """Get or create session ID"""
    return st.session_state.setdefault('session_id', f"session_{int(time.time())}")

def sanitize_for_json(obj):
    """Recursively convert non-serializable objects (e.g., numpy types) into JSON-safe types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# ==========================================================
# Processing Functions
# ==========================================================

def classify_intent(text: str) -> Tuple[str, float]:
    """Classify intent using the loaded classifier"""
    try:
        classifier = get_intent_classifier()
        embedder = get_embedder()
        
        if classifier is None or embedder is None:
            logger.warning("Classifier or embedder not loaded")
            return "unknown", 0.0
        
        # Generate embedding
        embedding = embedder.encode([text], convert_to_numpy=True, show_progress_bar=False)
        
        # Predict intent
        intent = classifier.predict(embedding)[0]
        
        # Get confidence
        confidence = 0.85
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(embedding)[0]
            confidence = float(max(proba))
        
        logger.info(f"Intent classified: {intent} (confidence: {confidence:.2f})")
        return intent, confidence
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "unknown", 0.0

def extract_date_info(entities: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict], Optional[Dict], str]:
    """Extract and format date information from entities"""
    date = None
    date_range = None
    time_of_day = None
    date_info = "current"
    
    if entities.get('dates'):
        date_entity = entities['dates'][0] if isinstance(entities['dates'], list) else entities['dates']
        
        if isinstance(date_entity, dict):
            # Handle realtime queries
            if date_entity.get('type') == 'realtime':
                subtype = date_entity.get('subtype', 'current')
                if subtype == 'current':
                    date_info = "current"
                elif subtype == 'duration':
                    duration = date_entity.get('duration', {})
                    date_info = f"last {duration.get('amount', 0)} {duration.get('unit', 'minutes')}"
            # Handle date ranges
            elif date_entity.get('type') == 'date_range':
                date_range = {
                    'start': date_entity['start'],
                    'end': date_entity['end']
                }
                date_info = f"{date_entity['start']} to {date_entity['end']}"
            # Handle single dates
            else:
                date = date_entity.get('value')
                date_info = str(date)
                if date_entity.get('raw_text'):
                    date_info = date_entity['raw_text']
            
            if 'time_of_day' in date_entity:
                time_of_day = date_entity.get('time_range')
        else:
            date = date_entity
            date_info = str(date)
    
    # Add inference indicators
    if entities.get('_inferred_date'):
        date_info += " (from context)"
    
    return date, date_range, time_of_day, date_info

def extract_metrics(entities: Dict[str, Any]) -> list:
    """Extract metrics from entities"""
    metrics = entities.get('metrics', [])
    if metrics and isinstance(metrics[0], dict):
        metrics = [m.get('type', m) for m in metrics]
    return metrics

def process_query(text: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    OPTIMIZED: Process query with InfluxDB data retrieval
    """
    t0 = time.perf_counter()

    def _attach_timing(res: Dict[str, Any]) -> Dict[str, Any]:
        res['response_time'] = round(time.perf_counter() - t0, 3)
        return res

    try:
        logger.info(f"Processing query: '{text}'")
        
        # Get components
        entity_extractor = get_entity_extractor()
        data_manager = get_data_manager()
        llm_processor = get_llm_processor() if use_llm else None
        conv_manager = get_conversation_manager()
        session_id = get_session_id()
        print(data_manager)
        
        if not entity_extractor or not data_manager or not conv_manager:
            return _attach_timing({
                'text': text,
                'error': 'Required components not loaded',
                'response': 'System components are not properly initialized.',
                'success': False
            })
        
        # STEP 1: Intent Classification
        intent, intent_confidence = classify_intent(text)
        logger.info(f"✅ Step 1 - Intent: {intent} ({intent_confidence:.2f})")
        
        # STEP 2: Entity Extraction (initial)
        entities = entity_extractor.extract_entities(text, intent)
        logger.info(f"✅ Step 2 - Entities (initial): {entities}")

        # STEP 3: Fill missing entities from conversation context
        state = conv_manager.get_state(session_id)
        
        if state:
            query_lower = text.lower()
            
            # Improved follow-up detection
            is_followup = any(phrase in query_lower for phrase in [
                'what about', 'how about', 'and', 'also', 'too',
                'as well', 'what was', 'how was', 'tell me about',
                'show me', 'what were', 'how were', 'current',
                'now', 'right now'
            ])
            
            # Also consider short queries as potential follow-ups
            is_short_query = len(text.split()) <= 5
            
            if is_followup or is_short_query:
                # Define query intent groups
                query_intents = ['query_steps', 'query_calories', 'query_sleep', 
                               'query_heart_rate', 'query_distance', 'query_blood_pressure',
                               'query_spo2', 'query_activity']
                
                # Check if current and last intents are both query-type
                is_query_intent = intent in query_intents
                was_query_intent = state.get('last_intent') in query_intents
                
                # INHERIT DATE if appropriate
                current_date = entities.get('dates', [{}])[0] if entities.get('dates') else {}
                has_explicit_date = current_date.get('raw_text') not in ['today', None]
                
                if not has_explicit_date and state.get('last_date') and is_query_intent and was_query_intent:
                    entities['dates'] = [state['last_date']]
                    entities['_inferred_date'] = True
                    logger.info(f"🔗 Inherited date from context: {state['last_date']}")
                
                # INHERIT METRIC if appropriate
                if not entities.get('metrics') and state.get('last_metric'):
                    if is_query_intent and was_query_intent:
                        entities['metrics'] = [state['last_metric']]
                        entities['_inferred_metric'] = True
                        logger.info(f"🔗 Inherited metric from context: {state['last_metric']}")

        logger.info(f"✅ Step 3 - Entities (after context): {entities}")
        
        # STEP 4: Process through LangGraph
        logger.info("🔄 Step 4 - Processing through LangGraph...")
        graph_state = conv_manager.process_turn(
            user_query=text,
            intent=intent,
            intent_confidence=intent_confidence,
            entities=entities,
            session_id=session_id
        )
        
        logger.info(f"✅ Step 4 - Graph complete={graph_state.get('is_complete')}, "
                   f"clarification={graph_state.get('awaiting_clarification')}")
        
        # STEP 5: Check if clarification is needed
        if graph_state.get('awaiting_clarification'):
            response = graph_state.get('clarification_question', 'Could you provide more details?')
            logger.info(f"⚠️ Step 5 - Clarification needed: {response}")
            return _attach_timing({
                'text': text,
                'intent': intent,
                'intent_confidence': intent_confidence,
                'entities': graph_state.get('entities', entities),
                'response': response,
                'needs_clarification': True,
                'missing_slots': graph_state.get('missing_slots', []),
                'conversation_state': 'awaiting_clarification',
                'graph_state': graph_state,
                'success': True
            })
        
        # STEP 6: Extract final processed entities
        processed_entities = graph_state.get('entities', entities)
        logger.info(f"✅ Step 6 - Final entities: {processed_entities}")
        
        # Extract date/time information
        date, date_range, time_of_day, date_info = extract_date_info(processed_entities)
        
        # Get metrics
        metrics = extract_metrics(processed_entities)
        
        logger.info(f"📊 Query params - Metrics: {metrics}, Date: {date_info}")
        
        # STEP 7: Query data from InfluxDB
        data = None
        formatted_data = ""
        
        if metrics and metrics[0]:
            try:
                # Use InfluxDB's query_from_entities method
                logger.info(f"Querying InfluxDB for entities: {processed_entities}")
                data = data_manager.query_from_entities(processed_entities)
                
                if data and 'error' not in data:
                    formatted_data = data_manager.format_for_llm(data, text)
                    logger.info(f"✅ Step 7 - Data retrieved: {data.get('query_type', 'unknown')}")
                    conv_manager.update_with_data(session_id, sanitize_for_json(data))
                else:
                    error_msg = data.get('error', 'Unknown error') if data else 'No data returned'
                    formatted_data = f"No data available: {error_msg}"
                    logger.warning(f"⚠️ Step 7 - {error_msg}")
                    
            except Exception as e:
                logger.error(f"InfluxDB query failed: {e}", exc_info=True)
                formatted_data = f"Error retrieving data: {str(e)}"
        else:
            logger.warning("⚠️ Step 7 - No metrics to query")
        
        # STEP 8: Generate response
        has_data = bool(data and 'error' not in data and data.get('summary'))
        response = ""
        
        if use_llm and llm_processor:
            try:
                logger.info("🤖 Step 8 - Generating LLM response...")
                
                # Get conversation history
                history = conv_manager.get_conversation_history(session_id, num_turns=3)
                history_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in history[-6:]
                ]) if history else ""
                
                # Prepare data context
                data_context = formatted_data if has_data else f"No data available for {date_info}."
                
                # Add inference context
                context_notes = []
                if processed_entities.get('_inferred_date'):
                    context_notes.append("Date inferred from previous conversation")
                if processed_entities.get('_inferred_metric'):
                    context_notes.append("Metric inferred from previous conversation")
                if context_notes:
                    data_context += f"\n\nContext: {'; '.join(context_notes)}"
                
                # Generate
                response = llm_processor.generate_response(
                    query=text,
                    data_context=data_context,
                    intent=intent,
                    conversation_history=history_text
                )
                
                conv_manager.update_with_response(session_id, sanitize_for_json(response))
                logger.info(f"✅ Step 8 - Response: {response[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ LLM failed: {e}", exc_info=True)
                response = formatted_data if has_data else f"I don't have data for {date_info}."
        else:
            response = formatted_data if has_data else f"No data found for {date_info}."
            logger.info(f"✅ Step 8 - Using formatted data (LLM disabled)")
        
        # Build result
        result = {
            'text': text,
            'intent': intent,
            'intent_confidence': intent_confidence,
            'entities': processed_entities,
            'metrics': metrics,
            'date_info': date_info,
            'data': data,
            'formatted_data': formatted_data,
            'response': response,
            'has_data': has_data,
            'needs_clarification': False,
            'conversation_state': 'completed',
            'inferred_date': processed_entities.get('_inferred_date', False),
            'inferred_metric': processed_entities.get('_inferred_metric', False),
            'cross_intent': processed_entities.get('_cross_intent', False),
            'graph_state': graph_state,
            'success': True
        }
        
        logger.info(f"✅ Processing complete in {time.perf_counter() - t0:.2f}s")
        return _attach_timing(result)
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return _attach_timing({
            'text': text,
            'error': str(e),
            'response': f"Sorry, I encountered an error: {str(e)}",
            'success': False
        })

def process_audio_and_respond(use_llm: bool = True) -> Dict[str, Any]:
    """Capture audio, transcribe, and process"""
    try:
        audio_capture = get_audio_capture()
        stt = get_speech_to_text()
        
        if not audio_capture or not stt:
            return {
                'error': 'Audio components not loaded',
                'response': 'Audio capture or speech-to-text not available.',
                'success': False
            }
        
        # Reset and capture
        audio_capture.reset()
        audio_bytes = audio_capture.start_listening()
        
        if not audio_bytes:
            return {
                'error': 'No audio captured',
                'response': 'No audio captured. Please try again.',
                'success': False
            }
        
        # Transcribe
        results = stt.transcribe_audio(audio_bytes, beam_size=5)
        
        if not results:
            return {
                'error': 'Transcription failed',
                'response': 'Failed to transcribe audio. Please try again.',
                'success': False
            }
        
        text = results[0]['text']
        logger.info(f"Transcribed: '{text}'")
        
        # Process
        return process_query(text, use_llm)
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}", exc_info=True)
        return {
            'error': str(e),
            'response': f'Error processing audio: {str(e)}',
            'success': False
        }

# ==========================================================
# Streamlit UI
# ==========================================================

def main():
    st.set_page_config(
        page_title="Voice Health Assistant",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('use_gpu', False)
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        .response-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            font-size: 1.2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .user-query {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #2196F3;
            color: black;
        }
        .metric-badge {
            background-color: #4CAF50;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .intent-badge {
            background-color: #2196F3;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .inference-badge {
            background-color: #FF9800;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .influx-badge {
            background-color: #00D4FF;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .langgraph-badge {
            background-color: #9C27B0;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            display: inline-block;
            margin: 0.2rem;
        }
        .clarification-badge {
            background-color: #F44336;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            display: inline-block;
            margin: 0.2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎙️ Voice Health Assistant (InfluxDB)</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 1.1rem; color: #666;'>
        Ask about your real-time health metrics using voice or text - powered by AI & InfluxDB
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        use_llm = st.checkbox(
            "🤖 Use AI Response Generation",
            value=True,
            help="Generate natural language responses using LLM"
        )
        
        show_debug = st.checkbox(
            "🔍 Show Debug Info",
            value=False,
            help="Display detailed processing information"
        )
        
        st.markdown("---")
        
        # System Status
        st.subheader("🟢 System Status")
        
        # Fetch cached components
        _conv_manager = get_conversation_manager()
        _intent_clf = get_intent_classifier()
        _entity_ext = get_entity_extractor()
        _data_mgr = get_data_manager()
        _audio = get_audio_capture()
        _stt = get_speech_to_text()
        _llm = get_llm_processor() if use_llm else None

        components_status = {
            "LangGraph Manager": _conv_manager is not None,
            "Intent Classifier": _intent_clf is not None,
            "Entity Extractor": _entity_ext is not None,
            "InfluxDB Manager": _data_mgr is not None,
            "LLM Processor": (_llm is not None) if use_llm else None,
            "Audio Capture": _audio is not None,
            "Speech-to-Text": _stt is not None,
        }

        for component, status in components_status.items():
            if status is None:
                st.info(f"⊘ {component} (disabled)")
            elif status:
                st.success(f"✓ {component}")
            else:
                st.error(f"✗ {component}")
        
        st.markdown("---")
        
        # InfluxDB Info
        st.subheader("📊 InfluxDB Info")
        if _data_mgr:
            st.success(f"✓ Connected")
            st.text(f"User: {INFLUX_USER_ID}")
            st.text(f"Bucket: {INFLUX_BUCKET}")
            
            try:
                summary = _data_mgr.get_data_summary()
                st.text(f"Metrics: {len(summary.get('available_metrics', []))}")
            except Exception as e:
                st.error(f"Error: {str(e)[:50]}")
        else:
            st.error("✗ Not connected")
        
        st.markdown("---")
        
        # LangGraph Session Info
        st.subheader("🔄 LangGraph Session")
        conv_manager = _conv_manager
        session_id = get_session_id()

        if conv_manager:
            state = conv_manager.get_state(session_id)
            if state:
                st.metric("Turn Count", state.get('turn_count', 0))
                st.text(f"Session: {session_id[:16]}...")
                
                if state.get('last_intent'):
                    st.text(f"Last Intent: {state['last_intent']}")
                if state.get('last_metric'):
                    st.text(f"Last Metric: {state['last_metric']}")
                if state.get('last_date'):
                    date_str = str(state['last_date'])
                    if isinstance(state['last_date'], dict):
                        date_str = state['last_date'].get('value', 'N/A')
                    st.text(f"Last Date: {date_str}")
        
        st.markdown("---")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **Real-time Queries:**
            - "What's my current step count?"
            - "Show me my heart rate right now"
            - "What's my blood pressure?"
            - "How many steps in the last 10 minutes?"
            
            **Historical Queries:**
            - "How many steps yesterday?"
            - "What was my average heart rate today?"
            - "Show me my steps for this week"
            
            **Follow-up Queries:**
            - "What about heart rate?" ← inherits date!
            - "And blood pressure?"
            """)
        
        st.markdown("---")
        
        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.history = []
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.history = []
                if 'session_id' in st.session_state:
                    del st.session_state['session_id']
                st.success("New session started!")
                st.rerun()
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🎙️ Voice Input", "⌨️ Text Input"])
    
    # Voice Input Tab
    with tab1:
        st.subheader("Voice Input")
        st.markdown("Click the button and speak your query")
        
        col1, col2 = st.columns([1, 2])
        
        with col2:
            if st.button("🎤 **Start Recording**", use_container_width=True, type="primary"):
                # Show listening spinner
                with st.spinner("🎧 Listening... Please speak clearly"):
                    audio_capture = get_audio_capture()
                    stt = get_speech_to_text()
                    
                    if not audio_capture or not stt:
                        st.error("Audio components not loaded")
                        return
                    
                    audio_capture.reset()
                    audio_bytes = audio_capture.start_listening()
                    
                    if not audio_bytes:
                        st.error("No audio captured. Please try again.")
                        return
                
                # Show transcribing spinner
                with st.spinner("📝 Transcribing audio..."):
                    results = stt.transcribe_audio(audio_bytes, beam_size=5)
                    if not results:
                        st.error("Failed to transcribe audio. Please try again.")
                        return
                    
                    text = results[0]['text']
                    st.info(f"Transcribed: '{text}'")
                
                # Show processing spinner
                with st.spinner("🤖 Generating response..."):
                    result = process_query(text, use_llm)
                    st.session_state.history.append(result)
                    st.rerun()
    
    # Text Input Tab
    with tab2:
        st.subheader("Text Input")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            text_input = st.text_input(
                "Enter your query",
                placeholder="e.g., What's my current heart rate?",
                key="text_query",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("📤 Send", use_container_width=True, type="primary")
        
        if send_button and text_input:
            with st.spinner("🤖 Processing..."):
                result = process_query(text_input, use_llm)
                st.session_state.history.append(result)
                st.rerun()
    
    # Quick examples (shown when no history)
    if not st.session_state.history:
        st.markdown("---")
        st.markdown("### 💡 Try These Examples:")
        
        examples = [
            "What's my current step count?",
            "Show me my heart rate right now",
            "What's my blood pressure?",
            "How many steps in the last 10 minutes?",
            "What was my average heart rate yesterday?",
            "Show me my steps for this week"
        ]
        
        cols = st.columns(2)
        for idx, example in enumerate(examples):
            with cols[idx % 2]:
                if st.button(f"💬 {example}", key=f"example_{idx}", use_container_width=True):
                    result = process_query(example, use_llm)
                    st.session_state.history.append(result)
                    st.rerun()

    # Display conversation history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")
        
        for i, result in enumerate(reversed(st.session_state.history)):
            # Safely get user query text
            query_text = result.get("text", "Unknown query")
            st.markdown(f'<div class="user-query"><strong>You:</strong> {query_text}</div>', unsafe_allow_html=True)
            
            # Badges with safe access
            badges_html = ""
            if result.get('intent'):
                badges_html += f'<span class="intent-badge">Intent: {result["intent"]}</span>'
            if result.get('metrics'):
                for metric in result['metrics']:
                    badges_html += f'<span class="metric-badge">{metric}</span>'
            if result.get('inferred_date', False):
                badges_html += f'<span class="inference-badge">🔗 Inferred Date</span>'
            if result.get('inferred_metric', False):
                badges_html += f'<span class="inference-badge">🔗 Inferred Metric</span>'
            if result.get('cross_intent', False):
                badges_html += f'<span class="inference-badge">🔀 Cross-Intent</span>'
            if result.get('data') and 'error' not in result.get('data', {}):
                query_type = result.get('data', {}).get('query_type', 'unknown')
                badges_html += f'<span class="influx-badge">⚡ InfluxDB ({query_type})</span>'
            if result.get('success', True):
                badges_html += f'<span class="langgraph-badge">✅ LangGraph</span>'
            if result.get('needs_clarification', False):
                badges_html += f'<span class="clarification-badge">❓ Clarification</span>'
            
            st.markdown(badges_html, unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f'<div class="response-box"><strong>Assistant:</strong> {result["response"]}</div>', unsafe_allow_html=True)
            
            # Debug info
            if show_debug:
                with st.expander("🔍 Debug Information", expanded=False):
                    tabs = st.tabs(["Intent", "Entities", "Data", "InfluxDB", "Raw"])
                    
                    with tabs[0]:
                        st.markdown("**Intent Classification**")
                        st.write(f"Intent: `{result.get('intent', 'N/A')}`")
                        st.write(f"Confidence: `{result.get('intent_confidence', 0):.2%}`")
                    
                    with tabs[1]:
                        st.markdown("**Extracted Entities**")
                        if 'entities' in result:
                            st.json(result['entities'])
                        else:
                            st.info("No entities extracted")
                        
                        # Show context inference
                        if result.get('inferred_date') or result.get('inferred_metric'):
                            st.markdown("**Context Inference:**")
                            if result.get('inferred_date'):
                                st.success("✓ Date inherited from previous conversation")
                            if result.get('inferred_metric'):
                                st.success("✓ Metric inherited from previous conversation")
                    
                    with tabs[2]:
                        st.markdown("**Retrieved Data Summary**")
                        if result.get('formatted_data'):
                            st.text(result['formatted_data'])
                        
                        if result.get('has_data'):
                            st.success("✓ Data found")
                            if result.get('data'):
                                data_summary = result['data'].get('summary', {})
                                st.write(f"Date Info: {result.get('date_info', 'N/A')}")
                                
                                # Display key metrics based on query type
                                for key, value in data_summary.items():
                                    st.write(f"{key}: {value}")
                        else:
                            st.warning("⚠ No data found")
                    
                    with tabs[3]:
                        st.markdown("**InfluxDB Query Details**")
                        if result.get('data'):
                            data = result['data']
                            
                            if 'error' in data:
                                st.error(f"Error: {data['error']}")
                            else:
                                st.write(f"**Query Type:** {data.get('query_type', 'unknown')}")
                                st.write(f"**Metric:** {data.get('metric', 'N/A')}")
                                st.write(f"**Date Range:** {data.get('date_range', 'N/A')}")
                                
                                # Show summary
                                if data.get('summary'):
                                    st.markdown("**Summary:**")
                                    st.json(data['summary'])
                                
                                # Show raw data if available
                                if data.get('data'):
                                    st.markdown("**Raw Data Points:**")
                                    st.json(data['data'][:5])  # Show first 5 records
                        else:
                            st.info("No InfluxDB data available")
                    
                    with tabs[4]:
                        st.markdown("**Raw Result**")
                        # Remove large objects for display
                        display_result = {k: v for k, v in result.items() if k not in ['data', 'graph_state']}
                        st.json(display_result)
                        
                        # Show graph state separately
                        if result.get('graph_state'):
                            st.markdown("**LangGraph State:**")
                            st.json(result['graph_state'])
                        
                        # Show timing
                        if result.get('response_time'):
                            st.metric("Response Time", f"{result['response_time']:.2f}s")
            
            st.markdown("---")
    else:
        # Welcome message
        st.info("👋 Welcome! Click the microphone button or type a question to get started!")
        st.markdown("""
        **Features:**
        - 🔴 **Real-time queries**: Get your current health metrics instantly
        - 📊 **Historical data**: Query data from past days, weeks, or custom ranges
        - 🔗 **Context tracking**: Follow-up questions inherit date and metric from previous queries
        - 💬 **Natural language**: Ask questions naturally, the AI understands context
        
        **Example flow:**
        1. "What's my current step count?" → Gets real-time steps
        2. "What about heart rate?" → Inherits "current" timeframe, shows heart rate
        3. "How was it yesterday?" → Shows yesterday's heart rate
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        logger.exception("Unhandled exception in Streamlit app")
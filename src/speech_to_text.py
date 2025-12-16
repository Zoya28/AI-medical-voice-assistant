import numpy as np
import logging
import torch
from faster_whisper import WhisperModel

# ==========================================================
# Logging Setup
# ==========================================================
logger = logging.getLogger("SpeechToText")

# ==========================================================
# Speech-to-Text Class
# ==========================================================
class SpeechToText:
    """
    Handles speech-to-text transcription using Faster Whisper model.
    Converts audio bytes to text with confidence scores.
    """
    
    def __init__(
        self,
        model_name="small.en",
        device="cuda" if torch.cuda.is_available() else "cpu",
        sample_rate=16000,
    ):
        """
        Initialize the Faster Whisper model.
        
        Args:
            model_name: Whisper model size (e.g., 'small.en')
            device: Device to run the model on ('cuda' or 'cpu')
            sample_rate: Audio sampling rate in Hz
        """
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate

        try:
            logger.info("Loading Whisper model '%s' on %s...", model_name, device)
            self.model = WhisperModel(model_name, device=device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise

    # ------------------------------------------------------
    # Transcription from bytes
    # ------------------------------------------------------
    def transcribe_audio(self, audio_bytes: bytes, beam_size=5):
        """
        Transcribe audio from raw PCM bytes.
        
        Args:
            audio_bytes: Raw PCM audio bytes (int16)
            beam_size: Beam size for transcription (higher = more accurate but slower)
            
        Returns:
            List of dictionaries containing:
                - text: Transcribed text
                - start: Start time of segment
                - end: End time of segment
                - confidence: Confidence score (if available)
        """
        try:
            logger.info("Transcribing audio...")
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            segments, info = self.model.transcribe(audio_array, beam_size=beam_size)
            
            results = []
            for seg in segments:
                result = {
                    'text': seg.text.strip(),
                    'start': seg.start,
                    'end': seg.end,
                    'confidence': getattr(seg, 'avg_logprob', None)  # Some models provide this
                }
                results.append(result)
                logger.info("Transcribed: %s", result['text'])
            
            return results
            
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return []

    # ------------------------------------------------------
    # Transcription from file
    # ------------------------------------------------------
    def transcribe_file(self, audio_path: str, beam_size=5):
        """
        Transcribe audio from a file.
        
        Args:
            audio_path: Path to the audio file
            beam_size: Beam size for transcription
            
        Returns:
            List of dictionaries containing transcription results
        """
        try:
            logger.info("Transcribing file: %s", audio_path)
            
            segments = self.model.transcribe(audio_path, beam_size=beam_size)
            
            results = []
            for seg in segments:
                result = {
                    'text': seg.text.strip(),
                    'start': seg.start,
                    'end': seg.end,
                    'confidence': getattr(seg, 'avg_logprob', None)
                }
                results.append(result)
                logger.info("Transcribed: %s", result['text'])
            
            return results
            
        except Exception as e:
            logger.error("File transcription failed: %s", e)
            return []

    # ------------------------------------------------------
    # Get full transcription text
    # ------------------------------------------------------
    def get_transcription_text(self, audio_bytes: bytes, beam_size=5):
        """
        Get complete transcription as a single string.
        
        Args:
            audio_bytes: Raw PCM audio bytes (int16)
            beam_size: Beam size for transcription
            
        Returns:
            Complete transcription text as string
        """
        results = self.transcribe_audio(audio_bytes, beam_size=beam_size)
        return ' '.join([r['text'] for r in results])

    # ------------------------------------------------------
    # Get confidence score
    # ------------------------------------------------------
    def get_confidence_score(self, results):
        """
        Calculate average confidence score from transcription results.
        
        Args:
            results: List of transcription result dictionaries
            
        Returns:
            Average confidence score or None if not available
        """
        confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        if confidences:
            return sum(confidences) / len(confidences)
        return None

import pyaudio
import numpy as np
import queue
import logging
import threading

# ==========================================================
# Logging Setup
# ==========================================================
logger = logging.getLogger("AudioCapture")

# ==========================================================
# Audio Capture Class
# ==========================================================
class AudioCapture:
    """
    Handles real-time audio streaming from microphone with Voice Activity Detection (VAD).
    Supports continuous listening mode with energy-based speech detection.
    """
    
    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        frame_ms=30,
        energy_threshold=300,
        dynamic_energy=True,
        max_silence_sec=1.0,
        min_segment_sec=0.25,
    ):
        """
        Initialize audio capture with VAD parameters.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            channels: Number of audio channels (1 for mono)
            frame_ms: Frame duration in milliseconds
            energy_threshold: Energy threshold for speech detection
            dynamic_energy: Enable dynamic energy threshold adjustment
            max_silence_sec: Maximum silence duration before stopping recording
            min_segment_sec: Minimum segment duration to be considered valid
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2 * channels)

        self.energy_threshold = energy_threshold
        self.dynamic_energy = dynamic_energy
        self.energy_floor = energy_threshold
        self.max_silence_sec = max_silence_sec
        self.min_segment_sec = min_segment_sec

        self.audio_interface = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        logger.info("AudioCapture initialized with sample_rate=%d, energy_threshold=%d", 
                   sample_rate, energy_threshold)

    # ------------------------------------------------------
    # Energy-based VAD
    # ------------------------------------------------------
    def _is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Determine if audio frame contains speech using energy-based VAD.
        
        Args:
            pcm_bytes: Raw PCM audio bytes
            
        Returns:
            True if speech is detected, False otherwise
        """
        try:
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            if arr.size == 0:
                return False
            rms = np.sqrt(np.mean(arr ** 2))

            if self.dynamic_energy:
                self.energy_floor = 0.99 * self.energy_floor + 0.01 * rms

            threshold = max(self.energy_threshold, self.energy_floor * 1.5)
            return rms > threshold
        except Exception as e:
            logger.debug("VAD check failed: %s", e)
            return False

    # ------------------------------------------------------
    # Audio Callback
    # ------------------------------------------------------
    def _callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio stream callback to queue incoming audio frames.
        
        Args:
            in_data: Audio data from the stream
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status flags
            
        Returns:
            Tuple of (None, pyaudio.paContinue)
        """
        if status:
            logger.warning("Audio stream status: %s", status)
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    # ------------------------------------------------------
    # Recording Loop
    # ------------------------------------------------------
    def start_listening(self, callback_func=None):
        """
        Start continuous audio listening with VAD-based segmentation.
        
        Args:
            callback_func: Optional callback function to process captured audio segments.
                          Should accept bytes as argument.
        
        Returns:
            Captured audio bytes if no callback provided, None otherwise
        """
        try:
            stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_bytes,
                stream_callback=self._callback,
            )
        except Exception as e:
            logger.error("Failed to open audio stream: %s", e)
            return None

        logger.info("Listening... Start speaking.")
        stream.start_stream()

        buffer = b""
        silence_frames = 0
        speaking = False
        captured_audio = None

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._is_speech(frame):
                    if not speaking:
                        logger.info("Speech detected. Recording started.")
                        speaking = True
                    buffer += frame
                    silence_frames = 0
                else:
                    if speaking:
                        silence_frames += 1
                        buffer += frame

                        if silence_frames * (self.frame_ms / 1000.0) > self.max_silence_sec:
                            duration_sec = len(buffer) / (2 * self.sample_rate)
                            if duration_sec >= self.min_segment_sec:
                                logger.info("Utterance ended. Duration: %.2fs", duration_sec)
                                
                                # Process the captured audio
                                if callback_func:
                                    callback_func(buffer)
                                else:
                                    captured_audio = buffer
                                
                                self.stop_event.set()
                                break
                            else:
                                logger.info("Ignored short utterance (%.2fs)", duration_sec)
                            buffer = b""
                            speaking = False
                            silence_frames = 0
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Stopping...")
        except Exception as e:
            logger.error("Error in main loop: %s", e)
        finally:
            stream.stop_stream()
            stream.close()
            logger.info("Audio stream closed.")
            
        return captured_audio

    # ------------------------------------------------------
    # Stop Listening
    # ------------------------------------------------------
    def stop_listening(self):
        """Stop the audio capture process."""
        self.stop_event.set()
        logger.info("Stop event set.")

    # ------------------------------------------------------
    # Reset
    # ------------------------------------------------------
    def reset(self):
        """Reset the audio capture state for a new recording session."""
        self.stop_event.clear()
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("AudioCapture reset for new session.")

    # ------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------
    def close(self):
        """Clean up and terminate the audio interface."""
        self.stop_event.set()
        self.audio_interface.terminate()
        logger.info("AudioCapture closed.")
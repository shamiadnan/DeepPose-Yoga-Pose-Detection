"""
DeepPose 2.0 — Voice Feedback
===============================
Non-blocking text-to-speech feedback using pyttsx3.
Runs speech in a background thread to avoid freezing the video feed.
"""

import os
import sys
import threading
import queue
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VoiceFeedback:
    """Non-blocking text-to-speech engine for corrective voice feedback.
    
    Uses pyttsx3 in a background daemon thread to speak feedback
    without blocking the main inference loop. Implements deduplication
    to avoid repeating the same instruction consecutively.
    
    Args:
        rate: Speech rate in words per minute (default: 150)
    
    Example:
        >>> voice = VoiceFeedback(rate=150)
        >>> voice.speak("Bend your knee more")  # Non-blocking
        >>> voice.speak("Bend your knee more")  # Ignored (duplicate)
        >>> voice.speak("Straighten your back")  # Spoken
        >>> voice.stop()  # Cleanup
    """
    
    def __init__(self, rate: int = None):
        self.rate = rate or config.VOICE_RATE
        self._available = False
        self._last_text: Optional[str] = None
        self._speech_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Try to initialize pyttsx3
        try:
            import pyttsx3
            self._engine_module = pyttsx3
            self._available = True
            self._running = True
            
            # Start background speech worker thread
            self._thread = threading.Thread(
                target=self._speech_worker, daemon=True
            )
            self._thread.start()
            
        except Exception as e:
            print(f"[VoiceFeedback] WARNING: pyttsx3 initialization failed: {e}")
            print(f"[VoiceFeedback] Continuing with visual-only feedback")
            self._available = False
    
    def _speech_worker(self):
        """Background worker thread that processes the speech queue.
        
        Creates a fresh pyttsx3 engine instance in the worker thread
        (required by pyttsx3's threading model) and processes queued text.
        """
        try:
            engine = self._engine_module.init()
            engine.setProperty('rate', self.rate)
        except Exception as e:
            print(f"[VoiceFeedback] Engine init failed in worker: {e}")
            self._available = False
            return
        
        while self._running:
            try:
                # Block with timeout to allow checking _running flag
                text = self._speech_queue.get(timeout=0.5)
                if text is None:  # Poison pill to stop
                    break
                
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[VoiceFeedback] Speech error: {e}")
                    
            except queue.Empty:
                continue
        
        # Cleanup
        try:
            engine.stop()
        except:
            pass
    
    def speak(self, text: str) -> None:
        """Queue text for speech in background thread (non-blocking).
        
        Implements deduplication: if text is identical to the last spoken
        text, the call is silently ignored to prevent repetition.
        
        Args:
            text: Non-empty string to speak
        
        Note:
            Returns immediately. Speech happens asynchronously.
        """
        if not self._available or not text:
            return
        
        # Deduplication: skip if same as last spoken
        if text == self._last_text:
            return
        
        self._last_text = text
        
        # Clear any pending speech (only speak the latest)
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        
        self._speech_queue.put(text)
    
    def stop(self) -> None:
        """Stop speech worker and clean up resources.
        
        Signals the background thread to terminate and waits for it
        to finish gracefully.
        """
        self._running = False
        self._speech_queue.put(None)  # Poison pill
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        self._available = False
    
    @property
    def is_available(self) -> bool:
        """Check if voice feedback is operational.
        
        Returns:
            True if pyttsx3 initialized successfully and worker is running
        """
        return self._available
    
    def reset(self) -> None:
        """Reset the last spoken text to allow re-speaking."""
        self._last_text = None
    
    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.stop()
        except:
            pass

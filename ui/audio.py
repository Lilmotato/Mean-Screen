import streamlit as st
import speech_recognition as sr
import tempfile
from audio_recorder_streamlit import audio_recorder


def record_audio() -> bytes | None:
    """
    Records audio using the browser microphone widget.

    Returns:
        Bytes of recorded audio if available, else None.
    """
    st.markdown("🎙️ **Record your voice** (click to start/stop):")
    return audio_recorder(pause_threshold=2.0)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes recorded audio to text using Google Speech Recognition.

    Args:
        audio_bytes: Raw audio bytes (WAV)

    Returns:
        Transcribed text as a string.
    """
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        with sr.AudioFile(tmp.name) as source:
            audio = recognizer.record(source)

        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            st.warning("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Speech Recognition service: {e}")

    return ""

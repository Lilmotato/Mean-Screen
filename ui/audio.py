import tempfile

import speech_recognition as sr
import streamlit as st
from audio_recorder_streamlit import audio_recorder


def record_audio() -> bytes | None:
    """
    Records audio using the browser microphone widget..
    """
    st.markdown("**Record your voice** (click to start/stop):")
    return audio_recorder(pause_threshold=2.0)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes recorded audio to text using Google Speech Recognition.
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

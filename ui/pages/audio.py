import streamlit as st
from audio import record_audio, transcribe_audio
from client import analyze_text
from components import (
    animated_header, show_classification,
    show_explanation, show_policies, show_recommendation
)

# Show animated header
animated_header("🎙️ Audio Mode", "assets/mic.json")

# Audio input
audio_bytes = record_audio()
if audio_bytes:
    transcribed = transcribe_audio(audio_bytes)
    if transcribed:
        st.success("📝 Transcript:")
        st.markdown(f"```text\n{transcribed}\n```")

        with st.spinner("Analyzing..."):
            result = analyze_text(transcribed)

        show_classification(result)
        show_explanation(result)
        show_policies(result)
        show_recommendation(result)
    else:
        st.warning("Transcription failed. Try again.")

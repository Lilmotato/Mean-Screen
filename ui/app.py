# ui/app.py

"""
Streamlit UI for Hate Speech Detection Bento Box

This interface allows users to:
- Enter text or record audio input
- Analyze the input using a 4-agent LLM pipeline via FastAPI
- View results as a structured Bento Box:
    1. Classification
    2. Policy Retrieval
    3. Reasoning
    4. Recommended Action

Modules Used:
- `client.py` for orchestrator API call
- `components.py` for visualization
- `audio.py` for recording/transcribing audio input

Author: [Your Name]
"""

import time
import streamlit as st
from client import analyze_text
from components import (
    show_classification,
    show_policies,
    show_explanation,
    show_recommendation
)
from audio import record_audio, transcribe_audio  # Audio module

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hate Speech Analyzer", layout="centered")
st.title("🍱 Hate Speech Detection Bento Box")
st.caption("Analyze input for hate speech using a 4-agent LLM pipeline")

# --- Custom CSS Styling ---
st.markdown("""
<style>
.bento {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5rem;
    padding-top: 1rem;
}
.bento-box {
    border: 2px solid #e5e5e5;
    border-radius: 12px;
    padding: 1rem;
    background: #fafafa;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
}
.bento-box h4 {
    margin-top: 0;
}
.agent-flow {
    text-align: center;
    font-weight: bold;
    color: #888;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("### 🎙️ Record Audio or 📝 Enter Text")

text_input = st.text_area("Text Input", height=150)
audio_bytes = record_audio()

# Determine final input source
user_input = None
if audio_bytes:
    st.info("🔄 Transcribing audio...")
    user_input = transcribe_audio(audio_bytes)
    st.text_area("📝 Transcribed Text", value=user_input, height=100)
elif text_input.strip():
    user_input = text_input.strip()

# --- Input Form (Submit Button) ---
if st.button("🚀 Analyze") and user_input:
    flow_status = st.empty()
    result_container = st.container()

    try:
        # Stepwise Flow Status
        flow_status.markdown("⏳ **Step 1/5**: Sending input to orchestrator...")
        result = analyze_text(user_input)

        flow_status.markdown("✅ **Step 2/5**: Detector classified the input...")
        time.sleep(0.4)

        flow_status.markdown("✅ **Step 3/5**: Retriever pulled relevant policies...")
        time.sleep(0.4)

        flow_status.markdown("✅ **Step 4/5**: Reasoner explained the decision...")
        time.sleep(0.4)

        flow_status.markdown("✅ **Step 5/5**: Recommender mapped action...")

        # --- Results Bento Box Display ---
        with result_container:
            st.markdown('<div class="agent-flow">Input → Orchestrator → [Detector, Retriever, Reasoner, Recommender] → Output</div>', unsafe_allow_html=True)
            st.markdown('<div class="bento">', unsafe_allow_html=True)

            st.markdown('<div class="bento-box">', unsafe_allow_html=True)
            show_classification(result)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="bento-box">', unsafe_allow_html=True)
            show_policies(result)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="bento-box">', unsafe_allow_html=True)
            show_explanation(result)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="bento-box">', unsafe_allow_html=True)
            show_recommendation(result)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        flow_status.error(f"❌ Error: {e}")

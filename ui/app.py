import time
import streamlit as st
from client import analyze_text
from components import (
    show_classification,
    show_policies,
    show_explanation,
    show_recommendation
)

# --- Page Setup ---
st.set_page_config(page_title="Hate Speech Analyzer", layout="centered")
st.title("🍱 Hate Speech Detection Bento Box")
st.caption("Analyze input for hate speech using a 4-agent LLM pipeline")

# --- Styling ---
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

# --- Input Form ---
with st.form(key="analyze_form"):
    user_input = st.text_area("✏️ Paste or type your text here", height=150)
    submit = st.form_submit_button("🚀 Analyze")

# --- Execution Flow ---
if submit and user_input.strip():
    flow_status = st.empty()
    response_container = st.container()

    try:
        flow_status.markdown("⏳ **Step 1/5**: Sending text to orchestrator...")

        # Analyze
        result = analyze_text(user_input)

        flow_status.markdown("✅ **Step 2/5**: Detector classified the input...")
        time.sleep(0.4)

        flow_status.markdown("✅ **Step 3/5**: Retriever pulled relevant policy docs...")
        time.sleep(0.4)

        flow_status.markdown("✅ **Step 4/5**: Reasoner explained classification based on policies...")
        time.sleep(0.4)

        flow_status.markdown("✅ **Step 5/5**: Recommender mapped action...")

        # Show Results Bento-style
        with response_container:
            st.markdown('<div class="agent-flow">Text Input → Orchestrator → [Detector, Retriever, Reasoner, Recommender] → Structured Response</div>', unsafe_allow_html=True)
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

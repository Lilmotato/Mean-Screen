import streamlit as st
from client import analyze_text
from components import (
    show_classification,
    show_policies,
    show_explanation,
    show_recommendation,
)

st.set_page_config(page_title="Hate Speech Dashboard", layout="wide")
st.title("🧠 Hate Speech Detection | Mean-Screen")

st.markdown("Analyze input text using a 4-agent moderation system.")

text_input = st.text_area("📝 Enter Text", height=180)

if st.button("🚀 Run Analysis") and text_input.strip():
    with st.spinner("Analyzing input..."):
        try:
            result = analyze_text(text_input.strip())

            st.markdown("## 🧩 Agent Pipeline")
            col1, col2 = st.columns(2)

            with col1:
                show_classification(result)
                show_explanation(result)

            with col2:
                show_policies(result)
                show_recommendation(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# streamlit_app/components.py
import streamlit as st

def show_classification(result):
    st.subheader("🔍 Classification Result")
    st.markdown(f"**Label**: `{result['label']}`")
    st.markdown(f"**Confidence**: `{result['confidence']:.2f}`")
    st.markdown(f"**Reasoning**: {result['reasoning']}")

def show_policies(result):
    st.subheader("📄 Relevant Policies")
    for i, policy in enumerate(result['policies'], 1):
        st.markdown(f"{i}. {policy}")

def show_explanation(result):
    st.subheader("🧠 Explanation")
    st.write(result['explanation'])

def show_recommendation(result):
    st.subheader("✅ Moderation Action")
    st.code(result['recommendation'], language="markdown")

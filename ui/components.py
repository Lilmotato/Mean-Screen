import json
import os
from typing import Any
import streamlit as st
from streamlit_lottie import st_lottie


def load_lottie(filepath: str) -> dict | None:
    """Load Lottie animation JSON."""
    if not os.path.exists(filepath):
        st.warning(f"Missing Lottie file: {filepath}")
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def show_lottie_animation(path: str, height: int = 150, key: str = "lottie"):
    """Display Lottie animation."""
    anim = load_lottie(path)
    if anim:
        st_lottie(anim, height=height, key=key)


def show_classification(result: dict[str, Any]):
    st.subheader("Classification Result")
    classification = result.get("hate_speech", {})
    st.markdown(f"- **Label**: `{classification.get('classification', 'N/A')}`")
    st.markdown(f"- **Confidence**: `{classification.get('confidence', 'N/A')}`")
    st.markdown(
        f"- **Reasoning**: {classification.get('reason', 'No reasoning provided.')}"
    )


def show_policies(result: dict[str, Any]):
    st.subheader("Retrieved Policies")
    policies = result.get("policies", [])

    if not policies:
        st.info("No policies found.")
        return

    for i, policy in enumerate(policies, 1):
        with st.expander(
            f"{policy.get('source', f'Policy {i}')} ({policy.get('relevance_score', 0)}%)"
        ):
            st.write(policy.get("summary", "No summary provided."))


def show_explanation(result: dict[str, Any]):
    st.subheader("Explanation")
    st.markdown(result.get("reasoning", "No explanation available."))


def show_recommendation(result: dict[str, Any]):
    st.subheader("Moderation Action")
    action = result.get("action", {})
    st.markdown(f"- **Action**: `{action.get('action', 'N/A')}`")
    st.markdown(f"- **Severity**: `{action.get('severity', 'N/A')}`")
    st.markdown(f"- **Reasoning**: {action.get('reasoning', 'No reasoning provided.')}")

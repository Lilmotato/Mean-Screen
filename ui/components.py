import streamlit as st
from typing import Any


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

import streamlit as st
from audio import record_audio, transcribe_audio
from client import analyze_text
from components import (
    show_classification,
    show_explanation,
    show_policies,
    show_recommendation,
)
from export_csv import create_csv_buffer, format_analysis_for_csv, generate_filename

st.set_page_config(page_title="Hate Speech Dashboard", layout="wide")
st.title("Hate Speech Detection | Mean-Screen")
st.markdown("Analyze input text using a 4-agent moderation system.")

tab_text, tab_audio = st.tabs(["Text Input", "Voice Input"])

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

with tab_text:
    text_input = st.text_area("Enter Text", height=180, key="manual_text")
    analyze_text_button = st.button("Run Analysis", key="analyze_text")

with tab_audio:
    audio_bytes = record_audio()
    transcribed_text = ""
    if audio_bytes:
        with st.spinner("Transcribing audio..."):
            try:
                transcribed = transcribe_audio(audio_bytes)
                if transcribed:
                    transcribed_text = transcribed
                    st.session_state.transcribed_text = transcribed_text
                    st.success("Audio transcribed.")
                else:
                    st.warning("No speech detected or transcription failed.")
            except Exception as e:
                st.error(f"Transcription error: {e}")

    transcribed_text = st.text_area(
        "Transcribed Text",
        value=st.session_state.transcribed_text,
        height=120,
        key="transcribed_text_display",
    )
    st.session_state.transcribed_text = transcribed_text

    analyze_voice_button = st.button("Run Analysis", key="analyze_voice")

input_to_analyze = ""
should_analyze = False

if analyze_text_button and text_input.strip():
    input_to_analyze = text_input.strip()
    should_analyze = True
elif analyze_voice_button and st.session_state.transcribed_text.strip():
    input_to_analyze = st.session_state.transcribed_text.strip()
    should_analyze = True
elif analyze_text_button:
    st.warning("Please enter some text to analyze.")
elif analyze_voice_button:
    st.warning("Please transcribe audio before analysis.")

if should_analyze:
    with st.spinner("Analyzing input..."):
        try:
            result = analyze_text(input_to_analyze)
            st.session_state.current_result = result
            st.session_state.current_input = input_to_analyze

            csv_record = format_analysis_for_csv(input_to_analyze, result)

            st.markdown("### Analysis Complete")
            st.info(f"Input: {input_to_analyze}")

            csv_buffer = create_csv_buffer([csv_record])
            st.download_button(
                label="Export Result as CSV",
                data=csv_buffer.getvalue(),
                file_name=generate_filename(),
                mime="text/csv",
                key="download_single",
            )

            col1, col2 = st.columns(2)
            with col1:
                show_classification(result)
                show_explanation(result)
            with col2:
                show_policies(result)
                show_recommendation(result)

        except Exception as e:
            st.error(f"Analysis error: {e}")

with st.sidebar:
    st.markdown("## Instructions")
    st.markdown(
        """
    ### Text Input
    - Type or paste text and run analysis.

    ### Voice Input
    - Record speech. It will be auto-transcribed.
    - Review/edit the transcribed text if needed.
    - Run analysis.

    ### Export
    - Download current result as CSV.
    """
    )

    st.markdown("---")
    st.markdown("### System Info")
    st.markdown(
        """
    - Speech Recognition: Google
    - Audio Format: WAV
    - Language: English
    - Export Format: CSV
    """
    )

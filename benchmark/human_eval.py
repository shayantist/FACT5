import streamlit as st
import pandas as pd
import json
from urllib.parse import urlparse

import sys
sys.path.append('../pipeline_v2/')
import main

def load_data(seed):
    model = "mistral"
    df = pd.read_pickle(f'results_v2_{model}.pkl')
    sample_df = df.sample(50, random_state=seed).reset_index(drop=True)
    return sample_df

def main():
    # ----------------------
    # Color variables (editable)
    # ----------------------
    COLOR_HEADER = "#1f77b4"           # For top-level headers (e.g., "Statement Evaluation")
    COLOR_STATEMENT = "#ff7f0e"        # For statement content
    COLOR_VERDICT = "#2ca02c"          # For verdict values (both statement and claims)
    COLOR_CONFIDENCE = "#d62728"       # For confidence numbers
    COLOR_REASONING = "#9467bd"        # For reasoning text
    COLOR_QUESTION = "#9467bd"         # For questions in claim components
    COLOR_ANSWER = "#2ca02c"           # For answers
    COLOR_CITATION = "#d62728"         # For citations

    st.title("LLM FactChecker Human Evaluation")

    # ----------------------
    # Sidebar: Evaluator Info, Global Progress, and Load/Download Progress
    # ----------------------
    st.sidebar.header("Evaluator Information")
    evaluator = st.sidebar.text_input("Evaluator ID")
    seed = st.sidebar.number_input("Seed (for random sampling)", value=42, step=1)

    # Initialize session state variables if not present
    if "data_loaded" not in st.session_state:
        if evaluator and seed is not None:
            st.session_state.sample_df = load_data(seed)
            st.session_state.data_loaded = True
            st.session_state.current_index = 0
            # evaluations stored as a dict: key=index, value=evaluation dict
            st.session_state.evaluations = {}
        else:
            st.sidebar.warning("Please enter both your name and seed to start.")

    # Sidebar: Show progress summary and allow download/upload of progress
    if st.session_state.get("data_loaded", False):
        total = len(st.session_state.sample_df)
        completed = len(st.session_state.evaluations)
        st.sidebar.markdown(f"**Progress:** Completed {completed} out of {total} evaluations.")
        # Optionally list which indices are done:
        if completed:
            done_list = sorted(st.session_state.evaluations.keys())
            st.sidebar.markdown(f"**Done:** {', '.join([str(i+1) for i in done_list])}")
        # Download current progress as JSON
        progress_json = json.dumps(st.session_state.evaluations, indent=4)
        st.sidebar.download_button("Download Progress (JSON)",
                                   data=progress_json,
                                   file_name="progress.json",
                                   mime="application/json")
        # Allow user to load previously saved progress
        uploaded_file = st.sidebar.file_uploader("Upload Progress (JSON)", type=["json"])
        if uploaded_file is not None:
            try:
                loaded_progress = json.load(uploaded_file)
                # Merge the uploaded progress into session_state evaluations
                st.session_state.evaluations.update(loaded_progress)
                st.sidebar.success("Progress loaded successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error("Error loading progress file.")

    # ----------------------
    # Main Content: Display current evaluation if data loaded
    # ----------------------
    if st.session_state.get("data_loaded", False):
        sample_df = st.session_state.sample_df
        idx = st.session_state.current_index

        if idx < len(sample_df):
            row = sample_df.iloc[idx]
            model = "mistral"
            results = row[f'{model}_results']
            result = results[0]
            claims = result['claims']
            statement_text = f"On {row['statement_date']}, {row['statement_originator']} claimed: {row['statement']}"
            reasoning = result['reasoning'].replace('\n', ' ')
            
            # Display Statement details (labels uncolored, dynamic content colored)
            st.markdown(f"<h2 style='color: {COLOR_HEADER};'>Statement Evaluation</h2>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Statement:</strong> <span style='color: {COLOR_STATEMENT};'>{statement_text}</span></p>", unsafe_allow_html=True)            
            st.markdown(f"<p><strong>Overall Verdict:</strong> <span style='color: {COLOR_VERDICT};'>{row['verdict']}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Overall Confidence:</strong> <span style='color: {COLOR_CONFIDENCE};'>{result['confidence']}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Overall Reasoning:</strong> <span style='color: {COLOR_REASONING};'>{reasoning}</span></p>", unsafe_allow_html=True)
            st.markdown("---")

            # Display each claim in a collapsible expander
            for i, claim in enumerate(claims):
                with st.expander(f"Claim {i+1}: {claim.text}"):
                    st.markdown(f"<p><strong>Verdict:</strong> <span style='color: {COLOR_VERDICT};'>{claim.verdict}</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Confidence:</strong> <span style='color: {COLOR_CONFIDENCE};'>{claim.confidence}</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Reasoning:</strong> <span style='color: {COLOR_REASONING};'>{claim.reasoning}</span></p>", unsafe_allow_html=True)
                    
                    # For each question-answer pair, show nicely formatted container
                    for component in claim.components:
                        st.markdown(
                            f"<div style='border:1px solid #d3d3d3; padding:10px; margin-bottom:10px;'>"
                            f"<p><strong>Question:</strong> <span style='color: {COLOR_QUESTION};'>{component.question}</span></p>"
                            f"<p><strong>Answer:</strong> <span style='color: {COLOR_ANSWER};'>{component.answer.text}</span></p>",
                            unsafe_allow_html=True)
                        if component.answer.citations:
                            st.markdown(f"<p><strong>Citations:</strong></p>", unsafe_allow_html=True)
                            for j, citation in enumerate(component.answer.citations, 1):
                                site = urlparse(citation.source_url).netloc.lower()
                                st.markdown(
                                    f"<p style='margin-left:20px;'><span style='color: {COLOR_CITATION};'>"
                                    f"[{j}] {citation.snippet} (Source: <a href='{citation.source_url}'>{citation.source_title} - {site}</a>)"
                                    f"</span></p>",
                                    unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            # Retrieve previously saved evaluation for current index, if any
            saved_eval = st.session_state.evaluations.get(idx, {})
            prev_eval = saved_eval.get("llm_evaluation", None)
            # Determine radio default index if saved
            options = ["STRONGLY AGREE", "AGREE", "DISAGREE", "STRONGLY DISAGREE"]
            default_index = options.index(prev_eval) if prev_eval in options else 0

            # Evaluation radio selection
            evaluation = st.radio("How do you rate the LLM's fact-check?",
                                  options,
                                  key=f"eval_{idx}",
                                  index=default_index)
            
            # Multiselect reasons (only for disagree choices)
            saved_reasons = saved_eval.get("disagree_reasons", [])
            disagree_reasons = []
            if evaluation in ["DISAGREE", "STRONGLY DISAGREE"]:
                disagree_reasons = st.multiselect("Select reasons for disagreement:",
                                                  ["IRRELEVANT/INCORRECT EVIDENCE RETRIEVED",
                                                   "INCORRECT ANALYSIS OF RETRIEVED EVIDENCE"],
                                                  key=f"reasons_{idx}",
                                                  default=saved_reasons)

            # Buttons for navigation and submission
            col_prev, col_submit, col_next = st.columns(3)

            if col_submit.button("Submit Evaluation", key=f"submit_{idx}"):
                eval_dict = {
                    "evaluator": evaluator,
                    "seed": seed,
                    "statement_index": idx,
                    "statement": statement_text,
                    "overall_verdict": row['verdict'],
                    "overall_confidence": result['confidence'],
                    "overall_reasoning": reasoning,
                    "llm_evaluation": evaluation,
                    "disagree_reasons": disagree_reasons
                }
                st.session_state.evaluations[idx] = eval_dict
                st.success("Evaluation saved for this statement.")

            if col_prev.button("Previous", key="prev_button", disabled=(idx == 0)):
                st.session_state.current_index = max(0, idx - 1)
                st.rerun()
            if col_next.button("Next", key="next_button", disabled=(idx == len(sample_df) - 1)):
                st.session_state.current_index = min(len(sample_df) - 1, idx + 1)
                st.rerun()

            st.markdown(f"**Viewing evaluation {idx + 1} of {len(sample_df)}**")

        else:
            # All evaluations completed.
            st.header("You have completed all evaluations!")
            if st.session_state.evaluations:
                final_json = json.dumps(
                    [st.session_state.evaluations[key] for key in sorted(st.session_state.evaluations.keys())],
                    indent=4
                )
                st.download_button(
                    label="Download Evaluations as JSON",
                    data=final_json,
                    file_name="evaluations.json",
                    mime="application/json"
                )
                st.success("Thank you for participating!")
            else:
                st.info("No evaluations have been submitted yet.")

if __name__ == "__main__":
    main()

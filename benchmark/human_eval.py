import streamlit as st
import pandas as pd
import json
from urllib.parse import urlparse

import sys
sys.path.append('../pipeline_v2/')
import main

def load_data(seed, model, num_samples=50):
    df = pd.read_pickle(f'results_v2_{model}.pkl')
    sample_df = df.sample(num_samples, random_state=seed).reset_index(drop=True)
    return sample_df

def main():
    # ----------------------
    # Color variables (editable)
    # ----------------------
    COLOR_CLAIM = "#1f77b4"            # For claim text
    COLOR_HEADER = "#1f77b4"           # For top-level headers (e.g., "Statement Evaluation")
    COLOR_STATEMENT = "#ff7f0e"        # For statement content
    COLOR_VERDICT = "#2ca02c"          # For verdict values (both statement and claims)
    COLOR_CONFIDENCE = "#d62728"       # For confidence numbers
    COLOR_REASONING = "#9467bd"        # For reasoning text
    COLOR_QUESTION = "#9467bd"         # For questions in claim components
    COLOR_ANSWER = "#2ca02c"           # For answers
    COLOR_CITATION = "#d62728"         # For citations

    st.title("LLM FactChecker Human Evaluation")

    # # CUSTOM CSS
    # styles = """
    # div[data-testid="stMarkdownContainer"] p {
    #     font-size: 1.5rem;
    # }
    # """
    # st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        ## 🚀 Getting Started
        1. 👤 Enter your evaluator ID and select model/seed in the sidebar
        2. 📊 Use sidebar to track progress and navigate between statements  
        3. 💾 Download your evaluations anytime using the 'Download Evaluations' button
        
        ## 📝 For Each Statement
        ### Step 1: Review the Analysis
        - ✅ Check the LLM's verdict
        - 💭 Examine reasoning provided
        - 📑 If needed, expand individual claims below to examine evidence
        
        ### Step 2: Rate Agreement Level
        Choose one:
        - 🌟 **STRONGLY AGREE**: Perfect verdict & reasoning
        - ✅ **AGREE**: Mostly correct analysis
        - ⚠️ **DISAGREE**: Significant issues found
        - ❌ **STRONGLY DISAGREE**: Completely incorrect
        
        ### Step 3: If Disagreeing, Select Why
        - 🔍 **IRRELEVANT/INCORRECT EVIDENCE**: Wrong evidence retrieved
        - 🤔 **INCORRECT ANALYSIS**: Evidence interpreted incorrectly
        
        ## 🧭 Navigation Options
        - ⬅️ / ➡️ Use Previous/Next buttons to move between statements
        - 📑 OR use the sidebar to navigate to a specific statement
        - 💾 Download progress anytime using the button on the sidebar
                    
        ‼️ If you don't see anything below, enter your evaluator ID and select a model/seed on the sidebar and press Enter.
        """)

    # ----------------------
    # Sidebar: Evaluator Info, Global Progress, and Load/Download Progress
    # ----------------------
    st.sidebar.header("Evaluator Information")
    evaluator = st.sidebar.text_input("Evaluator ID")
    model = st.sidebar.selectbox("Model", ["gemini", "mistral"])
    seed = st.sidebar.number_input("Seed (for random sampling)", value=42, step=1)
    num_samples = st.sidebar.number_input("Number of samples to evaluate", value=50, step=5)

    # Initialize session state variables if not present
    if "data_loaded" not in st.session_state:
        if evaluator and seed is not None:
            st.session_state.sample_df = load_data(seed, model, num_samples=num_samples)
            st.session_state.data_loaded = True
            st.session_state.current_index = 0
            # evaluations stored as a dict: key=index, value=evaluation dict
            st.session_state.evaluations = {}
        else:
            st.sidebar.warning("Please enter your ID to start, and double check your model, seed, and number of samples.")

    # Sidebar: Show progress summary and allow download/upload of progress
    if st.session_state.get("data_loaded", False):
        total = len(st.session_state.sample_df)
        completed = len(st.session_state.evaluations)
        st.sidebar.markdown(f"**Progress:** {completed}/{total} evaluations completed")
        st.sidebar.progress(completed / total)
        # Show which statements have been evaluated (and which are remaining)
        # Split indices into completed and remaining
        all_indices = list(range(len(st.session_state.sample_df)))
        done_list = sorted(st.session_state.evaluations.keys())
        remaining_list = [i for i in all_indices if i not in done_list]
        
        # Show completed statements section in a collapsible container
        with st.sidebar.expander("**✓ Completed Evaluations**", expanded=False):
            for idx in done_list:
                st.button(
                    f"Statement {idx+1} ✓", 
                    key=f"done_{idx}",
                    on_click=lambda i=idx: setattr(st.session_state, 'current_index', i),
                    type="primary"  # Green/blue styling
                )                
        # Show remaining statements section in a collapsible container   
        with st.sidebar.expander("**⏳ Remaining Evaluations**", expanded=True):
            for idx in remaining_list:
                st.button(
                    f"Statement {idx+1}",
                    key=f"todo_{idx}",
                    on_click=lambda i=idx: setattr(st.session_state, 'current_index', i),
                    type="secondary"  # Gray styling
                    )
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
            results = row[f'{model}_pipeline_results']
            result = results[0]
            claims = result['claims']
            statement_text = f"On {row['statement_date']}, {row['statement_originator']} claimed: {row['statement']}"
            reasoning = result['reasoning'].replace('\n', ' ')
            
            # Display Statement details (labels uncolored, dynamic content colored)
            st.markdown(f"<h2 style='color: {COLOR_HEADER};'>Statement Evaluation [{idx + 1} of {len(sample_df)}]</h2>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Statement:</strong> <span style='color: {COLOR_STATEMENT};'>{statement_text}</span></p>", unsafe_allow_html=True)            
            st.markdown(f"<p><strong>Overall Verdict:</strong> <span style='color: {COLOR_VERDICT};'>{row['verdict']}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Overall Confidence:</strong> <span style='color: {COLOR_CONFIDENCE};'>{result['confidence']}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Overall Reasoning:</strong> <span style='color: {COLOR_REASONING};'>{reasoning}</span></p>", unsafe_allow_html=True)
            st.markdown("---")

            # Display each claim in a collapsible expander
            st.markdown(f"<h3 style='color: {COLOR_HEADER};'>Claims Extracted & Independently Verified: {len(claims)}</h2>", unsafe_allow_html=True)
            for i, claim in enumerate(claims):
                with st.expander(f"Claim {i+1} of {len(claims)}"):
                    st.markdown(f"<p><strong>Claim:</strong> <span style='color: {COLOR_CLAIM};'>{claim.text}</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Verdict:</strong> <span style='color: {COLOR_VERDICT};'>{claim.verdict}</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Confidence:</strong> <span style='color: {COLOR_CONFIDENCE};'>{claim.confidence}</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Reasoning:</strong> <span style='color: {COLOR_REASONING};'>{claim.reasoning}</span></p>", unsafe_allow_html=True)
                    
                    # For each question-answer pair, show nicely formatted container
                    for component in claim.components:
                        with st.container(border=True):
                            answer_text = component.answer.text
                            answer_text = answer_text.replace('\n', ' ')
                            st.markdown(
                                f"<p><strong>Question:</strong> <span style='color: {COLOR_QUESTION};'>{component.question}</span></p>"
                                f"<p><strong>Answer:</strong> <span style='color: {COLOR_ANSWER};'>{component.answer.text}</span></p>",
                                unsafe_allow_html=True)
                            
                            if component.answer.citations:
                                st.markdown(f"<p><strong>Citations:</strong></p>", unsafe_allow_html=True)
                                for j, citation in enumerate(component.answer.citations, 1):
                                    site = urlparse(citation.source_url).netloc.lower()
                                    st.markdown(
                                        f"<p style='margin-left:20px;'><span style='color: {COLOR_CITATION};'>"
                                        f"[{j}] {citation.snippet} <br>— <a style='margin-left:20px;' href='{citation.source_url}'>{citation.source_title} - {site}</a>"
                                        f"</span></p>",
                                        unsafe_allow_html=True)
                            else: 
                                st.markdown(f"<p><strong>No explicit citations made by LM, but relevant documents used to synthesize answer:</strong></p>", unsafe_allow_html=True)
                                if component.answer.retrieved_docs:
                                    unique_sources = {doc.metadata['url']: (doc.metadata['title'], doc.content) for doc in component.answer.retrieved_docs}
                                    for url, (title, content) in unique_sources.items():
                                        st.markdown(
                                            f"<p style='margin-left:20px;'><span style='color: {COLOR_CITATION};'>"
                                            f"{content} — <a href='{url}'>{title}</a></span></p>",
                                            unsafe_allow_html=True)

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
                st.rerun()

            # If evaluation is already saved, show a success message
            if idx in st.session_state.evaluations:
                st.success("Evaluation done for this statement.")
            else:
                st.error("Evaluation not done for this statement.")

            if col_prev.button("Previous", key="prev_button", disabled=(idx == 0)):
                st.session_state.current_index = max(0, idx - 1)
                st.rerun()
            if col_next.button("Next", key="next_button", disabled=(idx == len(sample_df) - 1)):
                st.session_state.current_index = min(len(sample_df) - 1, idx + 1)
                st.rerun()

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

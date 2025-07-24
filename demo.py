import streamlit as st
import streamlit_nested_layout

from dataclasses import dataclass
import sys
import os
from typing import List
import time
import dspy

# Fix for pytorch path class instantiation error
import torch
torch.classes.__path__ = []

import dotenv
dotenv.load_dotenv(override=True)

# Import base classes and utilities from pipeline_v3 package
from pipeline_v3.main import (
    Document, VectorStore, SearchProvider, FactCheckPipeline
)

# Constants for Search Provider
NUM_SEARCH_RESULTS = 10
SCRAPE_TIMEOUT = 5

# Constants for Retrieval (Vector DB + BM25)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USE_BM25 = True
BM25_WEIGHT = 0.5

# Colors for UI (restored from original)
COLORS = {
    'HEADER': "#1f77b4",        # For top-level headers
    'STATEMENT': "#ff7f0e",     # For statement content
    'VERDICT': "#2ca02c",       # For verdict values
    'CONFIDENCE': "#d62728",    # For confidence numbers
    'REASONING': "#9467bd",     # For reasoning text
    'QUESTION': "#9467bd",      # For questions in claim components
    'ANSWER': "#2ca02c",        # For answers
    'CITATION': "#d62728",      # For citations
    'CLAIM': "#1f77b4"          # For claim text
}

class StreamlitFactCheckPipeline(FactCheckPipeline):
    """Enhanced FactCheckPipeline with improved Streamlit UI components"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create UI state containers
        self.status_container = st.empty()
        self.progress_bar = st.progress(0)
        self.results_container = st.empty()
        
        # Disable verbose console output
        global VERBOSE
        VERBOSE = False
        
    def update_status(self, message: str, progress: float = None):
        """Update status message and progress bar"""
        if progress < 1:
            self.status_container.info(f"**Status:** 🔄 {message}")
        else:
            self.status_container.success(f"**Status:** 🔄 {message}")
        if progress is not None:
            self.progress_bar.progress(progress)
        time.sleep(0.01)  # Allow UI to update
    
    def fact_check(self, statement: str, web_search: bool = True):
        """Enhanced fact-check with linear layout and original colors"""
        
        # Reset containers
        self.results_container.empty()
        time.sleep(0.01)

        with self.results_container.container():
            st.markdown("### Pipeline Progress")
            
            # Step 1: Extract Claims
            claims_container = st.expander("Step 1: Claim Extraction", expanded=True)
            with claims_container:
                self.update_status("Extracting claims...", 0.1)
                st.info("🔬 Analyzing statement to extract verifiable claims...")
                
                try:
                    claims = self.claim_extractor(statement)
                    
                    st.markdown(f"**Extracted {len(claims)} claims:**")
                    for i, claim in enumerate(claims, 1):
                        st.markdown(f"<p style='margin-left: 20px;'><strong>Claim {i}:</strong> <span style='color: {COLORS['CLAIM']};'>{claim.text}</span></p>", unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Failed to extract claims: {str(e)}")
                    return None, None, None, None
            
            # Step 2: Process each claim
            for claim_i, claim in enumerate(claims, 1):
                # Generate questions for the claim
                questions_container = st.expander(f"Step 2.{claim_i}: Question Generation for Claim {claim_i}", expanded=True)
                with questions_container:
                    self.update_status(f"Generating questions for claim {claim_i}/{len(claims)}...", 0.2 + (0.1 * (claim_i / len(claims))))
                    
                    st.markdown(f"<p><strong>Claim {claim_i}:</strong> <span style='color: {COLORS['CLAIM']};'>{claim.text}</span></p>", unsafe_allow_html=True)
                    
                    try:
                        components = self.question_generator(statement, claim)
                        
                        for j, component in enumerate(components, 1):
                            st.markdown(f"<p style='margin-left: 20px;'><strong>Question {j}:</strong> <span style='color: {COLORS['QUESTION']};'>{component.question}</span></p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='margin-left: 20px;'><strong>Search Queries:</strong> <span style='color: {COLORS['STATEMENT']};'>{component.search_queries}</span></p>", unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Failed to generate questions: {str(e)}")
                        continue

                # Process each component (question)
                for component_i, component in enumerate(components, 1):
                    evidence_container = st.expander(f"Step 3.{component_i}: Evidence Collection for Question {component_i}", expanded=True)
                    with evidence_container:
                        progress_val = 0.3 + (0.3 * ((claim_i-1)/len(claims) + (component_i-1)/len(components)/len(claims)))
                        self.update_status(f"Collecting evidence for claim {claim_i}, question {component_i}...", progress_val)
                        
                        st.markdown(f"<p><strong>Question:</strong> <span style='color: {COLORS['QUESTION']};'>{component.question}</span></p>", unsafe_allow_html=True)
                        
                        # Search and retrieve documents
                        relevant_docs = []
                        for query_i, query in enumerate(component.search_queries, 1):                                        
                            # If web search is enabled, perform web search
                            if web_search and self.search_provider:
                                st.markdown(f"Searching query {query_i} on the web: `{query}`")
                                try:
                                    search_results = self.search_provider.search(query, NUM_SEARCH_RESULTS)
                                    
                                    # Display search results in data table
                                    st.markdown(f"Retrieved {len(search_results)} sources from the web:")
                                    st.dataframe(search_results, use_container_width=True)
                                    
                                    # Add search results to vector store
                                    documents, metadata = [], []
                                    for result in search_results:
                                        if result.excerpt:
                                            documents.append(result.excerpt)
                                            metadata.append({
                                                "title": result.title,
                                                "url": result.url,
                                                "source": result.source
                                            })
                                    
                                    if documents:
                                        self.vector_store.add_documents(documents, metadata)
                                        
                                except Exception as e:
                                    st.warning(f"Search failed for query: {query}")
                            
                            # Get relevant documents from vector store
                            docs = self.vector_store.retrieve(query, k=self.num_retrieved_docs)
                            relevant_docs.extend(docs)
                        
                        # Synthesize answer
                        st.markdown("*Synthesizing answer...*")
                        try:
                            answer, has_sufficient_info = self.answer_synthesizer(component, documents=relevant_docs)
                            
                            st.markdown(f"""
                            <p style='margin-left:20px;'>
                            <span style='color: {COLORS['ANSWER']};'>{answer.text}</span>
                            </p>
                            """, unsafe_allow_html=True)
                            
                            # Display citations
                            if answer.citations:
                                for i, citation in enumerate(answer.citations, 1):
                                    if citation:
                                        st.markdown(f"""
                                        <p style='margin-left:40px; color: {COLORS['CITATION']};'>
                                        [{i}] {citation.snippet}<br>
                                        — <a href='{citation.source_url}'>{citation.source_title}</a>
                                        </p>
                                        """, unsafe_allow_html=True)
                            
                            component.answer = answer
                            
                        except Exception as e:
                            st.error(f"Failed to synthesize answer: {str(e)}")
                            continue
                
                # Set claim components (questions and answers)
                claim.components = components
                
                # Step 4: Evaluate claim
                evaluation_container = st.expander(f"Step 4: Evaluating Claim {claim_i}", expanded=True)
                with evaluation_container:
                    progress_val = 0.6 + (0.2 * (claim_i / len(claims)))
                    self.update_status(f"Evaluating claim {claim_i}/{len(claims)}...", progress_val)
                    
                    try:
                        verdict, confidence, reasoning = self.claim_evaluator(claim)
                        
                        st.markdown(f"<p><strong>Claim:</strong> <span style='color: {COLORS['CLAIM']};'>{claim.text}</span></p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Verdict:</strong> <span style='color: {COLORS['VERDICT']};'>{verdict}</span> <span style='color: {COLORS['CONFIDENCE']};'>({confidence*100:.2f}% confidence)</span></p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Reasoning:</strong> <span style='color: {COLORS['REASONING']};'>{reasoning}</span></p>", unsafe_allow_html=True)
                        
                        # Display question-answer pairs used for evaluation
                        st.markdown("**Evidence used for evaluation:**")
                        for j, component in enumerate(claim.components, 1):
                            st.markdown(f"""
                            <p style='margin-left:20px;'>
                            <strong>Question {j}:</strong> <span style='color: {COLORS['QUESTION']};'>{component.question}</span><br>
                            <strong>Answer:</strong> <span style='color: {COLORS['ANSWER']};'>{component.answer.text}</span>
                            </p>
                            """, unsafe_allow_html=True)
                        
                        claim.verdict, claim.confidence, claim.reasoning = verdict, confidence, reasoning
                        
                    except Exception as e:
                        st.error(f"Failed to evaluate claim: {str(e)}")
                        continue
            
            # Step 5: Overall Evaluation
            final_container = st.expander("Step 5: Final Verdict", expanded=True)
            with final_container:
                self.update_status("Determining final verdict...", 0.9)
                
                try:
                    # If multiple claims, evaluate overall statement
                    if len(claims) > 1:
                        overall_verdict, overall_confidence, overall_reasoning = self.overall_statement_evaluator(statement, claims)
                    else:
                        overall_verdict = claims[0].verdict
                        overall_confidence = claims[0].confidence
                        overall_reasoning = claims[0].reasoning
                    
                    st.markdown(f"<h2 style='color: {COLORS['HEADER']};'>Statement Evaluation</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Statement:</strong> <span style='color: {COLORS['STATEMENT']};'>{statement}</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Overall Verdict:</strong> <span style='color: {COLORS['VERDICT']};'>{overall_verdict}</span> <span style='color: {COLORS['CONFIDENCE']};'>({overall_confidence*100:.2f}% confidence)</span></p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Overall Reasoning:</strong> <span style='color: {COLORS['REASONING']};'>{overall_reasoning}</span></p>", unsafe_allow_html=True)
                    
                    # Display breakdown of claims
                    st.markdown("### Breakdown of Claims")
                    for i, claim in enumerate(claims, 1):
                        st.markdown(f"""
                        <p>
                        <strong>Claim {i}:</strong> <span style='color: {COLORS['CLAIM']};'>{claim.text}</span><br>
                        <strong>Verdict:</strong> <span style='color: {COLORS['VERDICT']};'>{claim.verdict}</span> 
                        <span style='color: {COLORS['CONFIDENCE']};'>({claim.confidence*100:.2f}% confidence)</span>
                        </p>
                        """, unsafe_allow_html=True)
                    
                    self.update_status("Fact-check complete!", 1.0)
                    return overall_verdict, overall_confidence, overall_reasoning, claims
                    
                except Exception as e:
                    st.error(f"Failed to generate final verdict: {str(e)}")
                    return None, None, None, None

def create_sidebar():
    """Create enhanced sidebar configuration"""
    st.sidebar.header("🔧 Pipeline Configuration")
    
    # Model selection with better descriptions
    st.sidebar.subheader("🤖 Language Model")
    model_options = {
        "gemini/gemini-2.0-flash": "gemini/gemini-2.0-flash (free)",
        "openai/gpt-4o-mini": "openai/gpt-4o-mini",
        "anthropic/claude-3-7-sonnet": "anthropic/claude-3-7-sonnet",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "CUSTOM": "Custom Model"
    }
    
    model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()), 
                                     format_func=lambda x: model_options[x])
    
    if model_name == "CUSTOM":
        model_name = st.sidebar.text_input("Custom Model (LiteLLM format)", 
                                          placeholder="e.g., openai/gpt-4o")
    
    # API key handling
    api_key = None
    if model_name == 'gemini/gemini-2.0-flash':
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            st.sidebar.info("ℹ️ Using default Gemini API key")
    elif model_name.startswith('openrouter/'):
        api_key = st.sidebar.text_input("🔑 OpenRouter API Key", type="password")
    elif model_name.startswith('anthropic/'):
        api_key = st.sidebar.text_input("🔑 Anthropic API Key", type="password")
    elif model_name.startswith('openai/'):
        api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    
    st.sidebar.markdown("---")
    
    # Search configuration
    st.sidebar.subheader("🌐 Web Search")
    use_web_search = st.sidebar.checkbox("Enable Web Search", value=True)
    search_provider = None
    search_api_key = None
    
    if use_web_search:
        search_options = {
            "tavily": "Tavily AI Search",
            "serper": "Google (via Serper)",
            "duckduckgo": "DuckDuckGo (Free)",
        }
        search_provider = st.sidebar.selectbox("Search Provider", list(search_options.keys()),
                                              format_func=lambda x: search_options[x])
        
        if search_provider == "serper":
            search_api_key = os.getenv('SERPER_API_KEY')
            if not search_api_key:
                search_api_key = st.sidebar.text_input("🔑 Serper API Key", type="password")
        elif search_provider == "tavily":
            search_api_key = os.getenv('TAVILY_API_KEY')
            if not search_api_key:
                search_api_key = st.sidebar.text_input("🔑 Tavily API Key", type="password")
    
    st.sidebar.markdown("---")
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        self_correct_answer = st.checkbox("Self-correct Answers", value=False,
                                        help="Regenerate search if insufficient information")
        self_correct_claim = st.checkbox("Self-correct Claims", value=False,
                                       help="Regenerate questions if claim unverifiable")
        
        max_retries = 1
        if self_correct_answer or self_correct_claim:
            max_retries = st.slider("Max Retries", 1, 5, 3)
        
        retriever_k = st.slider("Documents per Query", 3, 20, 10,
                               help="Number of documents to retrieve per search query")
        
        # Context documents
        use_context = st.checkbox("Add Context Documents", value=False)
        context_doc = None
        if use_context:
            context_text = st.text_area("Context Information",
                                       placeholder="Enter relevant background information...")
            if context_text:
                context_doc = Document(
                    content=context_text,
                    metadata={"title": "User Context", "url": ""}
                )
    
    return {
        "model_name": model_name,
        "api_key": api_key,
        "use_web_search": use_web_search,
        "search_provider": search_provider,
        "search_api_key": search_api_key,
        "self_correct_answer": self_correct_answer,
        "self_correct_claim": self_correct_claim,
        "max_retries": max_retries,
        "retriever_k": retriever_k,
        "context_doc": context_doc
    }

def main():
    st.set_page_config(
        page_title="FACT5: LLM Fact-Checking Pipeline",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔍 FACT5: LLM Fact-Checking Pipeline")
    st.info("💡 **Demo for ACL 2025** by Shayan Chowdhury, Sunny Fang, & Smaranda Muresan")
    
    # App description
    st.markdown("""
    As part of the **FACT5** paper, this demo automatically fact-checks statements by: (1) Breaking statements into atomic claims,
      (2) Generating targeted research questions,
      (3) Searching for evidence online,
      (4) Synthesizing answers with citations,
      (5) Evaluating truthfulness (into 5 categories + unverifiable) with confidence scores,
      (6) Providing transparent reasoning. 
    """)
    st.markdown("""Try out the pipeline on any of the following examples or enter your own statement to fact-check.""")
    
    
    # Get configuration from sidebar
    config = create_sidebar()
    
    # Main input area
    st.markdown("### 📝 Enter Statement to Fact-Check")
    
    # Initialize session state for statement if not exists
    if 'statement_text' not in st.session_state:
        st.session_state.statement_text = ""
    
    # Example statements
    st.markdown("**Quick Examples:**")
    
    # Simple examples (single claims)
    # st.markdown("*Simple statements:*")
    example_col1, example_col2, example_col3, example_col4 = st.columns(4)
    
    simple_examples = [
        "The Earth is flat",
        "COVID-19 vaccines contain microchips", 
        "Climate change is a hoax and the earth is not getting hotter because winters are still cold",
        "Immigrants are invading our country and replacing our cultural and ethnic background"
    ]
    
    with example_col1:
        if st.button("🌍 Earth is flat", help=simple_examples[0]):
            st.session_state.statement_text = simple_examples[0]
    
    with example_col2:
        if st.button("💉 Vaccine microchips", help=simple_examples[1]):
            st.session_state.statement_text = simple_examples[1]
    
    with example_col3:
        if st.button("🌡️ Climate hoax", help=simple_examples[2]):
            st.session_state.statement_text = simple_examples[2]

    with example_col4:
        if st.button("👥 Immigrants invading", help=simple_examples[3]):
            st.session_state.statement_text = simple_examples[3]
    
    # Complex examples (multiple claims)
    # st.markdown("*Complex statements (multiple claims):*")
    complex_col1, complex_col2, complex_col3, complex_col4 = st.columns(4)
    
    complex_examples = [
        "Joe Biden won the 2020 presidential election by over 7 million votes, becoming the 46th president at age 78, and was previously Obama's vice president for 8 years.",
        "A significant significant portion of the internet's content is generated by AI or bots. The date given for this death is generally around 2020.",
        "The Great Wall of China is over 13,000 miles long, took more than 2,000 years to build, is visible from space with the naked eye, and required millions of workers.",
        "Tesla was founded by Elon Musk in 2003, became the world's most valuable automaker in 2020, and has sold over 1 million electric vehicles annually since 2021."
    ]
    
    with complex_col1:
        if st.button("🗳️ 2020 Election Details", help=complex_examples[0]):
            st.session_state.statement_text = complex_examples[0]
    with complex_col2:
        if st.button("🤖 Dead Internet", help=complex_examples[1]):
            st.session_state.statement_text = complex_examples[1]
    with complex_col3:
        if st.button("🏛️ Great Wall Facts", help=complex_examples[2]):
            st.session_state.statement_text = complex_examples[1]
    with complex_col4:
        if st.button("🚗 Tesla History", help=complex_examples[3]):
            st.session_state.statement_text = complex_examples[3]
    
    # Main text input
    statement = st.text_area(
        "Statement to fact-check:",
        value=st.session_state.statement_text,
        height=120,
        placeholder="Enter a statement to fact-check (e.g., 'The Earth is flat and was proven by NASA in 2023')",
        key="statement_input"
    )
    
    # Update session state when text area changes
    if statement != st.session_state.statement_text:
        st.session_state.statement_text = statement
    
    # Submit button
    submitted = st.button("🚀 Start Fact-Check", type="primary", use_container_width=True)
    
    # Validation and execution
    if submitted or statement:
        if not statement:
            st.error("⚠️ Please enter a statement to fact-check.")
            return
        
        if not config["model_name"]:
            st.error("⚠️ Please select a language model.")
            return
        
        if config["model_name"] != 'gemini/gemini-2.0-flash' and not config["api_key"]:
            st.error("⚠️ Please enter an API key for the selected model.")
            return
        
        if config["use_web_search"] and config["search_provider"] == "serper" and not config["search_api_key"]:
            st.error("⚠️ Please enter a Serper API key for web search.")
            return
        
        # Initialize and run pipeline
        try:
            # Initialize language model
            lm = dspy.LM(config["model_name"], api_key=config["api_key"], 
                        max_tokens=4096, temperature=0.0)
            
            with dspy.context(lm=lm):
                # Initialize search provider
                search_provider_instance = None
                if config["use_web_search"]:
                    search_provider_instance = SearchProvider(
                        provider=config["search_provider"],
                        api_key=config["search_api_key"]
                    )
                
                # Initialize pipeline
                pipeline = StreamlitFactCheckPipeline(
                    model=lm,
                    vector_store=VectorStore(
                        model_name=EMBEDDING_MODEL,
                        use_bm25=USE_BM25,
                        bm25_weight=BM25_WEIGHT
                    ),
                    search_provider=search_provider_instance,
                    num_retrieved_docs=config["retriever_k"],
                    context=[config["context_doc"]] if config["context_doc"] else None,
                    self_correct_per_answer=config["self_correct_answer"],
                    self_correct_per_claim=config["self_correct_claim"],
                    num_retries_per_answer=config["max_retries"],
                    num_retries_per_claim=config["max_retries"]
                )
                
                # Run fact-check
                result = pipeline.fact_check(statement, web_search=config["use_web_search"])
                
                if result[0] is not None:
                    st.success("✅ Fact-check completed successfully!")
                else:
                    st.error("❌ Fact-check failed. Please try again.")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            with st.expander("🐛 Error Details"):
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
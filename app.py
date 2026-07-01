"""
app.py
Professional Streamlit UI for Multi-Agent Research System
"""

import streamlit as st
from pipeline import run_research_pipeline
from rag import add_pdf_to_vectorstore, search_documents
import tempfile
import os
import time

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Custom CSS — Professional Dark Theme
# -----------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #ffffff15;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8892a4;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Pipeline step cards */
    .pipeline-step {
        background: #1a1f2e;
        border: 1px solid #ffffff10;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        transition: all 0.3s ease;
    }
    .pipeline-step.active {
        border-color: #4f8ef7;
        background: #1a2744;
        box-shadow: 0 0 20px rgba(79,142,247,0.15);
    }
    .pipeline-step.done {
        border-color: #22c55e40;
        background: #0f2419;
    }
    .step-icon { font-size: 1.3rem; }
    .step-name { color: #ffffff; font-weight: 600; font-size: 0.9rem; }
    .step-desc { color: #8892a4; font-size: 0.78rem; }
    .step-status { margin-left: auto; font-size: 0.8rem; }

    /* Metric cards */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #ffffff10;
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.3rem 0;
    }
    .metric-label { color: #8892a4; font-size: 0.8rem; font-weight: 500; }
    .metric-sub { color: #4f8ef7; font-size: 0.78rem; margin-top: 0.3rem; }

    /* Keyword badges */
    .keyword-badge {
        display: inline-block;
        background: #1e2d4a;
        border: 1px solid #4f8ef740;
        color: #4f8ef7;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 0.2rem;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #ffffff10;
    }

    /* Tab content */
    .report-content {
        background: #1a1f2e;
        border: 1px solid #ffffff10;
        border-radius: 12px;
        padding: 1.5rem;
        color: #d1d5db;
        line-height: 1.7;
        font-size: 0.92rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg { background-color: #1a1f2e; }

    /* Input styling */
    .stTextInput input {
        background-color: #1a1f2e !important;
        border: 1px solid #ffffff20 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        padding: 0.7rem 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🔎 Multi-Agent Research Assistant</h1>
    <p>Autonomous AI pipeline powered by Groq LLM · Tavily Search · RAG · NLP · Deep Learning · ML Scoring</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Document Upload (RAG)")
    st.caption("Upload PDFs to include in research alongside web search")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    uploaded_docs_context = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            result = add_pdf_to_vectorstore(tmp_path)
            st.success(f"✅ {uploaded_file.name}")
            os.unlink(tmp_path)

    st.divider()

    st.markdown("### ⚙️ Pipeline Architecture")
    st.markdown("""
    **Phase 1 — Agents**
    - 🔍 Search Agent (Tavily)
    - 📖 Reader Agent (BeautifulSoup)
    - ✍️ Writer Chain (LCEL)
    - 🧐 Critic Chain (LCEL)

    **Phase 2 — Analysis**
    - 📄 RAG (ChromaDB + PDFs)
    - 📊 NLP (NLTK + TextBlob)
    - 🤖 DL Sentiment (BERT)
    - ⭐ ML Quality (Scikit-learn)
    """)

    st.divider()
    st.markdown("### 🛠️ Stack")
    st.code("""LangChain + Groq
Tavily + BeautifulSoup
ChromaDB (RAG)
NLTK + TextBlob (NLP)
HuggingFace BERT (DL)
Scikit-learn (ML)
Streamlit (UI)""", language="text")

# -----------------------------------------------------
# Main Input
# -----------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Latest developments in AI agents in 2026...",
        label_visibility="collapsed",
        key="topic"
    )
with col_btn:
    run_button = st.button("🚀 Research", type="primary", use_container_width=True)

# -----------------------------------------------------
# Pipeline Steps Definition
# -----------------------------------------------------
PIPELINE_STEPS = [
    ("🔍", "Search Agent",      "Searching live web via Tavily API"),
    ("📖", "Reader Agent",      "Scraping & extracting content from sources"),
    ("✍️", "Writer Chain",      "Writing structured research report"),
    ("🧐", "Critic Chain",      "Reviewing & scoring the report"),
    ("📊", "NLP Analysis",      "Extracting keywords & sentiment via NLTK"),
    ("🤖", "BERT Sentiment",    "Deep Learning sentiment analysis"),
    ("⭐", "ML Quality Score",  "Predicting report quality via ML model"),
]

# -----------------------------------------------------
# Run Pipeline
# -----------------------------------------------------
if run_button:
    if not topic.strip():
        st.warning("⚠️ Please enter a research topic first.")
    else:
        st.markdown('<p class="section-header">🔄 Live Pipeline Execution</p>', unsafe_allow_html=True)

        # Create placeholder for live pipeline display
        pipeline_placeholder = st.empty()

        def render_pipeline(current_step: int, statuses: list):
            html = ""
            for i, (icon, name, desc) in enumerate(PIPELINE_STEPS):
                if i < current_step:
                    css_class = "pipeline-step done"
                    status_html = '<span style="color:#22c55e">✅ Done</span>'
                elif i == current_step:
                    css_class = "pipeline-step active"
                    status_html = '<span style="color:#4f8ef7">⚡ Running...</span>'
                else:
                    css_class = "pipeline-step"
                    status_html = '<span style="color:#4b5563">⏳ Waiting</span>'

                html += f"""
                <div class="{css_class}">
                    <span class="step-icon">{icon}</span>
                    <div>
                        <div class="step-name">{name}</div>
                        <div class="step-desc">{desc}</div>
                    </div>
                    <div class="step-status">{status_html}</div>
                </div>"""
            pipeline_placeholder.markdown(html, unsafe_allow_html=True)

        # Show initial state
        render_pipeline(0, [])

        # Run each step manually with live updates
        from agents import search_agent, reader_agent, writer_chain, critic_chain
        from nlp_analyzer import analyze_report
        from dl_sentiment import analyze_sentiment_dl
        from ml_scorer import predict_quality_score

        state = {"topic": topic}

        # Step 1: Search
        render_pipeline(0, [])
        search_result = search_agent.invoke({
            "messages": [("user", f"Search for recent relevant sources about: {topic}")]
        })
        search_output = search_result["messages"][-1].content
        state["search_results"] = search_output

        # Step 2: Reader
        render_pipeline(1, [])
        combined_context = search_output
        if uploaded_docs_context:
            combined_context += f"\n\nFrom uploaded documents:\n{uploaded_docs_context}"

        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Sources for '{topic}':\n\n{combined_context}\n\nExtract key facts."
            )]
        })
        reader_output = reader_result["messages"][-1].content
        state["research_notes"] = reader_output

        # Step 3: Writer
        
        report = writer_chain.invoke({
        "topic": topic,
        "research_notes": reader_output,
        "document_context": uploaded_docs_context if uploaded_docs_context else ""
        })
        state["report"] = report

        # Step 4: Critic
        render_pipeline(3, [])
        feedback = critic_chain.invoke({"report": report})
        state["critic_feedback"] = feedback

        # Step 5: NLP
        render_pipeline(4, [])
        nlp_results = analyze_report(report)
        state["nlp_results"] = nlp_results

        # Step 6: DL
        render_pipeline(5, [])
        dl_results = analyze_sentiment_dl(report)
        state["dl_results"] = dl_results

        # Step 7: ML
        render_pipeline(6, [])
        ml_results = predict_quality_score(report, nlp_results)
        state["ml_results"] = ml_results

        # All done
        render_pipeline(7, [])
        time.sleep(0.5)
        pipeline_placeholder.success("✅ All 7 pipeline steps completed successfully!")

        # -----------------------------------------------------
        # Metrics Row
        # -----------------------------------------------------
        st.markdown('<p class="section-header">📊 Analysis Summary</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⭐</div>
                <div class="metric-label">ML Quality Score</div>
                <div class="metric-value">{state['ml_results']['ml_quality_score']}</div>
                <div class="metric-sub">{state['ml_results']['ml_quality_label']}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-label">NLP Sentiment</div>
                <div class="metric-value" style="font-size:1.2rem">{state['nlp_results']['sentiment']}</div>
                <div class="metric-sub">Polarity: {state['nlp_results']['polarity_score']}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🤖</div>
                <div class="metric-label">BERT DL Sentiment</div>
                <div class="metric-value" style="font-size:1.2rem">{state['dl_results']['dl_sentiment']}</div>
                <div class="metric-sub">Confidence: {state['dl_results']['dl_confidence']}</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📝</div>
                <div class="metric-label">Report Stats</div>
                <div class="metric-value">{state['nlp_results']['word_count']}</div>
                <div class="metric-sub">{state['nlp_results']['sentence_count']} sentences</div>
            </div>""", unsafe_allow_html=True)

        # Keywords
        st.markdown('<p class="section-header">🔑 Top Keywords Extracted</p>', unsafe_allow_html=True)
        keywords_html = " ".join([
            f'<span class="keyword-badge">{kw}</span>'
            for kw in state['nlp_results']['keywords']
        ])
        st.markdown(keywords_html, unsafe_allow_html=True)

        # -----------------------------------------------------
        # Tabs
        # -----------------------------------------------------
        st.markdown('<p class="section-header">📋 Detailed Results</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Final Report",
            "🧐 Critic Feedback",
            "🔍 Search Results",
            "📖 Research Notes"
        ])

        with tab1:
            st.markdown(f'<div class="report-content">{state["report"]}</div>',
                       unsafe_allow_html=True)

        with tab2:
            st.markdown(f'<div class="report-content">{state["critic_feedback"]}</div>',
                       unsafe_allow_html=True)

        with tab3:
            st.markdown(f'<div class="report-content">{state["search_results"]}</div>',
                       unsafe_allow_html=True)

        with tab4:
            st.markdown(f'<div class="report-content">{state["research_notes"]}</div>',
                       unsafe_allow_html=True)
import streamlit as st
from pathlib import Path
import os

from src.config import BASE_DIR, DATA_DIR, GEMINI_API_KEY
from src.parsers.resume_parser import ResumeParser
from src.rag.vector_store import ChromaVectorStore
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.knowledge_base import KnowledgeBaseIngestor
import importlib
import sys

# Force live reload of service modules so edits apply immediately without restarting server
for mod_name in [
    "src.services.llm_client",
    "src.services.analyzer_service",
    "src.services.resume_enhancer",
    "src.services.interview_service",
    "src.services.roadmap_service"
]:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from src.services.llm_client import LLMClient
from src.services.analyzer_service import AnalyzerService
from src.services.resume_enhancer import ResumeEnhancer
from src.services.interview_service import InterviewService
from src.services.roadmap_service import RoadmapService

# Page Setup
st.set_page_config(
    page_title="AI Career Coach & Interview Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .tag-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px;
        font-weight: 500;
    }
    .tag-green { background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .tag-red { background-color: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State & Caching
@st.cache_resource(show_spinner="Initializing Local Vector Store & Knowledge Base...")
def init_rag_system():
    vector_store = ChromaVectorStore(collection_name="career_knowledge_base")
    ingestor = KnowledgeBaseIngestor(vector_store)
    ingestor.ingest_all(force_refresh=False)
    retriever = HybridRetriever(vector_store)
    return vector_store, ingestor, retriever

vector_store, ingestor, retriever = init_rag_system()

# Sidebar
with st.sidebar:
    st.header("⚙️ System Control")

    api_key_input = st.text_input(
        "Google Gemini API Key (Free)",
        value=st.session_state.get("gemini_key", GEMINI_API_KEY),
        type="password",
        help="Free API key from https://aistudio.google.com/ - NO credit card required. Leave empty for offline heuristic mode."
    )
    if api_key_input != st.session_state.get("gemini_key"):
        st.session_state["gemini_key"] = api_key_input

    model_option = st.selectbox(
        "LLM Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-1.5-flash is faster and has generous free-tier limits."
    )

    st.markdown("---")
    st.subheader("📚 RAG Knowledge Base")
    doc_count = vector_store.count()
    st.info(f"Indexed Knowledge Chunks: **{doc_count}**\n- Competency Taxonomies\n- Interview Questions\n- STAR Transformation Rules")

    if st.button("🔄 Re-index Knowledge Base"):
        with st.spinner("Re-indexing knowledge base..."):
            ingestor.ingest_all(force_refresh=True)
            retriever.reload()
            st.success("Knowledge base re-indexed successfully!")

    st.markdown("---")
    st.subheader("🚀 Quick Test Demo")
    if st.button("Load Sample Profile & Job"):
        sample_res_path = BASE_DIR / "samples" / "sample_resume.txt"
        sample_jd_path = BASE_DIR / "samples" / "sample_job_description.txt"
        if sample_res_path.exists() and sample_jd_path.exists():
            st.session_state["resume_text"] = sample_res_path.read_text(encoding="utf-8")
            st.session_state["jd_text"] = sample_jd_path.read_text(encoding="utf-8")
            # Clear previous results to guarantee fresh generation
            for k in ["analysis_result", "enhancement_result", "interview_result", "roadmap_result", "eval_output"]:
                st.session_state.pop(k, None)
            st.rerun()

    if st.button("🧹 Clear All Results"):
        for k in ["analysis_result", "enhancement_result", "interview_result", "roadmap_result", "eval_output"]:
            st.session_state.pop(k, None)
        st.rerun()

# Instantiate services with active key
active_key = st.session_state.get("gemini_key", GEMINI_API_KEY)
llm_client = LLMClient(api_key=active_key, model_name=model_option)
analyzer_service = AnalyzerService(retriever, llm_client)
resume_enhancer = ResumeEnhancer(retriever, llm_client)
interview_service = InterviewService(retriever, llm_client)
roadmap_service = RoadmapService(retriever, llm_client)

with st.sidebar:
    st.markdown("---")
    if llm_client.is_available:
        st.success("🟢 **LLM Active**: Google Gemini API Connected")
    else:
        st.info("🟡 **Mode**: Smart Offline Heuristic (Zero-Key Mode)")

# Header
st.title("🎯 AI Career Coach & Technical Interview Copilot")
st.caption("Production-grade Career Intelligence powered by **Hybrid RAG** (Dense Vector Search + BM25 Lexical + Reciprocal Rank Fusion)")

# Input Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    
    extracted_text = ""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            extracted_text = ResumeParser.extract_text_from_pdf(uploaded_file.read())
        else:
            extracted_text = ResumeParser.clean_text(uploaded_file.read().decode("utf-8", errors="ignore"))
        st.session_state["resume_text"] = extracted_text

    resume_input = st.text_area(
        "Resume Content",
        value=st.session_state.get("resume_text", ""),
        height=240,
        placeholder="Paste resume text or upload above..."
    )

with col2:
    st.subheader("💼 Target Job Description")
    target_role_name = st.selectbox(
        "Target Role Category",
        options=["AI Engineer", "Machine Learning Engineer", "Backend Engineer"],
        index=0
    )
    jd_input = st.text_area(
        "Job Description (JD)",
        value=st.session_state.get("jd_text", ""),
        height=240,
        placeholder="Paste target job description here..."
    )

# Execution Button
run_analysis = st.button("🚀 Analyze & Generate Full Coaching Plan", type="primary", use_container_width=True)

if run_analysis:
    if not resume_input.strip() or not jd_input.strip():
        st.warning("⚠️ Please provide both a Resume and a Job Description to proceed.")
    else:
        with st.spinner("Executing Hybrid RAG Retrieval & Multi-dimensional Gap Analysis..."):
            # 1. Gap Analysis
            analysis_result = analyzer_service.analyze(
                resume_text=resume_input,
                jd_text=jd_input,
                target_role=target_role_name
            )
            st.session_state["analysis_result"] = analysis_result

            # 2. Resume Enhancements
            enhancement_result = resume_enhancer.enhance_bullets(
                raw_bullets_or_experience=resume_input,
                target_jd=jd_input
            )
            st.session_state["enhancement_result"] = enhancement_result

            # 3. Targeted Interview Questions
            missing_skills = analysis_result["metrics"]["missing_keywords"]
            interview_result = interview_service.generate_interview_prep(
                missing_skills=missing_skills,
                target_role=target_role_name
            )
            st.session_state["interview_result"] = interview_result

            # 4. Learning Roadmap
            roadmap_result = roadmap_service.generate_roadmap(
                missing_skills=missing_skills,
                target_role=target_role_name,
                duration_weeks=4
            )
            st.session_state["roadmap_result"] = roadmap_result

        st.success("✅ Analysis & Coaching Pipeline completed successfully!")

# Display Results if Available
if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]
    metrics = res["metrics"]

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Fit & Gap Matrix",
        "✍️ STAR Resume Enhancer",
        "🎙️ Mock Interview Prep",
        "🗺️ 4-Week Career Roadmap"
    ])

    # TAB 1: Fit & Skill Gap Matrix
    with tab1:
        st.subheader("Candidate Fit Assessment")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{metrics['overall_match_score']}%</div>
                <div class="metric-label">Blended Role Fit Score</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{metrics['keyword_match_pct']}%</div>
                <div class="metric-label">Exact Technical Keyword Match (BM25)</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{metrics['semantic_match_pct']}%</div>
                <div class="metric-label">Semantic Competency Alignment (Vectors)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        kcol1, kcol2 = st.columns(2)
        with kcol1:
            st.markdown("##### ✅ Matched Verified Skills")
            if metrics["matched_keywords"]:
                html_matched = "".join([f"<span class='tag-badge tag-green'>{k}</span>" for k in metrics["matched_keywords"]])
                st.markdown(html_matched, unsafe_allow_html=True)
            else:
                st.write("None detected.")

        with kcol2:
            st.markdown("##### ⚠️ High-Priority Missing Gaps")
            if metrics["missing_keywords"]:
                html_missing = "".join([f"<span class='tag-badge tag-red'>{k}</span>" for k in metrics["missing_keywords"]])
                st.markdown(html_missing, unsafe_allow_html=True)
            else:
                st.write("No major technical gaps detected.")

        st.markdown("---")
        st.markdown("### 💡 AI Mentor Analysis")
        st.markdown(res["ai_insights"])

        with st.expander("🔍 View Retrieved Competency Ground Truth (RAG Context)"):
            for i, doc in enumerate(res["retrieved_context"], start=1):
                st.markdown(f"**Source Document {i}** (RRF Score: `{doc.get('rrf_score', 'N/A')}`):")
                st.code(doc["content"])

    # TAB 2: STAR Resume Enhancer
    with tab2:
        st.subheader("STAR-Method Resume Optimization")
        st.info("Transforms weak, task-based descriptions into quantified accomplishments using the Google XYZ Formula: *Accomplished [X] measured by [Y] by doing [Z]*.")
        enh_res = st.session_state["enhancement_result"]
        st.markdown(enh_res["enhanced_content"])

        with st.expander("🔍 View Retrieved STAR Impact Guidelines (RAG Context)"):
            for i, doc in enumerate(enh_res["retrieved_patterns"], start=1):
                st.markdown(f"**Pattern {i}**:")
                st.code(doc["content"])

    # TAB 3: Mock Interview Prep
    with tab3:
        st.subheader("Targeted Technical Interview Simulator")
        st.caption("Real interview questions retrieved from the knowledge base targeting your specific skill gaps.")
        int_res = st.session_state["interview_result"]
        st.markdown(int_res["interview_guide"])

        st.markdown("---")
        st.subheader("Interactive Answer Grader")
        sample_q = st.text_input(
            "Select or paste an interview question to practice:",
            value="Why use Hybrid Search (BM25 + Dense Vectors) instead of dense embeddings alone in a production RAG system?"
        )
        user_answer_text = st.text_area("Your Answer:", height=120, placeholder="Explain the concept, architecture, and trade-offs...")

        if st.button("Evaluate My Answer"):
            if user_answer_text.strip():
                with st.spinner("Senior AI Interviewer evaluating response..."):
                    eval_output = interview_service.evaluate_answer(sample_q, user_answer_text)
                    st.session_state["eval_output"] = eval_output
            else:
                st.warning("Please type an answer before evaluating.")

        if "eval_output" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["eval_output"])

    # TAB 4: Career Roadmap
    with tab4:
        st.subheader("Personalized 4-Week Career Transition Roadmap")
        st.caption("A step-by-step project-driven curriculum designed to bridge verified competency gaps.")
        road_res = st.session_state["roadmap_result"]
        st.markdown(road_res["roadmap_plan"])

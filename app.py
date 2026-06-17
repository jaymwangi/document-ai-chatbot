"""
RAG Chatbot - Streamlit UI (PURE UI/UX - NO BUSINESS LOGIC)

This file ONLY handles:
- UI layout and rendering
- User interactions
- Session state management
- Display of results

All business logic (retrieval, question generation, etc.) is in:
- pipeline/orchestrator.py (new modular pipeline)
- services/question_generator.py
"""

import streamlit as st
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

# Import business logic - USING NEW MODULAR PIPELINE
from pipeline.orchestrator import RAGOrchestrator as RAGPipeline
from services.question_generator import (
    generate_possible_questions, 
    generate_followup_questions,
    get_autocomplete_suggestions
)

# ============================================================
# TIMESTAMPED LOGGING CONFIGURATION (Nairobi Time)
# ============================================================

class TimestampFilter(logging.Filter):
    def filter(self, record):
        nairobi_tz = ZoneInfo("Africa/Nairobi")
        record.timestamp = datetime.now(nairobi_tz).strftime("%H:%M:%S.%f")[:-3]
        return True

logging.basicConfig(level=logging.INFO, format='%(timestamp)s | %(levelname)s | %(message)s')
for handler in logging.root.handlers:
    handler.addFilter(TimestampFilter())

logger = logging.getLogger(__name__)


# ============================================================
# FORCE MODEL LOAD AT APP STARTUP
# ============================================================

from services.embeddings import preload_embedder

# ⚡ Load the embedding model NOW (not during ingestion)
logger.info("🔧 Preloading embedder at app startup...")
preload_embedder()
logger.info("✅ Embedder preloaded")


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# CUSTOM CSS (UI ONLY)
# =========================

def apply_custom_css():
    """Apply custom CSS for better UX."""
    st.markdown("""
    <style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .source-card {
        background-color: #f0f2f6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #ff4b4b;
    }
    .score-badge {
        background-color: #e6f4ff;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .main .block-container { padding-bottom: 5rem; }
    .stButton button { transition: all 0.2s ease; }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)


# =========================
# SESSION STATE INITIALIZATION - SINGLE SOURCE OF TRUTH
# =========================

def init_session_state():
    """Initialize ALL session state variables. One place, one truth."""
    
    # ✅ ONLY session_state - NO cache_resource mixing
    if "pipeline" not in st.session_state:
        logger.info("🏗️ Creating pipeline instance (first and only time)...")
        st.session_state.pipeline = RAGPipeline(
            # ✅ NO embedding_model parameter - orchestrator uses singleton embedder
            llm_provider="groq",
            llm_model="llama-3.1-8b-instant",
            top_k=8,
            score_threshold=0.05,
            temperature=0.2,
            enable_query_guard=False,
            enable_reranking=False,
            persist_stores=True,
            use_hybrid_retrieval=True,
            bm25_k1=1.5,
            bm25_b=0.75,
            rrf_k=60,
            dense_weight=1.0,
            bm25_weight=1.0,
        )
        logger.info("✅ Pipeline instance created")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "is_ready" not in st.session_state:
        st.session_state.is_ready = False
    
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0
    
    if "auto_submit" not in st.session_state:
        st.session_state.auto_submit = None
    
    # ✅ Track processed files to prevent duplicate ingestion
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    # ✅ Track if we're currently processing (prevent race conditions)
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # Hybrid retriever status tracking
    if "hybrid_status" not in st.session_state:
        st.session_state.hybrid_status = {
            "is_ready": False,
            "is_building": False,
            "last_build_time": None,
            "document_count": 0
        }


# =========================
# PIPELINE ACCESS - SIMPLE, NO DUPLICATION
# =========================

def get_pipeline():
    """Simple getter - pipeline ALWAYS exists in session_state."""
    return st.session_state.pipeline

def clear_pipeline():
    """Clear pipeline and all related state."""
    logger.info("🗑️ Clearing pipeline and all state...")
    
    st.session_state.pipeline = None
    st.session_state.is_ready = False
    st.session_state.messages = []
    st.session_state.document_count = 0
    st.session_state.processed_files = set()
    st.session_state.is_processing = False
    st.session_state.hybrid_status = {
        "is_ready": False,
        "is_building": False,
        "last_build_time": None,
        "document_count": 0
    }
    
    logger.info("✅ Pipeline cleared")
    st.rerun()  # ✅ Force UI refresh


# =========================
# UI COMPONENTS
# =========================

def display_hybrid_status():
    """Display hybrid retriever status in UI."""
    pipeline = get_pipeline()
    if not pipeline or not pipeline.use_hybrid_retrieval:
        return
    
    status = st.session_state.hybrid_status
    
    if status["is_ready"] and not status.get("is_stale", False):
        status_text = "✅ Hybrid Ready"
    elif status.get("is_building", False):
        status_text = "🔄 Building BM25..."
    elif status.get("is_stale", False):
        status_text = "⏳ Will rebuild on next query"
    else:
        status_text = "⏳ Not built yet (first query)"
    
    st.caption(f"🔍 {status_text} | Documents: {status['document_count']}")


def display_sources(sources: List[Dict[str, Any]], max_sources: int = 3):
    """Display retrieved sources."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} relevant passages)", expanded=False):
        for i, source in enumerate(sources[:max_sources], 1):
            text = source.get('text', '')
            score = source.get('score', 0.0)
            source_name = source.get('source', 'Unknown')
            
            has_hybrid_scores = 'dense_score' in source and 'bm25_score' in source
            score_color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
            
            if has_hybrid_scores:
                dense = source.get('dense_score', 0.0)
                bm25 = source.get('bm25_score', 0.0)
                rrf = source.get('rrf_score', 0.0)
                score_display = f"RRF: {rrf:.3f} | Dense: {dense:.3f} | BM25: {bm25:.3f}"
            else:
                score_display = f"relevance: {score:.3f}"
            
            st.markdown(f"""
            <div class="source-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span><strong>Source {i}</strong> — {source_name}</span>
                    <span class="score-badge">{score_color} {score_display}</span>
                </div>
                <div style="font-size: 0.9rem; color: #555;">
                    {text[:300]}...
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_confidence_score(avg_score: float):
    """Display confidence score with color coding."""
    if avg_score > 0.7:
        color = "🟢"
        label = "High"
    elif avg_score > 0.4:
        color = "🟡"
        label = "Medium"
    else:
        color = "🔴"
        label = "Low"
    
    st.caption(f"{color} Confidence: {label} ({avg_score:.2f})")


def display_possible_questions(questions: List[str], max_to_show: int = 4):
    """Display possible questions as clickable buttons."""
    if not questions:
        return
    
    st.markdown("---")
    st.markdown("### 💡 Questions you can ask about this document")
    st.caption("Click any question to ask it")
    
    cols = st.columns(2)
    for i, question in enumerate(questions[:max_to_show]):
        col_idx = i % 2
        with cols[col_idx]:
            display_q = question[:70] + "..." if len(question) > 70 else question
            if st.button(f"❓ {display_q}", key=f"possible_q_{i}", use_container_width=True):
                st.session_state.auto_submit = question
                st.rerun()


def display_autocomplete_suggestions(suggestions: List[str]):
    """Display autocomplete suggestions as buttons."""
    if not suggestions:
        return
    
    st.markdown("### 💡 Suggestions while typing:")
    cols = st.columns(min(len(suggestions), 3))
    for i, suggestion in enumerate(suggestions[:3]):
        with cols[i]:
            display_s = suggestion[:60] + "..." if len(suggestion) > 60 else suggestion
            if st.button(f"📝 {display_s}", key=f"autocomplete_{i}", use_container_width=True):
                st.session_state.auto_submit = suggestion
                st.rerun()


# =========================
# PROCESS UPLOADED DOCUMENTS - ONCE PER FILE, NO DUPLICATION
# =========================
def process_uploaded_documents(files, llm_provider, llm_model, top_k, score_threshold, temperature, use_hybrid=True):
    """
    Process uploaded PDF files - ONLY when new files are uploaded.
    Includes pipeline existence check.
    """
    
    # ✅ Check if already processing
    if st.session_state.is_processing:
        st.warning("⏳ Already processing documents. Please wait...")
        return
    
    # ✅ Check if pipeline exists
    pipeline = get_pipeline()
    if pipeline is None:
        st.error("❌ Pipeline not initialized. Please restart the app.")
        logger.error("Pipeline is None when trying to process documents")
        return
    
    # Create unique file IDs
    file_ids = [f"{file.name}_{file.size}" for file in files]
    
    # ✅ Check if already processed
    if all(fid in st.session_state.processed_files for fid in file_ids):
        logger.info("📌 Files already processed, skipping")
        st.info("✅ Files already processed. Ask a question!")
        return
    
    logger.info(f"📌 Processing {len(files)} new files...")
    st.session_state.is_processing = True
    
    try:
        # ✅ Now safe to set attributes
        pipeline.top_k = top_k
        pipeline.score_threshold = score_threshold
        pipeline.use_hybrid_retrieval = use_hybrid
        
        # Save uploaded files
        temp_dir = Path(tempfile.mkdtemp())
        for file in files:
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        # ✅ Run ingestion with status
        with st.status("📄 Processing documents...", expanded=True) as status:
            status.update(label="Processing...", state="running")
            
            result = pipeline.ingest_documents(str(temp_dir))
            
            if result["success"]:
                status.update(label="✅ Complete!", state="complete")
            else:
                status.update(label=f"❌ Failed: {result.get('error', 'Unknown error')}", state="error")
                st.error(f"❌ Failed to process: {result.get('error', 'Unknown error')}")
                return
        
        if result["success"]:
            # ✅ Mark files as processed
            for fid in file_ids:
                st.session_state.processed_files.add(fid)
            
            st.session_state.is_ready = True
            st.session_state.document_count = result["documents"]
            
            st.balloons()
            
            hybrid_info = ""
            if use_hybrid:
                hybrid_info = "\n\n🔍 **Hybrid Search**: BM25 index will be built on your first query (0.5-2s)."
            
            st.success(f"✅ Processed {result['documents']} documents with {result['vectors']} vectors{hybrid_info}")
            
            welcome_msg = f"Hello! I've processed {result['documents']} document(s). Ask me anything!"
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg,
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
            })
            
            time.sleep(0.5)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        logger.error(f"Document processing error: {e}", exc_info=True)
    finally:
        st.session_state.is_processing = False


# =========================
# SIDEBAR
# =========================

def render_sidebar():
    """Render the sidebar with configuration and status."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Retrieval Type
        st.subheader("🔍 Retrieval Mode")
        retrieval_type = st.selectbox(
            "Search Method",
            options=["Hybrid (Dense + BM25)", "Dense Only"],
            help="Hybrid combines semantic (dense) and keyword (BM25) search."
        )
        use_hybrid = retrieval_type == "Hybrid (Dense + BM25)"
        
        # LLM Provider
        st.subheader("🤖 LLM Settings")
        llm_provider = st.selectbox(
            "Provider",
            options=["groq", "openai", "mock"],
            help="Groq has a free tier. OpenAI requires paid API key."
        )
        
        if llm_provider == "groq":
            llm_model = st.selectbox(
                "Model",
                options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
                help="8b is faster and free. 70b is more accurate but slower."
            )
            st.info("💡 Set GROQ_API_KEY in .env file")
        elif llm_provider == "openai":
            llm_model = st.selectbox(
                "Model",
                options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                help="gpt-4o-mini is cheapest and fast."
            )
            st.info("💡 Set OPENAI_API_KEY in .env file")
        else:
            llm_model = "mock-model"
            st.info("🔧 Mock mode - no API key required")
        
        # Vector Store Status
        st.subheader("🗄️ Vector Store")
        st.info("🔍 FAISS (production backend)")
        
        if use_hybrid:
            st.subheader("🔍 Hybrid Retriever")
            display_hybrid_status()
            st.caption("BM25 builds on first query (0.5-2s)")
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Chunks to retrieve", 3, 15, 8)
        score_threshold = st.slider("Relevance threshold", 0.0, 0.5, 0.05, 0.01)
        temperature = st.slider("LLM Creativity", 0.0, 1.0, 0.2, 0.05)
        
        st.divider()
        
        # Document Management
        st.subheader("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Process Documents", type="primary", use_container_width=True):
                    process_uploaded_documents(
                        uploaded_files, llm_provider, llm_model, 
                        top_k, score_threshold, temperature,
                        use_hybrid
                    )
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    clear_pipeline()
                    st.rerun()
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        
        pipeline = get_pipeline()
        if pipeline and st.session_state.is_ready:
            status = pipeline.get_status()
            st.success("✅ Pipeline Ready")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", status.get("documents", 0))
            with col2:
                st.metric("Chunks", status.get("chunks", 0))
            with col3:
                st.metric("Vectors", status.get("vectors", 0))
            
            if 'hybrid_retriever' in status:
                hybrid_status = status['hybrid_retriever']
                if hybrid_status.get('is_ready', False):
                    st.caption(f"🔍 BM25: ✅ Ready ({hybrid_status.get('document_count', 0)} docs)")
                elif hybrid_status.get('is_stale', True):
                    st.caption("🔍 BM25: ⏳ Stale (will rebuild)")
                else:
                    st.caption("🔍 BM25: ⏳ Not built yet")
            
            st.caption(f"🧠 Model: {status.get('generator_info', {}).get('model', 'N/A')}")
            
            with st.expander("📄 Document Preview", expanded=False):
                try:
                    texts = pipeline.vector_store.texts
                    if texts:
                        st.text_area("First chunk:", texts[0][:300] + "...", height=100)
                        st.caption(f"Total chunks: {len(texts)}")
                    else:
                        st.info("No document loaded")
                except Exception as e:
                    st.error(f"Preview error: {e}")
            
            if st.button("🗑️ Clear Documents", use_container_width=True):
                clear_pipeline()
                st.rerun()
        else:
            st.warning("⚠️ No documents loaded")
            st.info("Upload PDFs above to get started.")
        
        st.divider()
        
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **RAG Document Chatbot**
            
            **How it works:**
            1. Upload PDF documents
            2. System extracts text and creates embeddings
            3. Ask questions - system finds relevant chunks
            4. LLM generates answers from your documents
            
            **Built with:**
            - PDF Loader (PyPDF)
            - Sentence Transformers (all-MiniLM-L6-v2)
            - FAISS Vector Store
            - Groq/OpenAI LLM
            - BM25 + RRF Hybrid Search
            """)


# =========================
# MAIN CHAT INTERFACE
# =========================

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 RAG Document Chatbot")
    st.caption("Ask questions about your documents - answers are grounded in your uploaded content")
    
    pipeline = get_pipeline()
    
    if pipeline and pipeline.use_hybrid_retrieval:
        hybrid = pipeline.hybrid_retriever
        if hybrid and not hybrid.is_ready():
            st.info("🔍 **Hybrid Search Initializing**: BM25 index will build on your first query (0.5-2s).")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("timestamp"):
                    st.caption(f"🕐 {message['timestamp'].strftime('%I:%M %p')}")
                if message.get("sources") and message["role"] == "assistant":
                    with st.expander(f"📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"][:3], 1):
                            if 'dense_score' in source and 'bm25_score' in source:
                                score_str = f"RRF: {source.get('rrf_score', 0):.3f} | Dense: {source.get('dense_score', 0):.3f} | BM25: {source.get('bm25_score', 0):.3f}"
                            else:
                                score_str = f"score: {source['score']:.3f}"
                            st.markdown(f"**Source {i}** ({score_str})\n> {source['text'][:200]}...")
    
    if st.session_state.is_ready and len(st.session_state.messages) <= 1:
        with chat_container:
            possible_questions = generate_possible_questions(pipeline, max_questions=4)
            display_possible_questions(possible_questions, max_to_show=4)
    
    user_input = st.chat_input("Ask a question about your documents...", key="chat_input_main")
    
    auto_query = st.session_state.get("auto_submit")
    if auto_query:
        st.session_state.auto_submit = None
        user_input = auto_query
    
    if user_input:
        nairobi_tz = ZoneInfo("Africa/Nairobi")
        logger.info(f"🔍 USER QUERY: {user_input} | Time: {datetime.now(nairobi_tz).strftime('%H:%M:%S')}")
    
    if user_input and len(user_input) > 2 and st.session_state.is_ready:
        suggestions = get_autocomplete_suggestions(pipeline, user_input, max_suggestions=3)
        if suggestions:
            with chat_container:
                display_autocomplete_suggestions(suggestions)
    
    if user_input:
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
        })
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        with st.spinner("🤔 Thinking..."):
            if pipeline and st.session_state.is_ready:
                try:
                    if pipeline.use_hybrid_retrieval and pipeline.hybrid_retriever:
                        if pipeline.hybrid_retriever.is_stale():
                            st.info("🔍 Building hybrid search index for better results (0.5-2s)...")
                    
                    result = pipeline.ask_with_sources(user_input)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    avg_score = result.get("confidence", 0.0)
                    
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            display_confidence_score(avg_score)
                            
                            if pipeline.use_hybrid_retrieval:
                                st.caption("🔍 Hybrid (Dense + BM25) retrieval used")
                            
                            if sources:
                                display_sources(sources, max_sources=3)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
                        "sources": sources,
                    })
                    
                    followups = generate_followup_questions(pipeline, answer, user_input)
                    if followups:
                        with chat_container:
                            st.markdown("---")
                            st.markdown("### 💭 You might also ask:")
                            cols = st.columns(min(len(followups), 3))
                            for i, q in enumerate(followups[:3]):
                                with cols[i]:
                                    if st.button(f"❓ {q[:50]}...", key=f"followup_{i}", use_container_width=True):
                                        st.session_state.auto_submit = q
                                        st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
                    })
            else:
                error_msg = "⚠️ Please upload documents first"
                with chat_container:
                    with st.chat_message("assistant"):
                        st.warning(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
                })
        
        st.rerun()


# =========================
# MAIN
# =========================

def main():
    """Main entry point."""
    apply_custom_css()
    init_session_state()
    render_sidebar()
    
    st.divider()
    
    render_chat_interface()


if __name__ == "__main__":
    main()
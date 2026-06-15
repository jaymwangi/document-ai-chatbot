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
import random
import threading
from datetime import datetime
import logging

# Import business logic - USING NEW MODULAR PIPELINE
from pipeline.orchestrator import RAGOrchestrator as RAGPipeline
from services.question_generator import (
    generate_possible_questions, 
    generate_followup_questions,
    get_autocomplete_suggestions
)

# ============================================================
# FUN FACTS DATABASE (shown during long embedding processes)
# ============================================================

FUN_FACTS = [
    "💡 Did you know? The first computer virus was created in 1983 as a security experiment.",
    "🚀 Did you know? The first search engine 'Archie' was created in 1990, before the World Wide Web!",
    "🧠 Did you know? RAG (Retrieval-Augmented Generation) was introduced by Meta AI in 2020.",
    "⚡ Did you know? FAISS (Facebook AI Similarity Search) can search billions of vectors in milliseconds!",
    "📚 Did you know? The first hard drive, IBM 350, weighed over a ton and stored only 3.75MB.",
    "🔍 Did you know? Google handles over 8.5 billion searches per day - about 99,000 per second!",
    "🤖 Did you know? The term 'Machine Learning' was coined by Arthur Samuel in 1959.",
    "💾 Did you know? The first SSD was introduced in 1991 with 20MB capacity and cost $1,000!",
    "🌐 Did you know? The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
    "🐍 Did you know? Python is named after Monty Python, not the snake!",
    "📊 Did you know? The first 1GB hard drive (IBM 3380) was released in 1980 and cost $40,000!",
    "🎮 Did you know? The first video game 'Pong' was released in 1972 by Atari.",
    "📱 Did you know? The first iPhone was released in 2007 with 4GB storage and no App Store.",
    "☁️ Did you know? The first cloud storage service (iDrive) was launched in 1995!",
    "🔐 Did you know? The first ransomware attack (AIDS Trojan) occurred in 1989, delivered via floppy disks.",
]

# ============================================================
# TIMESTAMPED LOGGING CONFIGURATION
# ============================================================

class TimestampFilter(logging.Filter):
    def filter(self, record):
        record.timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(timestamp)s | %(levelname)s | %(message)s')
for handler in logging.root.handlers:
    handler.addFilter(TimestampFilter())

logger = logging.getLogger(__name__)


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
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
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
    
    .debug-card {
        background-color: #2d2d2d;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #00ff00;
        font-family: monospace;
        font-size: 0.8rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.3s ease-in;
    }
    
    /* Fix for chat input positioning */
    .main .block-container {
        padding-bottom: 5rem;
    }
    
    /* Question button styling */
    .stButton button {
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# =========================
# SMART SPINNER UTILITY (UI ONLY)
# =========================

def smart_status(task_func, label: str, delay: float = 0.5):
    """Only shows loading UI if task takes longer than `delay` seconds."""
    start = time.time()
    result = task_func()
    elapsed = time.time() - start
    
    if elapsed > delay:
        with st.status(f"{label}...", expanded=False) as status:
            status.update(label=f"{label}...", state="running")
            time.sleep(0.1)
            status.update(label=f"✅ {label} complete!", state="complete")
    
    return result


# =========================
# SESSION STATE INITIALIZATION (UI ONLY)
# =========================

def init_session_state():
    """Initialize all session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "is_ready" not in st.session_state:
        st.session_state.is_ready = False
    
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0
    
    if "auto_submit" not in st.session_state:
        st.session_state.auto_submit = None
    
    if "fun_fact_index" not in st.session_state:
        st.session_state.fun_fact_index = 0
    
    # NEW: Store last retrieval results for debug panel
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = {
            "query": "",
            "chunks": [],
            "scores": [],
            "sources": [],
            "timestamp": None
        }


# =========================
# PIPELINE MANAGER (UI ONLY)
# =========================

@st.cache_resource
def get_pipeline_instance():
    """
    Create and cache pipeline instance using FAISS (production backend).
    """
    pipeline = RAGPipeline(
        embedding_model="mini-lm",
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        top_k=8,
        score_threshold=0.05,
        temperature=0.2,
        enable_query_guard=False,   # OFF for speed
        enable_reranking=False,      # OFF for speed
        persist_stores=True,
    )
    
    # Pre-warm the embedder
    try:
        pipeline.embedder.embed_single("warmup")
    except Exception as e:
        logger.warning(f"Pre-warm failed: {e}")
    
    return pipeline


def ensure_pipeline_ready():
    """Ensure pipeline exists and is ready."""
    if st.session_state.pipeline is None:
        with st.spinner("🚀 Initializing AI engine..."):
            st.session_state.pipeline = get_pipeline_instance()


# =========================
# UI COMPONENTS
# =========================

def display_sources(sources: List[Dict[str, Any]], max_sources: int = 3):
    """Display retrieved sources (UI only)."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} relevant passages)", expanded=False):
        for i, source in enumerate(sources[:max_sources], 1):
            # Simple access - schema guaranteed by orchestrator
            text = source.get('text', '')
            score = source.get('score', 0.0)
            source_name = source.get('source', 'Unknown')
            
            score_color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
            
            st.markdown(f"""
            <div class="source-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span><strong>Source {i}</strong> — {source_name}</span>
                    <span class="score-badge">{score_color} relevance: {score:.3f}</span>
                </div>
                <div style="font-size: 0.9rem; color: #555;">
                    {text[:300]}...
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_retrieval_debug_panel():
    """
    NEW: Display retrieved chunks with scores for debugging.
    Shows exactly what the system found for the last query.
    """
    last = st.session_state.last_retrieval
    
    if not last["query"] or not last["chunks"]:
        return
    
    with st.expander("🔍 RETRIEVAL DEBUG PANEL (shows what the system found)", expanded=False):
        st.markdown(f"**Query:** `{last['query'][:100]}`")
        st.markdown(f"**Timestamp:** {last['timestamp'].strftime('%H:%M:%S') if last['timestamp'] else 'N/A'}")
        st.markdown(f"**Chunks retrieved:** {len(last['chunks'])}")
        st.divider()
        
        # Show top 5 chunks with scores
        for i, (chunk, score, source) in enumerate(zip(last["chunks"][:5], last["scores"][:5], last["sources"][:5]), 1):
            score_color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
            
            st.markdown(f"""
            <div class="debug-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span><strong>Chunk #{i}</strong> — {source}</span>
                    <span class="score-badge">{score_color} Score: {score:.4f}</span>
                </div>
                <div style="font-size: 0.8rem; color: #ccc; white-space: pre-wrap;">
                    {chunk[:400]}{'...' if len(chunk) > 400 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if len(last["chunks"]) > 5:
            st.caption(f"... and {len(last['chunks']) - 5} more chunks (see logs for full details)")


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
    """Display possible questions as clickable buttons below chat area."""
    if not questions:
        return
    
    st.markdown("---")
    st.markdown("### 💡 Questions you can ask about this document")
    st.caption("Click any question to ask it")
    
    # Display as 2x2 grid
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
# SIDEBAR (UI ONLY)
# =========================

def render_sidebar():
    """Render the sidebar with configuration and status."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # LLM Provider Selection
        st.subheader("🤖 LLM Settings")
        llm_provider = st.selectbox(
            "Provider",
            options=["groq", "openai", "mock"],
            help="Groq has a free tier (recommended for testing). OpenAI requires paid API key. Mock mode for testing without API calls."
        )
        
        if llm_provider == "groq":
            llm_model = st.selectbox(
                "Model",
                options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
                help="llama-3.1-8b-instant is faster and free. 70b model is more accurate but slower."
            )
            st.info("💡 Set GROQ_API_KEY in .env file")
        elif llm_provider == "openai":
            llm_model = st.selectbox(
                "Model",
                options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                help="gpt-4o-mini is cheapest and fast. gpt-4o is most capable but expensive."
            )
            st.info("💡 Set OPENAI_API_KEY in .env file")
        else:
            llm_model = "mock-model"
            st.info("🔧 Mock mode - no API key required")
        
        # FAISS Status (read-only, not a toggle)
        st.subheader("🗄️ Vector Store")
        st.info("🔍 FAISS (production backend) - high-performance vector search")
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Chunks to retrieve", 
            3, 15, 8, 
            help="Higher = more context but slower and more expensive. Lower = faster but may miss information. Recommended: 5-8 for most use cases."
        )
        score_threshold = st.slider(
            "Relevance threshold", 
            0.0, 0.5, 0.05, 0.01, 
            help="Lower (0.0-0.1) = more results (may include irrelevant). Higher (0.3-0.5) = fewer but more accurate results. Recommended: 0.05-0.1 for broad search."
        )
        temperature = st.slider(
            "LLM Creativity", 
            0.0, 1.0, 0.2, 0.05,
            help="Lower (0.0-0.3) = more factual/consistent. Higher (0.7-1.0) = more creative but may hallucinate. Recommended: 0.1-0.3 for factual Q&A."
        )
        
        st.divider()
        
        # Document Management
        st.subheader("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files. First-time processing may take 5-10 minutes for large books. Subsequent uploads will be cached and faster."
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Process Documents", type="primary", use_container_width=True):
                    process_uploaded_documents(
                        uploaded_files, llm_provider, llm_model, 
                        top_k, score_threshold, temperature
                    )
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.pipeline = None
                    st.session_state.is_ready = False
                    st.session_state.messages = []
                    st.session_state.document_count = 0
                    st.session_state.last_retrieval = {
                        "query": "", "chunks": [], "scores": [], "sources": [], "timestamp": None
                    }
                    st.cache_resource.clear()
                    st.rerun()
        
        st.divider()
        
        # =========================================================
        # NEW: RETRIEVAL DEBUG PANEL IN SIDEBAR
        # =========================================================
        display_retrieval_debug_panel()
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        
        if st.session_state.pipeline and st.session_state.is_ready:
            status = st.session_state.pipeline.get_status()
            st.success("✅ Pipeline Ready")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", status.get("documents", 0))
            with col2:
                st.metric("Chunks", status.get("chunks", 0))
            with col3:
                st.metric("Vectors", status.get("vectors", 0))
            
            # Show FAISS index type
            faiss_type = status.get('config', {}).get('faiss_index_type', 'N/A')
            st.caption(f"🗄️ FAISS Index: {faiss_type}")
            st.caption(f"🧠 Model: {status.get('generator_info', {}).get('model', 'N/A')}")
            st.caption(f"⚡ Reranking: {'ON' if status.get('config', {}).get('reranking') else 'OFF'}")
            st.caption(f"🛡️ Query Guard: {'ON' if status.get('config', {}).get('query_guard') else 'OFF'}")
            
            # Document Preview (UI only)
            with st.expander("📄 Document Preview", expanded=False):
                try:
                    texts = st.session_state.pipeline.vector_store.texts
                    if texts:
                        st.text_area("First chunk:", texts[0][:300] + "...", height=100)
                        st.caption(f"Total chunks: {len(texts)}")
                        st.caption(f"First time processing large docs may take 5-10 minutes. Subsequent loads are instant due to caching.")
                    else:
                        st.info("No document loaded")
                except Exception as e:
                    st.error(f"Preview error: {e}")
            
            if st.button("🗑️ Clear Documents", use_container_width=True):
                st.session_state.pipeline.clear()
                st.session_state.is_ready = False
                st.session_state.messages = []
                st.session_state.last_retrieval = {
                    "query": "", "chunks": [], "scores": [], "sources": [], "timestamp": None
                }
                st.rerun()
        else:
            st.warning("⚠️ No documents loaded")
            st.info("Upload PDFs above to get started. First-time processing of large books may take 5-10 minutes.")
        
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
            - FAISS Vector Store (fast similarity search)
            - Groq/OpenAI LLM
            
            **Tips:**
            - First large document takes 5-10 min (embedding)
            - Subsequent queries are fast (cached)
            - Use specific questions for best results
            """)


# =========================
# PROCESS UPLOADED DOCUMENTS (UI ONLY)
# =========================

def process_uploaded_documents(files, llm_provider, llm_model, top_k, score_threshold, temperature):
    """Process uploaded PDF files (UI wrapper) with rotating fun facts and better progress."""
    
    # Create progress indicators
    progress_bar = st.progress(0, text="Starting document processing...")
    status_text = st.empty()
    fun_fact_text = st.empty()
    
    # Flag to control fun fact rotation
    stop_rotation = False
    
    def rotate_fun_facts():
        """Rotate fun facts every 12 seconds during processing."""
        fact_index = 0
        while not stop_rotation:
            current_fact = FUN_FACTS[fact_index % len(FUN_FACTS)]
            fun_fact_text.info(f"✨ {current_fact}")
            fact_index += 1
            time.sleep(12)  # Change fact every 12 seconds
    
    def ingestion_task():
        """Actual ingestion logic wrapped for safety."""
        nonlocal stop_rotation
        
        if st.session_state.pipeline is None:
            st.session_state.pipeline = get_pipeline_instance()
        
        # Update runtime config
        st.session_state.pipeline.top_k = top_k
        st.session_state.pipeline.score_threshold = score_threshold
        
        # Save uploaded files with progress
        temp_dir = Path(tempfile.mkdtemp())
        for i, file in enumerate(files):
            status_text.text(f"📄 Saving: {file.name}")
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            progress_bar.progress((i + 1) / len(files), text=f"Saved {file.name}")
        
        # Start fun facts rotation in background
        fact_thread = threading.Thread(target=rotate_fun_facts, daemon=True)
        fact_thread.start()
        
        # Run ingestion with better progress messaging
        status_text.text("🔄 Generating embeddings (this may take 5-10 minutes for large PDFs)...")
        progress_bar.progress(0.3, text="Embedding in progress - check logs for detailed progress...")
        
        # Optional: Set up progress callback if pipeline supports it
        if hasattr(st.session_state.pipeline, 'set_progress_callback'):
            def progress_callback(current, total, eta):
                progress = 0.3 + (0.6 * (current / total))  # Scale from 30% to 90%
                progress_bar.progress(progress, text=f"Processing chunk {current}/{total} (ETA: {eta:.1f}s)")
            
            st.session_state.pipeline.set_progress_callback(progress_callback)
        
        result = st.session_state.pipeline.ingest_documents(str(temp_dir))
        
        # Stop fun facts rotation
        stop_rotation = True
        time.sleep(0.2)  # Give thread time to exit
        
        return result, temp_dir
    
    try:
        # Run ingestion with smart status handling
        result, temp_dir = smart_status(ingestion_task, "Processing documents", delay=0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        fun_fact_text.empty()
        
        if result["success"]:
            st.session_state.is_ready = True
            st.session_state.document_count = result["documents"]
            
            st.balloons()
            st.success(f"✅ Processed {result['documents']} documents with {result['vectors']} vectors")
            
            welcome_msg = f"Hello! I've processed {result['documents']} document(s). Ask me anything about your documents!"
            
            st.session_state.messages = []  # Clear any old messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg,
                "timestamp": datetime.now(),
            })
            
            time.sleep(0.5)
            st.rerun()
        else:
            st.error(f"❌ Failed to process: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        # Ensure rotation stops on error
        stop_rotation = True
        progress_bar.empty()
        status_text.empty()
        fun_fact_text.empty()
        st.error(f"❌ Processing failed: {str(e)}")
        logger.error(f"Document processing error: {e}", exc_info=True)


# =========================
# MAIN CHAT INTERFACE (UI ONLY)
# =========================

def render_chat_interface():
    """Render the main chat interface with proper layout."""
    st.title("🤖 RAG Document Chatbot")
    st.caption("Ask questions about your documents - answers are grounded in your uploaded content")
    
    ensure_pipeline_ready()
    
    # Create chat container FIRST (critical for layout)
    chat_container = st.container()
    
    # Display existing messages in chat container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("timestamp"):
                    st.caption(f"🕐 {message['timestamp'].strftime('%I:%M %p')}")
                if message.get("sources") and message["role"] == "assistant":
                    with st.expander(f"📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"][:3], 1):
                            st.markdown(f"**Source {i}** (score: {source['score']:.3f})\n> {source['text'][:200]}...")
    
    # =========================================================
    # POSSIBLE QUESTIONS SECTION - Below chat, above input
    # =========================================================
    if st.session_state.is_ready and len(st.session_state.messages) <= 1:
        with chat_container:
            possible_questions = generate_possible_questions(st.session_state.pipeline, max_questions=4)
            display_possible_questions(possible_questions, max_to_show=4)
    
    # =========================================================
    # CHAT INPUT (at bottom)
    # =========================================================
    user_input = st.chat_input("Ask a question about your documents...", key="chat_input_main")
    
    # Auto-submit handling
    auto_query = st.session_state.get("auto_submit")
    if auto_query:
        st.session_state.auto_submit = None
        user_input = auto_query
    
    # 🔥 LOG USER QUERY TO TERMINAL
    if user_input:
        logger.info(f"🔍 USER QUERY: {user_input}")
    
    # =========================================================
    # AUTOCOMPLETE SUGGESTIONS (while typing)
    # =========================================================
    if user_input and len(user_input) > 2 and st.session_state.is_ready:
        suggestions = get_autocomplete_suggestions(st.session_state.pipeline, user_input, max_suggestions=3)
        if suggestions:
            with chat_container:
                display_autocomplete_suggestions(suggestions)
    
    # =========================================================
    # PROCESS QUERY
    # =========================================================
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now(),
        })
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            if st.session_state.pipeline and st.session_state.is_ready:
                try:
                    # Perform retrieval and get result
                    result = st.session_state.pipeline.ask_with_sources(user_input)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    
                    # =========================================================
                    # NEW: STORE RETRIEVAL RESULTS FOR DEBUG PANEL
                    # =========================================================
                    if sources:
                        chunks = [s.get("text", "") for s in sources]
                        scores = [s.get("score", 0.0) for s in sources]
                        source_names = [s.get("source", "Unknown") for s in sources]
                        
                        st.session_state.last_retrieval = {
                            "query": user_input,
                            "chunks": chunks,
                            "scores": scores,
                            "sources": source_names,
                            "timestamp": datetime.now()
                        }
                        logger.info(f"📊 DEBUG: Stored {len(chunks)} chunks for debug panel")
                    
                    # Calculate confidence
                    avg_score = result.get("confidence", 0.0)
                    
                    # Display assistant message
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            display_confidence_score(avg_score)
                            if sources:
                                display_sources(sources, max_sources=3)
                    
                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now(),
                        "sources": sources,
                    })
                    
                    # Show follow-up questions after answer
                    followups = generate_followup_questions(st.session_state.pipeline, answer, user_input)
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
                        "timestamp": datetime.now(),
                    })
            else:
                error_msg = "⚠️ Please upload documents first"
                with chat_container:
                    with st.chat_message("assistant"):
                        st.warning(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now(),
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
"""
RAG Chatbot - Streamlit UI (Balanced Architecture)

A clean implementation that balances:
- Robust queue-based processing with automatic retries
- Clean, maintainable code with minimal state
- Good UX with clear status indicators
- Simple 429 handling without complex countdown UIs

Design:
- Single-flight lock (one request at a time)
- FIFO queue with automatic retry on 429
- Simple rate limit handling (3s backoff)
- Clean separation of concerns
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import uuid
import hashlib

# Import business logic
from pipeline.orchestrator import RAGOrchestrator as RAGPipeline
from services.question_generator import (
    generate_possible_questions, 
    get_question_tracker,
)

# Import AppState
from app_state import get_app_state, AppState

# ============================================================
# LOGGING
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

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS
# =========================

def apply_custom_css():
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
    .stButton button:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }
    .answer-separator {
        margin: 20px 0 10px 0;
        border-top: 1px solid #e0e0e0;
        padding-top: 10px;
        font-size: 0.85rem;
        color: #666;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .status-processing {
        background: #fff3e0;
        color: #e65100;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .status-queued {
        background: #e3f2fd;
        color: #0d47a1;
    }
    .status-rate-limited {
        background: #fff3cd;
        color: #856404;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

def init_session_state():
    """Initialize all session state variables."""
    state = get_app_state()
    
    # Pipeline
    if state.pipeline is None:
        logger.info("🏗️ Creating pipeline...")
        state.pipeline = RAGPipeline(
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
        logger.info("✅ Pipeline created")
    
    # Embedder
    if "embedder_loaded" not in st.session_state:
        logger.info("🔧 Loading embedder...")
        from services.embeddings import preload_embedder
        preload_embedder()
        st.session_state.embedder_loaded = True
        logger.info("✅ Embedder loaded")
    
    # === QUEUE STATE ===
    if "request_queue" not in st.session_state:
        st.session_state.request_queue = []
    
    # === PROCESSING STATE ===
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # === RATE LIMIT ===
    if "next_allowed_time" not in st.session_state:
        st.session_state.next_allowed_time = 0
    if "is_rate_limited" not in st.session_state:
        st.session_state.is_rate_limited = False
    
    # === TRIGGER ===
    if "trigger_process" not in st.session_state:
        st.session_state.trigger_process = False
    
    if "initial_suggestions_shown" not in st.session_state:
        st.session_state.initial_suggestions_shown = False
    
    if "hybrid_status" not in st.session_state:
        st.session_state.hybrid_status = {
            "is_building": False,
            "last_build_time": None,
            "document_count": 0
        }

# =========================
# HELPERS
# =========================

def get_pipeline():
    return get_app_state().pipeline

def clear_pipeline():
    logger.info("🗑️ Clearing pipeline...")
    state = get_app_state()
    state.reset()
    st.session_state.initial_suggestions_shown = False
    st.session_state.request_queue = []
    st.session_state.is_processing = False
    st.session_state.next_allowed_time = 0
    st.session_state.is_rate_limited = False
    st.session_state.trigger_process = False
    st.session_state.hybrid_status = {"is_building": False, "last_build_time": None, "document_count": 0}
    logger.info("✅ Pipeline cleared")
    st.rerun()

def make_stable_hash(text: str, length: int = 6) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]

# =========================
# QUEUE PROCESSOR
# =========================

def process_queue():
    """
    Process the queue with single-flight lock and automatic retry.
    """
    state = get_app_state()
    pipeline = state.pipeline
    
    # Single-flight lock
    if st.session_state.get("is_processing", False):
        return
    
    if not st.session_state.request_queue:
        return
    
    # Get first request
    request = st.session_state.request_queue[0]
    user_input = request["text"]
    
    # Rate limit check
    now = time.time()
    if now < st.session_state.next_allowed_time:
        wait_time = st.session_state.next_allowed_time - now
        st.session_state.is_rate_limited = True
        logger.info(f"⏳ Rate limited: {wait_time:.1f}s remaining")
        # Trigger a rerun when rate limit expires
        st.session_state.trigger_process = True
        return
    
    # Clear rate limit state
    st.session_state.is_rate_limited = False
    
    # Acquire lock
    st.session_state.is_processing = True
    
    logger.info(f"📌 Processing: {user_input[:50]}...")
    
    try:
        # Guard: no pipeline
        if not pipeline or not state.is_ready:
            state.messages.append({
                "role": "assistant",
                "content": "⚠️ Please upload documents first",
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
            })
            st.session_state.request_queue.pop(0)
            st.session_state.is_processing = False
            return
        
        # Build hybrid if needed (first query)
        if pipeline.use_hybrid_retrieval and pipeline.hybrid_retriever:
            if not pipeline.hybrid_retriever.is_ready():
                logger.info("🔍 Building BM25 index...")
                st.session_state.hybrid_status["is_building"] = True
                pipeline.hybrid_retriever.rebuild_index()
                st.session_state.hybrid_status["is_building"] = False
                st.session_state.hybrid_status["last_build_time"] = time.time()
                st.session_state.hybrid_status["document_count"] = len(pipeline.vector_store.texts)
        
        # Get answer (blocking)
        start_time = time.time()
        result = pipeline.ask_with_sources(user_input)
        response_time = time.time() - start_time
        
        # Update rate gate
        st.session_state.next_allowed_time = time.time() + 0.5
        
        # Remove from queue
        st.session_state.request_queue.pop(0)
        
        # Add to chat
        new_messages = [
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
            },
            {
                "role": "assistant",
                "content": result["answer"],
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "response_time": response_time,
            }
        ]
        state.messages.extend(new_messages)
        
        # Track used questions
        if state.current_doc_id:
            tracker = get_question_tracker()
            tracker.mark_used(state.current_doc_id, user_input)
        
        logger.info(f"✅ Done in {response_time:.2f}s")
        
    except Exception as e:
        error_str = str(e)
        logger.error(f"Error: {e}", exc_info=True)
        
        # Check for rate limit
        if "429" in error_str or "too many requests" in error_str.lower():
            # Keep in queue, set rate limit
            backoff = 3.0
            st.session_state.next_allowed_time = time.time() + backoff
            st.session_state.is_rate_limited = True
            logger.info(f"⏳ 429 received, backing off {backoff}s")
            
            # Show rate limit message
            state.messages.append({
                "role": "assistant",
                "content": f"⏳ Rate limit reached. Auto-retrying in {backoff:.0f}s...",
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
            })
        else:
            # Remove from queue (non-retryable)
            state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}",
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
            })
            st.session_state.request_queue.pop(0)
    
    finally:
        st.session_state.is_processing = False
        # Trigger next if queue not empty
        if st.session_state.request_queue:
            st.session_state.trigger_process = True

# =========================
# UI COMPONENTS
# =========================

def display_status():
    """Display current processing status."""
    if st.session_state.get("is_processing", False):
        st.markdown(
            '<span class="status-badge status-processing">🔄 Processing...</span>',
            unsafe_allow_html=True
        )
    elif st.session_state.get("is_rate_limited", False):
        remaining = max(0, st.session_state.next_allowed_time - time.time())
        st.markdown(
            f'<span class="status-badge status-rate-limited">⏳ Rate limited... {remaining:.1f}s</span>',
            unsafe_allow_html=True
        )
    elif st.session_state.request_queue:
        queue_size = len(st.session_state.request_queue)
        st.markdown(
            f'<span class="status-badge status-queued">📋 {queue_size} in queue</span>',
            unsafe_allow_html=True
        )

def display_sources(sources: List[Dict[str, Any]], max_sources: int = 3):
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} relevant passages)", expanded=False):
        for i, source in enumerate(sources[:max_sources], 1):
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

def render_suggestions(state: AppState, max_to_show: int = 4):
    doc_id = state.current_doc_id
    if not doc_id or not state.is_ready:
        return
    
    if st.session_state.get("is_processing", False):
        return
    
    tracker = get_question_tracker()
    remaining = tracker.get_remaining_questions(doc_id)
    summary = tracker.get_summary(doc_id)
    
    if summary["total"] == 0:
        pipeline = state.pipeline
        if pipeline:
            questions = generate_possible_questions(pipeline, max_questions=8, doc_id=doc_id)
            if questions:
                tracker.set_questions(doc_id, questions)
                remaining = tracker.get_remaining_questions(doc_id)
                summary = tracker.get_summary(doc_id)
    
    if summary["total"] == 0 or not remaining:
        return
    
    st.markdown('<div class="answer-separator">📌 Suggested Next Questions</div>', unsafe_allow_html=True)
    st.caption(f"💡 {len(remaining)} questions remaining")
    
    cols = st.columns(2)
    for i, question in enumerate(remaining[:max_to_show]):
        with cols[i % 2]:
            display_q = question[:70] + "..." if len(question) > 70 else question
            stable_key = make_stable_hash(question, 6)
            button_key = f"suggested_{doc_id}_{i}_{stable_key}"
            
            if st.button(
                f"❓ {display_q}",
                key=button_key,
                disabled=st.session_state.get("is_processing", False),
                use_container_width=True
            ):
                # Enqueue
                if not any(r["text"] == question for r in st.session_state.request_queue):
                    request_id = str(uuid.uuid4())
                    st.session_state.request_queue.append({"id": request_id, "text": question})
                    st.session_state.trigger_process = True
                    st.rerun()

# =========================
# SIDEBAR
# =========================

def render_sidebar():
    state = get_app_state()
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.subheader("🔍 Retrieval Mode")
        retrieval_type = st.selectbox(
            "Search Method",
            options=["Hybrid (Dense + BM25)", "Dense Only"]
        )
        use_hybrid = retrieval_type == "Hybrid (Dense + BM25)"
        
        st.subheader("🤖 LLM Settings")
        llm_provider = st.selectbox("Provider", options=["groq", "openai", "mock"])
        
        if llm_provider == "groq":
            llm_model = st.selectbox("Model", options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
            st.info("💡 Set GROQ_API_KEY in .env file")
        elif llm_provider == "openai":
            llm_model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
            st.info("💡 Set OPENAI_API_KEY in .env file")
        else:
            llm_model = "mock-model"
            st.info("🔧 Mock mode - no API key required")
        
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Chunks to retrieve", 3, 15, 8)
        score_threshold = st.slider("Relevance threshold", 0.0, 0.5, 0.05, 0.01)
        temperature = st.slider("LLM Creativity", 0.0, 1.0, 0.2, 0.05)
        
        st.divider()
        
        st.subheader("📄 Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                disabled = state.is_processing
                if st.button("📥 Process Documents", type="primary", use_container_width=True, disabled=disabled):
                    process_uploaded_documents(
                        uploaded_files, llm_provider, llm_model, 
                        top_k, score_threshold, temperature, use_hybrid
                    )
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    clear_pipeline()
        
        st.divider()
        
        st.subheader("📊 Status")
        pipeline = state.pipeline
        if pipeline and state.is_ready:
            status = pipeline.get_status()
            st.success("✅ Ready")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Docs", status.get("documents", 0))
            with col2:
                st.metric("Chunks", status.get("chunks", 0))
            with col3:
                st.metric("Vectors", status.get("vectors", 0))
            
            # Queue status
            queue_size = len(st.session_state.request_queue)
            if queue_size > 0:
                if st.session_state.get("is_processing", False):
                    st.info(f"🔄 Processing... ({queue_size} in queue)")
                elif st.session_state.get("is_rate_limited", False):
                    remaining = max(0, st.session_state.next_allowed_time - time.time())
                    st.warning(f"⏳ Rate limited ({remaining:.1f}s)... ({queue_size} in queue)")
                else:
                    st.info(f"📋 {queue_size} in queue")
            
            if st.button("🗑️ Clear", use_container_width=True):
                clear_pipeline()
        else:
            if state.is_processing:
                st.warning("🔄 Processing documents...")
            else:
                st.warning("⚠️ No documents loaded")

# =========================
# PROCESS UPLOAD
# =========================

def process_uploaded_documents(files, llm_provider, llm_model, top_k, score_threshold, temperature, use_hybrid=True):
    state = get_app_state()
    
    if state.is_processing:
        st.warning("⏳ Already processing. Please wait...")
        return
    
    pipeline = state.pipeline
    if pipeline is None:
        st.error("❌ Pipeline not initialized.")
        return
    
    file_ids = [f"{file.name}_{file.size}" for file in files]
    if all(state.is_file_processed(fid) for fid in file_ids):
        st.info("✅ Already processed.")
        return
    
    logger.info(f"📌 Processing {len(files)} files...")
    
    try:
        pipeline.top_k = top_k
        pipeline.score_threshold = score_threshold
        pipeline.use_hybrid_retrieval = use_hybrid
        
        state.start_processing()
        
        temp_dir = Path(tempfile.mkdtemp())
        for file in files:
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        state.update_processing_status(25.0, "chunking")
        state.update_processing_status(30.0, "embedding")
        
        def progress_callback(batch, total, eta):
            progress = 30 + (50 * (batch / total))
            state.update_processing_status(progress, "embedding")
        
        if hasattr(pipeline, 'set_progress_callback'):
            pipeline.set_progress_callback(progress_callback)
        
        result = pipeline.ingest_documents(str(temp_dir))
        
        if result["success"]:
            state.update_processing_status(90.0, "indexing")
            
            doc_id = str(uuid.uuid4())[:8]
            state.add_document(doc_id, files[0].name, result["chunks"], result["vectors"])
            state.document_count = result["documents"]
            
            for fid in file_ids:
                state.mark_file_processed(fid)
            
            state.update_processing_status(95.0, "finalizing")
            
            # Generate questions
            questions = generate_possible_questions(pipeline, max_questions=8, doc_id=doc_id)
            q_count = len(questions) if questions else 0
            
            if questions:
                tracker = get_question_tracker()
                tracker.set_questions(doc_id, questions)
                logger.info(f"✅ Generated {q_count} questions")
            
            state.finish_processing(success=True)
            st.session_state.initial_suggestions_shown = False
            
            st.balloons()
            
            welcome_msg = f"Hello! I've processed {result['documents']} document(s). I've generated {q_count} questions from your document to help you get started!"
            state.messages = [{
                "role": "assistant",
                "content": welcome_msg,
                "timestamp": datetime.now(ZoneInfo("Africa/Nairobi")),
            }]
            
            st.success(f"✅ Processed {result['documents']} documents")
            time.sleep(0.5)
            st.rerun()
        else:
            state.finish_processing(success=False, error=result.get('error', 'Unknown error'))
            st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        state.finish_processing(success=False, error=str(e))
        st.error(f"❌ Failed: {str(e)}")
        logger.error(f"Processing error: {e}", exc_info=True)

# =========================
# CHAT INTERFACE
# =========================

def render_chat_interface():
    state = get_app_state()
    pipeline = state.pipeline
    
    st.title("🤖 RAG Document Chatbot")
    st.caption("Ask questions about your documents - answers are grounded in your uploaded content")
    
    # Show hybrid status
    if pipeline and pipeline.use_hybrid_retrieval:
        hybrid = pipeline.hybrid_retriever
        if hybrid and not hybrid.is_ready():
            st.info("🔍 Hybrid Search: BM25 index will build on first query")
    
    # Display status
    display_status()
    
    # Process queue if triggered
    if st.session_state.get("trigger_process", False):
        st.session_state.trigger_process = False
        process_queue()
        # Rerun to show results
        st.rerun()
    
    # Render messages
    for message in state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("timestamp"):
                st.caption(f"🕐 {message['timestamp'].strftime('%I:%M %p')}")
            if message.get("sources") and message["role"] == "assistant":
                display_sources(message["sources"], max_sources=3)
            if message.get("response_time") and message["role"] == "assistant":
                st.caption(f"⚡ {message['response_time']:.2f}s")
    
    # Suggestions
    if state.is_ready and not st.session_state.get("is_processing", False):
        if state.messages:
            last_msg = state.messages[-1]
            if last_msg["role"] == "assistant":
                render_suggestions(state, max_to_show=4)
        elif not st.session_state.initial_suggestions_shown:
            st.markdown("### 💡 Get Started")
            render_suggestions(state, max_to_show=4)
            st.session_state.initial_suggestions_shown = True
    
    # Chat input
    user_input = st.chat_input(
        "Ask a question...",
        key="chat_input_main",
        disabled=st.session_state.get("is_processing", False)
    )
    
    if user_input:
        # Enqueue
        if not any(r["text"] == user_input for r in st.session_state.request_queue):
            request_id = str(uuid.uuid4())
            st.session_state.request_queue.append({"id": request_id, "text": user_input})
            st.session_state.trigger_process = True
            st.rerun()

# =========================
# MAIN
# =========================

def main():
    apply_custom_css()
    init_session_state()
    render_sidebar()
    st.divider()
    render_chat_interface()

if __name__ == "__main__":
    main()
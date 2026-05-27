"""
RAG Chatbot - Streamlit UI (Task 8 Final Integration)

This is the user-facing interface for the complete RAG system.

Features:
    - Document upload (PDF)
    - Chat interface for Q&A
    - Source attribution
    - System status monitoring
    - Support for Groq (recommended) and OpenAI
"""

import streamlit as st
from pathlib import Path
import tempfile
import time

# Import the RAG pipeline
from rag_pipeline import RAGPipeline


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
# SESSION STATE INITIALIZATION
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


# =========================
# SIDEBAR - CONFIGURATION
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
            help="Groq has a free tier (recommended). Mock for testing without API."
        )
        
        # Model selection based on provider
        if llm_provider == "groq":
            llm_model = st.selectbox(
                "Model",
                options=[
                    "llama-3.1-8b-instant",     # Fast + Stable (BEST FOR RAG)
                    "llama-3.3-70b-versatile",   # Higher Quality
                ],
                help="llama-3.1-8b-instant is recommended for speed and stability"
            )
            st.info("💡 Set GROQ_API_KEY in .env file or environment variables")
            
        elif llm_provider == "openai":
            llm_model = st.selectbox(
                "Model",
                options=[
                    "gpt-4o-mini",      # Cheap and fast (recommended)
                    "gpt-4o",           # Higher quality
                    "gpt-3.5-turbo",    # Legacy
                ],
                help="gpt-4o-mini is recommended for cost-effectiveness"
            )
            st.info("💡 Set OPENAI_API_KEY in .env file or environment variables")
            
        else:  # mock
            llm_model = st.text_input(
                "Mock Model Name",
                value="mock-model",
                disabled=True,
                help="Mock mode doesn't use a real API"
            )
            st.info("🔧 Mock mode - no API key required, returns test responses")
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More chunks = more context but slower"
        )
        
        score_threshold = st.slider(
            "Relevance threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,  # Lowered for better retrieval
            step=0.05,
            help="Lower = more results but less relevant. 0.2 is recommended."
        )
        
        temperature = st.slider(
            "LLM Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Higher = more creative but less focused"
        )
        
        st.divider()
        
        # Document Management
        st.subheader("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDFs to add to the knowledge base"
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Process Uploaded Documents", type="primary", use_container_width=True):
                    process_uploaded_documents(
                        uploaded_files, 
                        llm_provider, 
                        llm_model,
                        top_k, 
                        score_threshold, 
                        temperature
                    )
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.pipeline = None
                    st.session_state.is_ready = False
                    st.session_state.messages = []
                    st.session_state.document_count = 0
                    st.rerun()
        
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
            
            # Show current model
            st.caption(f"🧠 Model: {status.get('generator_info', {}).get('model', 'N/A')}")
            
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state.pipeline.clear()
                st.session_state.is_ready = False
                st.session_state.messages = []
                st.rerun()
        else:
            st.warning("⚠️ No documents loaded")
            st.info("Upload PDFs above to get started")
        
        st.divider()
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **RAG Document Chatbot**
            
            Built with:
            - PDF Loader (PyPDF)
            - Sentence Transformers (all-MiniLM-L6-v2)
            - Vector Store (in-memory with cosine similarity)
            - LLM: Groq (recommended) or OpenAI
            
            **How it works:**
            1. Upload PDF documents
            2. System chunks and embeds the text
            3. Ask questions in natural language
            4. System retrieves relevant chunks
            5. LLM generates grounded answers
            
            **Tips:**
            - Use specific questions for best results
            - Lower threshold = more results
            - llama-3.1-8b-instant is fastest for RAG
            """)


def process_uploaded_documents(files, llm_provider, llm_model, top_k, score_threshold, temperature):
    """Process uploaded PDF files and initialize the pipeline."""
    with st.spinner("Processing documents..."):
        # Save uploaded files to temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        for file in files:
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        # Initialize pipeline with selected model
        st.session_state.pipeline = RAGPipeline(
            llm_provider=llm_provider,
            llm_model=llm_model,  # Pass the selected model
            top_k=top_k,
            score_threshold=score_threshold,
            temperature=temperature,
            auto_load=False,
        )
        
        # Ingest documents
        result = st.session_state.pipeline.ingest_documents(str(temp_dir))
        
        if result["success"]:
            st.session_state.is_ready = True
            st.session_state.document_count = result["documents"]
            st.success(f"✅ Processed {result['documents']} documents with {result['chunks']} chunks")
            
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Hello! I've processed {result['documents']} document(s) using {llm_model}. Ask me anything about them!"
            })
            st.rerun()
        else:
            st.error(f"❌ Failed to process documents: {result.get('error', 'Unknown error')}")


# =========================
# MAIN CHAT INTERFACE
# =========================

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 RAG Document Chatbot")
    st.caption("Ask questions about your documents - answers are grounded in your uploaded content")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents... 🤔 Generating answer..."):
                if st.session_state.pipeline and st.session_state.is_ready:
                    try:
                        # Use the regular ask method (which has generic question handling)
                        # instead of ask_with_sources for better generic question handling
                        answer = st.session_state.pipeline.ask(prompt)
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Also try to get sources if available (for specific questions)
                        try:
                            result = st.session_state.pipeline.ask_with_sources(prompt)
                            sources = result.get("sources", [])
                            if sources:
                                with st.expander(f"📚 Sources ({len(sources)} relevant passages)"):
                                    for i, source in enumerate(sources, 1):
                                        st.markdown(f"**{i}. From: {source['source']}** (relevance: {source['score']:.3f})")
                                        st.caption(source['text'])
                                        st.divider()
                        except:
                            pass  # No sources to show
                        
                        # Store assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                else:
                    error_msg = "Pipeline not ready. Please upload documents first."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# =========================
# MAIN
# =========================

def main():
    """Main entry point for Streamlit app."""
    init_session_state()
    render_sidebar()
    
    st.divider()
    
    render_chat_interface()


if __name__ == "__main__":
    main()
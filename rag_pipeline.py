"""
RAG Pipeline - Task 8: Complete System Integration (OPTIMIZED + QUERY GUARD + FAISS + CHROMADB + HYBRID SEARCH)

This is the final assembly of all RAG components into a working AI assistant.

OPTIMIZATIONS APPLIED:
- Task 2: Similarity threshold filtering
- Task 3: Top-K optimization
- Task 4: Chunking optimization (500 chars, 75 overlap)
- Task 5: Lazy loading embeddings
- Task 6: Persistent vector storage (NPZ format)
- Task 7: Token + context optimization
- Task 8: Production-grade error handling
- FAISS: High-performance vector search (10x-100x faster)
- ChromaDB: Production vector database with automatic persistence
- Hybrid Search: BM25 + FAISS for optimal retrieval
- Query Guard: Pre-retrieval intelligence layer

Pipeline flow:
    User Question → Query Guard → Hybrid Search → Optimize Context → Generate → Return

Usage:
    pipeline = RAGPipeline(vector_store_backend="chromadb", enable_hybrid_search=True)
    pipeline.ingest_documents("data/")
    answer = pipeline.ask("What is RAG?")
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import os
import logging
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all RAG components
from core.pdf_loader import load_pdf, get_first_pdf
from core.chunker import Chunker, ChunkingStrategy
from services.embeddings import embed_chunks, get_embedder
from core.vector_store import VectorStore, create_vector_store_from_chunks, load_or_create_store
from services.retriever import Retriever, RetrieverConfig
from services.generator import LLMGenerator, create_generator
from services.query_guard import QueryGuard, QueryAction, create_query_guard

# FAISS import (optional)
try:
    from core.faiss_index import FaissIndex, create_faiss_index_from_vectorstore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

# ChromaDB import (optional)
try:
    from core.chromadb_store import ChromaDBStore, create_chromadb_store_from_chunks
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

# Hybrid Search import
try:
    from core.hybrid_search import HybridSearch, create_hybrid_search_from_vectorstore
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    logger.warning("Hybrid search dependencies not available")


class RAGPipeline:
    """
    Complete RAG pipeline orchestrating all components.
    
    PRODUCTION FEATURES:
    - Multiple vector store backends (Custom, FAISS, ChromaDB)
    - Hybrid search (BM25 + Vector)
    - Persistent storage (no recomputation)
    - Query Guard (pre-retrieval intelligence)
    - Graceful error handling
    - Context optimization
    - Token limit safety
    
    Usage:
        pipeline = RAGPipeline(vector_store_backend="chromadb", enable_hybrid_search=True)
        pipeline.ingest_documents("data/")
        answer = pipeline.ask("What is the document about?")
    """
    
    # Configuration constants for optimization
    MAX_CONTEXT_CHARS = 3000          # Maximum context length before trimming
    MAX_RESULTS_TO_LLM = 3            # Limit chunks sent to LLM (reduces tokens)
    MIN_RESULTS_FOR_FALLBACK = 1      # Always return at least 1 result
    
    # Query Guard thresholds
    QUERY_GUARD_HIGH_THRESHOLD = 0.7   # Score above this → USE as-is
    QUERY_GUARD_LOW_THRESHOLD = 0.3    # Score below this → SUGGEST alternatives
    
    # Persistence settings
    STORE_DIR = Path("data/stores")
    VECTOR_STORE_FILENAME = "vector_store.npz"
    FAISS_INDEX_FILENAME = "faiss_index"
    CHROMADB_DIR = "./chroma_db"
    
    def __init__(
        self,
        embedding_model: str = "mini-lm",
        llm_provider: str = "groq",
        llm_model: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.2,
        temperature: float = 0.2,
        auto_load: bool = True,
        enable_query_guard: bool = True,
        vector_store_backend: str = "faiss",  # "custom", "faiss", "chromadb", or "auto"
        enable_hybrid_search: bool = True,
        hybrid_vector_weight: float = 0.7,
        hybrid_keyword_weight: float = 0.3,
        persist_stores: bool = True,
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            embedding_model: "mini-lm", "mpnet", or "openai-small"
            llm_provider: "groq", "openai", or "mock"
            llm_model: Model name (e.g., "llama-3.1-8b-instant")
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            temperature: LLM creativity (0-1)
            auto_load: If True, automatically load documents from data/
            enable_query_guard: If True, enable pre-retrieval query processing
            vector_store_backend: "custom", "faiss", "chromadb", or "auto"
            enable_hybrid_search: If True, enable BM25 + FAISS hybrid search
            hybrid_vector_weight: Weight for vector search in hybrid (0-1)
            hybrid_keyword_weight: Weight for keyword search in hybrid (0-1)
            persist_stores: If True, save vector store and FAISS index to disk
        """
        print("=" * 60)
        print("🚀 Initializing RAG Pipeline...")
        print("=" * 60)
        
        # Store configuration
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.enable_query_guard = enable_query_guard
        self.vector_store_backend = vector_store_backend
        self.enable_hybrid_search = enable_hybrid_search and HYBRID_SEARCH_AVAILABLE
        self.hybrid_vector_weight = hybrid_vector_weight
        self.hybrid_keyword_weight = hybrid_keyword_weight
        self.persist_stores = persist_stores
        
        # Create store directory if needed
        if self.persist_stores:
            self.STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set default model if not provided
        if llm_model is None:
            if llm_provider == "groq":
                llm_model = "llama-3.1-8b-instant"
            elif llm_provider == "openai":
                llm_model = "gpt-4o-mini"
            else:
                llm_model = "mock-model"
        
        # Initialize components
        print("\n📦 Component Initialization:")
        
        # 1. Embedder (Task 5 - Lazy Loading)
        try:
            print("   - Loading embedder (lazy)...")
            self.embedder = get_embedder(model_name=embedding_model)
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            raise RuntimeError(f"Embedder initialization failed: {e}")
        
        # 2. Vector Store (Multiple backends)
        print(f"   - Initializing vector store (backend: {vector_store_backend})...")
        self.vector_store = None
        self.faiss_index = None
        self.chromadb_store = None
        self.hybrid_search = None
        self._loaded_from_cache = False
        
        # Initialize based on backend
        self._init_vector_store()
        
        # 3. Retriever (Task 6) - for linear search fallback
        print("   - Configuring retriever...")
        retriever_config = RetrieverConfig(
            top_k=top_k,
            score_threshold=score_threshold,
            min_results=self.MIN_RESULTS_FOR_FALLBACK,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store or VectorStore(),
            config=retriever_config,
            embedder=self.embedder,
        )
        
        # 4. Generator (Task 7)
        try:
            print(f"   - Initializing LLM ({llm_provider}) with model: {llm_model}...")
            self.generator = create_generator(
                provider=llm_provider,
                model=llm_model,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}\nCheck API keys in .env file")
        
        # 5. Chunker (Task 4 - Optimized)
        self.chunker = Chunker(
            strategy="sentence",
            chunk_size=500,      # Task 4: Optimized to 500
            chunk_overlap=75,    # Task 4: Optimized to 75
        )
        
        # 6. Query Guard (Pre-retrieval intelligence)
        self.query_guard = None  # Lazy initialized after documents are loaded
        
        # Track state
        self.is_ready = self._loaded_from_cache
        self.document_count = 0
        self.chunk_count = 0
        self.documents_hash = None
        
        # Auto-load documents if requested (only if not loaded from cache)
        if auto_load and not self._loaded_from_cache:
            self.ingest_documents()
        
        print("\n" + "=" * 60)
        if self.is_ready:
            print("✅ Pipeline ready! Ask questions with pipeline.ask()")
            if self._loaded_from_cache:
                print("   📦 Loaded from cache (fast startup)")
            if self.vector_store_backend == "chromadb" and self.chromadb_store:
                print(f"   🗄️  ChromaDB backend active")
            elif self.faiss_index:
                print(f"   🔍 FAISS backend active")
            if self.enable_hybrid_search and self.hybrid_search:
                print(f"   🔗 Hybrid search enabled (vector:{hybrid_vector_weight} + keyword:{hybrid_keyword_weight})")
            if self.enable_query_guard:
                print("🛡️  Query Guard enabled (pre-retrieval intelligence)")
        else:
            print("⚠️  Pipeline initialized. Load documents with pipeline.ingest_documents()")
        print("=" * 60)
    
    def _init_vector_store(self):
        """Initialize vector store based on selected backend."""
        backend = self.vector_store_backend
        
        # Auto-select best available backend
        if backend == "auto":
            if CHROMADB_AVAILABLE:
                backend = "chromadb"
                print("   - Auto-selected backend: chromadb")
            elif FAISS_AVAILABLE:
                backend = "faiss"
                print("   - Auto-selected backend: faiss")
            else:
                backend = "custom"
                print("   - Auto-selected backend: custom")
        
        # Initialize backend
        if backend == "chromadb" and CHROMADB_AVAILABLE:
            try:
                self.chromadb_store = ChromaDBStore(
                    collection_name="rag_documents",
                    persist_directory=self.CHROMADB_DIR,
                )
                self.vector_store = self.chromadb_store
                self.vector_store_backend = "chromadb"
                print("   ✅ ChromaDB vector store initialized")
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}, falling back to custom")
                self.vector_store = VectorStore()
                self.vector_store_backend = "custom"
                print("   ✅ Custom vector store initialized (fallback)")
        
        elif backend == "faiss" and FAISS_AVAILABLE:
            self.vector_store = VectorStore()  # FAISS index created after ingestion
            self.faiss_index = None
            self.vector_store_backend = "faiss"
            print("   ✅ Custom vector store + FAISS (index created after ingestion)")
        
        else:  # custom backend
            self.vector_store = VectorStore()
            self.vector_store_backend = "custom"
            print("   ✅ Custom vector store initialized")
        
        # Try to load existing stores (for custom/FAISS backends)
        if self.persist_stores and self.vector_store_backend != "chromadb":
            self._load_stores()
    
    def _load_stores(self):
        """Try to load existing vector store and FAISS index from disk."""
        vector_store_path = self.STORE_DIR / self.VECTOR_STORE_FILENAME
        
        if vector_store_path.exists():
            try:
                print(f"   📂 Loading cached vector store from {vector_store_path}...")
                self.vector_store = VectorStore.load_from_disk(str(vector_store_path))
                self._loaded_from_cache = True
                print(f"      ✅ Loaded {self.vector_store.size()} vectors from cache")
                
                # Try to load FAISS index if enabled
                if self.vector_store_backend == "faiss":
                    faiss_path = self.STORE_DIR / self.FAISS_INDEX_FILENAME
                    if faiss_path.with_suffix(".faiss").exists():
                        print(f"   📂 Loading FAISS index from {faiss_path}...")
                        self.faiss_index = FaissIndex.load(str(faiss_path))
                        print(f"      ✅ Loaded FAISS index with {self.faiss_index.size()} vectors")
                    else:
                        print(f"   🔍 Creating FAISS index from loaded vectors...")
                        self._create_faiss_index()
                        
            except Exception as e:
                logger.warning(f"Failed to load cached stores: {e}")
                self.vector_store = VectorStore()
                self._loaded_from_cache = False
    
    def _save_stores(self):
        """Save vector store and FAISS index to disk."""
        if not self.persist_stores:
            return
        
        # Skip saving for ChromaDB (auto-persists)
        if self.vector_store_backend == "chromadb":
            return
        
        if self.vector_store and self.vector_store.size() > 0:
            vector_store_path = self.STORE_DIR / self.VECTOR_STORE_FILENAME
            self.vector_store.save_to_disk(str(vector_store_path), format="npz")
            print(f"   💾 Saved vector store to {vector_store_path}")
            
            if self.faiss_index and self.faiss_index.size() > 0:
                faiss_path = self.STORE_DIR / self.FAISS_INDEX_FILENAME
                self.faiss_index.save(str(faiss_path))
                print(f"   💾 Saved FAISS index to {faiss_path}")
                    
    def _create_faiss_index(self):
        """Create FAISS index from current vector store with auto optimization."""
        if self.vector_store_backend != "faiss" or not FAISS_AVAILABLE:
            return
        
        if self.vector_store and self.vector_store.size() > 0:
            n_vectors = self.vector_store.size()
            print(f"   🔍 Creating FAISS index for {n_vectors} vectors...")
            
            # Auto-select optimal index type based on dataset size
            if n_vectors < 10000:
                index_type = "flat_ip"
                print(f"      Using Flat index (optimal for {n_vectors} vectors)")
            elif n_vectors < 50000:
                index_type = "ivf"
                n_lists = min(256, int(4 * np.sqrt(n_vectors)))
                print(f"      Using IVF index with {n_lists} lists (optimal for {n_vectors} vectors)")
            else:
                index_type = "hnsw"
                print(f"      Using HNSW index (optimal for {n_vectors} vectors)")
            
            self.faiss_index = create_faiss_index_from_vectorstore(
                self.vector_store,
                index_type=index_type,
            )
            print(f"      ✅ FAISS index created with {self.faiss_index.size()} vectors")
    
    def _init_hybrid_search(self):
        """Initialize hybrid search after documents are loaded."""
        if not self.enable_hybrid_search:
            return
        
        if self.hybrid_search is None and self.vector_store:
            print("🔗 Initializing hybrid search (BM25 + Vector)...")
            try:
                # Get vector store texts safely (backend-agnostic)
                texts = self._get_vector_store_texts()
                metadata = self._get_vector_store_metadata()
                
                if not texts:
                    print("   ⚠️ No texts available for hybrid search")
                    return
                
                # Create hybrid search
                self.hybrid_search = HybridSearch(
                    vector_weight=self.hybrid_vector_weight,
                    keyword_weight=self.hybrid_keyword_weight,
                    fusion_method="rrf",
                )
                
                # Initialize with texts and FAISS index if available
                self.hybrid_search.initialize(
                    chunks=texts,
                    metadata=metadata,
                    faiss_index=self.faiss_index,
                    build_bm25=True,
                )
                print(f"   ✅ Hybrid search ready (vector:{self.hybrid_vector_weight} + keyword:{self.hybrid_keyword_weight})")
            except Exception as e:
                logger.error(f"Failed to initialize hybrid search: {e}")
                self.hybrid_search = None
    
    def _init_query_guard(self):
        """Lazy initialize Query Guard after documents are loaded."""
        if not self.enable_query_guard:
            return
        
        if self.query_guard is None and self.vector_store:
            print("🛡️  Initializing Query Guard...")
            try:
                # Get chunk texts safely (backend-agnostic)
                chunk_texts = self._get_vector_store_texts()
                
                if chunk_texts:
                    self.query_guard = create_query_guard(
                        chunk_texts=chunk_texts,
                        high_threshold=self.QUERY_GUARD_HIGH_THRESHOLD,
                        low_threshold=self.QUERY_GUARD_LOW_THRESHOLD,
                    )
                    print(f"   ✅ Query Guard ready with {len(chunk_texts)} chunks")
                else:
                    print("   ⚠️ No chunks available for Query Guard")
                    self.query_guard = None
            except Exception as e:
                logger.error(f"Failed to initialize Query Guard: {e}")
                self.query_guard = None
    
    def _get_vector_store_texts(self) -> List[str]:
        """
        Get texts from vector store (backend-agnostic).
        
        Returns:
            List of chunk texts
        """
        if not self.vector_store:
            return []
        
        # Try different methods based on backend type
        if hasattr(self.vector_store, 'texts'):
            # Custom VectorStore or ChromaDBStore with texts property
            try:
                texts = self.vector_store.texts
                if texts is not None:
                    return texts
            except Exception as e:
                logger.warning(f"Failed to access .texts property: {e}")
        
        # Fallback for ChromaDBStore
        if hasattr(self.vector_store, 'collection'):
            try:
                results = self.vector_store.collection.get(limit=10000)
                if results and results.get('documents'):
                    return results['documents']
            except Exception as e:
                logger.warning(f"Failed to get texts from ChromaDB collection: {e}")
        
        # Last resort: try to get from _cached_texts
        if hasattr(self.vector_store, '_cached_texts') and self.vector_store._cached_texts:
            return self.vector_store._cached_texts
        
        return []
    
    def _get_vector_store_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata from vector store (backend-agnostic).
        
        Returns:
            List of metadata dicts
        """
        if not self.vector_store:
            return []
        
        # Try metadata property first
        if hasattr(self.vector_store, 'metadata'):
            try:
                metadata = self.vector_store.metadata
                if metadata is not None:
                    return metadata
            except Exception as e:
                logger.warning(f"Failed to access .metadata property: {e}")
        
        # Fallback for ChromaDB collection
        if hasattr(self.vector_store, 'collection'):
            try:
                results = self.vector_store.collection.get(limit=10000)
                if results and results.get('metadatas'):
                    metadatas = results['metadatas']
                    # Clean up placeholder metadata
                    cleaned = []
                    for meta in metadatas:
                        if meta and meta.get("_placeholder") is True:
                            cleaned.append({})
                        else:
                            cleaned.append(meta or {})
                    return cleaned
            except Exception as e:
                logger.warning(f"Failed to get metadata from ChromaDB: {e}")
        
        return []
    
    def _get_document_context(self, max_chars: int = 2000) -> str:
        """Get document context summary for relevance scoring (backend-agnostic)."""
        if not self.vector_store:
            return ""
        
        # Get texts safely with error handling
        try:
            texts = self._get_vector_store_texts()
            
            if not texts:
                return ""
            
            # Take first few chunks as document summary
            context_chunks = texts[:5]
            context = "\n\n".join(context_chunks)
            
            if len(context) > max_chars:
                context = context[:max_chars]
                last_space = context.rfind(' ')
                if last_space > 0:
                    context = context[:last_space] + "..."
            
            return context
        except Exception as e:
            logger.warning(f"Failed to get document context: {e}")
            return ""
    
    def _handle_query_guard_suggestions(self, decision) -> str:
        """Format Query Guard suggestions into a user-friendly message."""
        suggestions = decision.suggestions[:5]
        
        suggestions_text = "\n".join([f"• {s}" for s in suggestions])
        
        return f"""📚 I couldn't find relevant information for your question: "{decision.original_query}"

Here are some things you can ask about this document:

{suggestions_text}

Please try one of these questions, or ask something more specific about the document's content."""
    
    def _optimize_context(self, chunks: List[Any], max_chunks: Optional[int] = None) -> str:
        """Optimize context before sending to LLM."""
        if not chunks:
            return ""
        
        # Limit number of chunks
        limit = max_chunks or self.MAX_RESULTS_TO_LLM
        limited_chunks = chunks[:limit]
        
        # Format context
        texts = []
        for c in limited_chunks:
            if hasattr(c, 'text'):
                texts.append(c.text)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            else:
                texts.append(str(c))
        
        context = "\n\n---\n\n".join(texts)
        
        # Trim by characters if still too long
        if len(context) > self.MAX_CONTEXT_CHARS:
            logger.warning(f"Context too long ({len(context)} chars), trimming to {self.MAX_CONTEXT_CHARS}")
            context = context[:self.MAX_CONTEXT_CHARS]
            last_space = context.rfind(' ')
            if last_space > 0:
                context = context[:last_space] + "..."
        
        return context
    
    def ingest_documents(self, data_dir: str = "data") -> Dict[str, Any]:
        """
        Load and index all PDF documents from a directory with progress tracking.
        
        Args:
            data_dir: Directory containing PDF files
            
        Returns:
            Dictionary with ingestion statistics
        """
        import time
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("📄 Ingesting Documents")
        print("=" * 60)
        
        # Skip if already loaded from cache
        if self._loaded_from_cache and self.vector_store and self.vector_store.size() > 0:
            print("📦 Using cached documents (skipping ingestion)")
            return {
                "success": True,
                "documents": self.document_count,
                "chunks": self.chunk_count,
                "vectors": self.vector_store.size(),
                "from_cache": True,
            }
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Directory not found: {data_dir}")
            return {"success": False, "error": "Directory not found"}
        
        pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {data_dir}")
            return {"success": False, "error": "No PDF files found"}
        
        print(f"📁 Found {len(pdf_files)} PDF files")
        
        all_chunks = []
        all_metadata = []
        failed_files = []
        
        # Track processed files to avoid duplicates
        processed_sources = set()
        
        # Progress tracking for PDF loading
        pdf_load_start = time.time()
        
        for pdf_idx, pdf_path in enumerate(pdf_files):
            # Skip duplicate files
            if pdf_path.name in processed_sources:
                print(f"\n   ⚠️ Skipping duplicate: {pdf_path.name}")
                continue
            processed_sources.add(pdf_path.name)
            
            # Show PDF progress
            print(f"\n   [{pdf_idx + 1}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            try:
                text = load_pdf(str(pdf_path), light_clean=True)
                if not text or len(text.strip()) < 50:
                    print(f"      ⚠️  PDF has very little text (might be scanned)")
                    failed_files.append(pdf_path.name)
                    continue
                print(f"      Extracted {len(text)} characters")
            except Exception as e:
                print(f"      ❌ Error loading: {e}")
                failed_files.append(pdf_path.name)
                continue
            
            try:
                chunks = self.chunker.chunk(text)
                if not chunks:
                    print(f"      ⚠️  No chunks extracted")
                    failed_files.append(pdf_path.name)
                    continue
                print(f"      Created {len(chunks)} chunks")
            except Exception as e:
                print(f"      ❌ Chunking error: {e}")
                failed_files.append(pdf_path.name)
                continue
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": pdf_path.name,
                    "page": i // 10,
                    "chunk_index": i,
                })
        
        pdf_load_time = time.time() - pdf_load_start
        print(f"\n📊 PDF processing completed in {pdf_load_time:.2f}s")
        
        if not all_chunks:
            print("\n❌ No chunks extracted from any documents")
            return {"success": False, "error": "No chunks extracted", "failed_files": failed_files}
        
        try:
            print(f"\n🧠 Generating embeddings for {len(all_chunks)} chunks...")
            
            # Dynamic batch size based on chunks count
            if len(all_chunks) > 5000:
                batch_size = 32
                print(f"   Large document detected ({len(all_chunks)} chunks)")
                print(f"   Using larger batch size: {batch_size}")
            elif len(all_chunks) > 1000:
                batch_size = 24
                print(f"   Using batch size: {batch_size}")
            else:
                batch_size = 16
                print(f"   Using batch size: {batch_size}")
            
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            print(f"   Total batches: {total_batches}")
            
            successful_batches = 0
            failed_batches = 0
            
            # Progress tracking variables
            embedding_start = time.time()
            
            # Progress callback function for embeddings
            def update_progress(batch_num, total, eta):
                percent = (batch_num * 100) // total
                if eta < 60:
                    eta_str = f"{eta:.1f}s"
                else:
                    eta_str = f"{eta/60:.1f}m"
                
                # Use carriage return to update same line
                print(f"\r   Progress: {batch_num}/{total} ({percent}%) | ETA: {eta_str}", end="")
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                batch_metadata = all_metadata[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    # Get embeddings with progress callback
                    embedded = embed_chunks(
                        batch, 
                        batch_size=batch_size,
                        progress_callback=update_progress if batch_num == 1 else None  # Only show for first batch
                    )
                    
                    # Create NEW items with metadata
                    items_to_add = []
                    for j, item in enumerate(embedded):
                        new_item = {
                            "id": item.get("id", f"chunk_{i+j}"),
                            "text": item.get("text", batch[j]),
                            "vector_np": item.get("vector_np"),
                            "metadata": batch_metadata[j]
                        }
                        items_to_add.append(new_item)
                    
                    # Add to vector store
                    self.vector_store.add_batch(items_to_add)
                    successful_batches += 1
                    
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"Batch {batch_num} embedding failed: {e}")
                    continue
            
            print()  # New line after progress
            
            embedding_time = time.time() - embedding_start
            
            # Format time nicely
            if embedding_time < 60:
                embed_time_str = f"{embedding_time:.1f}s"
            else:
                embed_time_str = f"{embedding_time/60:.1f}m"
            
            print(f"\n\n✅ Embedding completed in {embed_time_str}!")
            print(f"   Documents: {len(pdf_files) - len(failed_files)}")
            print(f"   Failed files: {len(failed_files)}")
            print(f"   Total chunks: {len(all_chunks)}")
            print(f"   Successful batches: {successful_batches}")
            print(f"   Failed batches: {failed_batches}")
            
            vectors_stored = self.vector_store.size() if self.vector_store else 0
            print(f"   Vectors stored: {vectors_stored}")
            
            # Calculate and show speed
            if embedding_time > 0:
                chunks_per_second = len(all_chunks) / embedding_time
                print(f"   Speed: {chunks_per_second:.1f} chunks/second")
            
            # Validate vectors
            if vectors_stored < len(all_chunks) * 0.9:
                print(f"   ⚠️ WARNING: Only {vectors_stored}/{len(all_chunks)} vectors stored!")
                print(f"   Some embeddings may have failed. Check logs above.")
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Embedding failed: {e}"}
        
        # Create FAISS index if using FAISS backend
        if self.vector_store_backend == "faiss":
            self._create_faiss_index()
        
        # Initialize hybrid search if enabled
        if self.enable_hybrid_search:
            self._init_hybrid_search()
        
        # Save to disk for next time (only for non-ChromaDB backends)
        if self.persist_stores and self.vector_store_backend != "chromadb":
            self._save_stores()
        
        self.is_ready = True
        self.document_count = len(pdf_files) - len(failed_files)
        self.chunk_count = len(all_chunks)
        
        # Initialize Query Guard after documents are loaded
        if self.enable_query_guard and vectors_stored > 0:
            self._init_query_guard()
        
        total_time = time.time() - start_time
        if total_time < 60:
            total_time_str = f"{total_time:.1f}s"
        else:
            total_time_str = f"{total_time/60:.1f}m"
        
        print(f"\n🎉 Total ingestion time: {total_time_str}")
        
        return {
            "success": vectors_stored > 0,
            "documents": self.document_count,
            "chunks": len(all_chunks),
            "vectors": vectors_stored,
            "failed_files": failed_files,
            "from_cache": False,
            "processing_time": total_time,
            "speed_chunks_per_second": len(all_chunks) / embedding_time if embedding_time > 0 else 0
        }
    
    def _retrieve_chunks(self, query: str, query_vector: np.ndarray) -> List[Any]:
        """
        Retrieve chunks using the configured search method.
        
        FIXED: Always returns results, never filters too aggressively.
        Priority:
        1. Hybrid search (if enabled)
        2. ChromaDB (if using ChromaDB backend)
        3. FAISS (if available)
        4. Linear search (fallback)
        5. Absolute fallback: first chunks
        """
        results = []
        
        # 1. Try hybrid search first (best quality)
        if self.enable_hybrid_search and self.hybrid_search:
            try:
                results = self.hybrid_search.search(
                    query=query,
                    query_vector=query_vector,
                    top_k=self.top_k,
                    vector_top_k=self.top_k * 2,
                    keyword_top_k=self.top_k * 2,
                )
                if results:
                    logger.info(f"Hybrid search returned {len(results)} results")
                    return results[:self.top_k]  # ← Always return top K, no threshold
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")
        
        # 2. Try ChromaDB search (if using ChromaDB backend)
        if self.vector_store_backend == "chromadb" and hasattr(self.vector_store, 'search'):
            try:
                # CRITICAL: Pass None for score_threshold to disable filtering
                results = self.vector_store.search(query_vector, self.top_k, None)
                if results:
                    logger.info(f"ChromaDB search returned {len(results)} results")
                    # Convert to consistent format
                    formatted_results = []
                    for r in results[:self.top_k]:
                        if hasattr(r, 'text'):
                            formatted_results.append({
                                'text': r.text,
                                'score': r.score if hasattr(r, 'score') else 0.5,
                                'metadata': r.metadata if hasattr(r, 'metadata') else {},
                                'rank': r.rank if hasattr(r, 'rank') else 0
                            })
                        elif isinstance(r, dict):
                            formatted_results.append(r)
                        else:
                            formatted_results.append({'text': str(r), 'score': 0.5, 'metadata': {}})
                    return formatted_results
            except Exception as e:
                logger.warning(f"ChromaDB search failed: {e}")
        
        # 3. Try FAISS search
        if self.faiss_index and self.faiss_index.size() > 0:
            try:
                # CRITICAL: Pass None for score_threshold to disable filtering
                results = self.faiss_index.search(query_vector, self.top_k, None)
                if results:
                    logger.info(f"FAISS search returned {len(results)} results")
                    return results[:self.top_k]
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")
        
        # 4. Fallback to linear search via retriever
        if self.retriever:
            try:
                # Temporarily disable threshold for retriever
                original_threshold = self.retriever.config.score_threshold
                self.retriever.config.score_threshold = 0.0  # Disable filtering
                search_results = self.retriever.retrieve(query, top_k=self.top_k)
                self.retriever.config.score_threshold = original_threshold  # Restore
                
                if search_results:
                    logger.info(f"Linear search returned {len(search_results)} results")
                    return search_results[:self.top_k]
            except Exception as e:
                logger.error(f"Linear search failed: {e}")
        
        # 5. ABSOLUTE LAST RESORT: Return first N chunks from vector store
        texts = self._get_vector_store_texts()
        if texts:
            logger.warning(f"⚠️ Using fallback: first {min(self.top_k, len(texts))} chunks")
            fallback_results = []
            for i, text in enumerate(texts[:self.top_k]):
                fallback_results.append({
                    'text': text,
                    'score': 0.5,  # Neutral score
                    'metadata': {'source': 'fallback', 'rank': i},
                    'rank': i
                })
            return fallback_results
        
        # 6. Nothing worked
        logger.error("No chunks could be retrieved from any method")
        return []
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer.
        
        Args:
            question: User's question
            
        Returns:
            Generated answer string
        """
        if not self.is_ready:
            return "⚠️ Pipeline not ready. Please upload documents first."
        
        if not question or not question.strip():
            return "📝 Please provide a valid question."
        
        # Step 1: Query Guard
        query_to_use = question
        if self.enable_query_guard and self.query_guard:
            try:
                doc_context = self._get_document_context()
                decision = self.query_guard.process(
                    query=question,
                    document_context=doc_context,
                    document_chunks=self._get_vector_store_texts(),
                )
                
                logger.info(f"Query Guard decision: {decision.action.value} (score: {decision.relevance_score:.3f})")
                
                if decision.action == QueryAction.SUGGEST:
                    return self._handle_query_guard_suggestions(decision)
                
                query_to_use = decision.get_query_for_retrieval() or question
                
                if decision.action == QueryAction.REWRITE and decision.final_query:
                    logger.info(f"Query rewritten: '{question}' → '{decision.final_query}'")
                    
            except Exception as e:
                logger.error(f"Query Guard failed: {e}")
        
        # Step 2: Embed query
        try:
            query_vector = self.embedder.embed_single(query_to_use)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return "🔍 I couldn't process your question. Please try again."
        
        # Step 3: Retrieve chunks
        results = self._retrieve_chunks(query_to_use, query_vector)
        
        # Step 4: Handle empty results
        if not results:
            texts = self._get_vector_store_texts()
            if texts:
                fallback_chunks = texts[:2]
                context = "\n\n".join(fallback_chunks)
                try:
                    return self.generator.generate(
                        f"Answer this question based ONLY on the provided context. If you cannot answer, say so.\n\nQuestion: {question}\n\nContext:\n{context}",
                        context
                    )
                except Exception as e:
                    logger.error(f"Fallback generation failed: {e}")
                    return "📚 I couldn't find relevant information. Please try a different question."
            return "📚 I couldn't find any relevant information to answer your question."
        
        # Step 5: Generate answer
        try:
            # Prepare chunks for generator
            chunks_for_generation = []
            for r in results:
                if hasattr(r, 'text'):
                    chunks_for_generation.append({"text": r.text, "metadata": getattr(r, 'metadata', {})})
                elif isinstance(r, dict):
                    chunks_for_generation.append(r)
                else:
                    chunks_for_generation.append({"text": str(r), "metadata": {}})
            
            answer = self.generator.generate_from_chunks(query_to_use, chunks_for_generation)
            
            if not answer or len(answer.strip()) < 5:
                return "🤔 I found some information but couldn't formulate a clear answer. Please rephrase your question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            error_msg = str(e).lower()
            
            if "api key" in error_msg or "authentication" in error_msg:
                return "🔑 API key issue. Please check your API keys in the .env file."
            elif "rate limit" in error_msg:
                return "⏱️ API rate limit reached. Please wait a moment and try again."
            else:
                return f"❌ I encountered an error: {str(e)[:100]}. Please try again."
    
    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get answer with source attribution.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and retrieval info
        """
        if not self.is_ready:
            return {
                "answer": "⚠️ Pipeline not ready. Please load documents first.",
                "sources": [],
                "retrieval_info": {"error": "Not ready"},
            }
        
        if not question or not question.strip():
            return {
                "answer": "📝 Please provide a valid question.",
                "sources": [],
                "retrieval_info": {"error": "Empty question"},
            }
        
        # Apply Query Guard
        query_to_use = question
        if self.enable_query_guard and self.query_guard:
            try:
                doc_context = self._get_document_context()
                decision = self.query_guard.process(
                    question, 
                    doc_context, 
                    self._get_vector_store_texts()
                )
                
                if decision.action == QueryAction.SUGGEST:
                    return {
                        "answer": self._handle_query_guard_suggestions(decision),
                        "sources": [],
                        "retrieval_info": {"action": "suggest", "original_query": question},
                    }
                
                query_to_use = decision.get_query_for_retrieval() or question
                
            except Exception as e:
                logger.error(f"Query Guard failed: {e}")
        
        # Embed query
        try:
            query_vector = self.embedder.embed_single(query_to_use)
        except Exception as e:
            return {
                "answer": "🔍 I couldn't process your question. Please try again.",
                "sources": [],
                "retrieval_info": {"error": str(e)},
            }
        
        # Retrieve chunks
        results = self._retrieve_chunks(query_to_use, query_vector)
        
        if not results:
            return {
                "answer": "📚 I couldn't find any relevant information.",
                "sources": [],
                "retrieval_info": {"chunks_found": 0, "question": question},
            }
        
        # Generate answer with sources
        try:
            chunks_for_generation = []
            for r in results:
                if hasattr(r, 'text'):
                    chunks_for_generation.append({"text": r.text, "metadata": getattr(r, 'metadata', {})})
                elif isinstance(r, dict):
                    chunks_for_generation.append(r)
            
            answer = self.generator.generate_from_chunks(query_to_use, chunks_for_generation)
            
            # Prepare sources
            max_sources = 5
            sources = []
            for r in results[:max_sources]:
                if hasattr(r, 'text'):
                    text = r.text
                    score = getattr(r, 'score', 0.0)
                    metadata = getattr(r, 'metadata', {})
                    rank = getattr(r, 'rank', 0)
                elif isinstance(r, dict):
                    text = r.get("text", str(r))
                    score = r.get("score", 0.0)
                    metadata = r.get("metadata", {})
                    rank = r.get("rank", 0)
                else:
                    text = str(r)
                    score = 0.0
                    metadata = {}
                    rank = 0
                
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "score": round(score, 4),
                    "source": metadata.get("source", "Unknown"),
                    "rank": rank,
                })
            
            # Get top score safely
            top_score = 0.0
            if results:
                if hasattr(results[0], 'score'):
                    top_score = results[0].score
                elif isinstance(results[0], dict):
                    top_score = results[0].get("score", 0)
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_info": {
                    "chunks_found": len(results),
                    "top_score": round(top_score, 4),
                    "question": question,
                    "query_used": query_to_use if query_to_use != question else None,
                    "search_method": "hybrid" if self.enable_hybrid_search and self.hybrid_search else "vector",
                    "backend": self.vector_store_backend,
                },
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": f"⚠️ Could not generate answer: {str(e)[:100]}",
                "sources": [],
                "retrieval_info": {"error": str(e), "chunks_found": len(results)},
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and statistics."""
        status = {
            "ready": self.is_ready,
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "vectors": self.vector_store.size() if self.vector_store else 0,
            "config": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "max_context_chars": self.MAX_CONTEXT_CHARS,
                "max_results_to_llm": self.MAX_RESULTS_TO_LLM,
                "query_guard_enabled": self.enable_query_guard,
                "query_guard_high_threshold": self.QUERY_GUARD_HIGH_THRESHOLD,
                "query_guard_low_threshold": self.QUERY_GUARD_LOW_THRESHOLD,
                "vector_store_backend": self.vector_store_backend,
                "enable_hybrid_search": self.enable_hybrid_search and self.hybrid_search is not None,
                "hybrid_vector_weight": self.hybrid_vector_weight,
                "hybrid_keyword_weight": self.hybrid_keyword_weight,
                "persist_stores": self.persist_stores,
                "loaded_from_cache": self._loaded_from_cache,
            },
            "embedder_info": self.embedder.get_info(),
            "generator_info": self.generator.get_info(),
        }
        
        # Add backend-specific stats
        if self.faiss_index:
            status["faiss_stats"] = self.faiss_index.get_stats()
        
        if self.chromadb_store:
            status["chromadb_stats"] = self.chromadb_store.get_stats()
        
        if self.hybrid_search:
            status["hybrid_search_stats"] = self.hybrid_search.get_stats()
        
        return status
    
    def clear(self):
        """Clear all documents from the pipeline."""
        if self.vector_store:
            self.vector_store.clear()
        if self.faiss_index:
            self.faiss_index.clear()
        if self.chromadb_store:
            try:
                self.chromadb_store.delete_collection()
            except:
                pass
        if self.hybrid_search:
            self.hybrid_search = None
        self.is_ready = False
        self.document_count = 0
        self.chunk_count = 0
        self.query_guard = None
        self._loaded_from_cache = False
        print("🗑️ Pipeline cleared. All documents removed.")

    def get_document_preview(self, max_chunks: int = 3, max_chars_per_chunk: int = 300) -> Dict[str, Any]:
        """
        Get document preview information for UI display.
        
        This is a business logic method that the UI can call to display
        document preview without exposing internal implementation.
        
        Args:
            max_chunks: Number of chunks to preview
            max_chars_per_chunk: Max characters per chunk
        
        Returns:
            Dictionary with preview data
        """
        if not self.is_ready:
            return {"available": False, "message": "No documents loaded"}
        
        try:
            texts = self._get_vector_store_texts()
            if not texts:
                return {"available": False, "message": "No text chunks found"}
            
            preview_chunks = []
            for i, text in enumerate(texts[:max_chunks]):
                preview = text[:max_chars_per_chunk]
                if len(text) > max_chars_per_chunk:
                    preview = preview[:preview.rfind(' ')] + "..."
                preview_chunks.append({
                    "index": i,
                    "text": preview,
                    "full_length": len(text)
                })
            
            # Extract top keywords from document
            from collections import Counter
            import re
            
            all_text = " ".join(texts[:10])
            words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
            stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were',
                        'from', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
                        'their', 'there', 'they', 'them', 'what', 'when', 'where', 'which'}
            keywords = [w for w in words if w not in stop_words]
            top_keywords = [w for w, _ in Counter(keywords).most_common(8)]
            
            return {
                "available": True,
                "total_chunks": len(texts),
                "avg_chunk_size": sum(len(t) for t in texts[:10]) // min(10, len(texts)),
                "preview_chunks": preview_chunks,
                "top_keywords": top_keywords,
            }
            
        except Exception as e:
            logger.error(f"Failed to get document preview: {e}")
            return {"available": False, "message": str(e)}

    def get_possible_questions(self, max_questions: int = 5) -> List[str]:
        """
        Generate possible questions that can be answered from the document.
        
        This analyzes document content to suggest answerable questions.
        
        Args:
            max_questions: Maximum number of questions to return
        
        Returns:
            List of question strings
        """
        if not self.is_ready:
            return []
        
        # Cache key based on document content
        try:
            texts = self._get_vector_store_texts()
            if not texts:
                return []
            
            # Simple question generation from document content
            questions = []
            seen = set()
            
            # Method 1: Look for capitalized phrases (potential topics)
            import re
            all_text = " ".join(texts[:10])
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
            
            for phrase in list(dict.fromkeys(capitalized))[:10]:
                if len(phrase) > 3 and len(phrase) < 40 and phrase not in seen:
                    questions.append(f"What is {phrase}?")
                    seen.add(phrase)
            
            # Method 2: Look for sentences that contain definition patterns
            definition_patterns = [' is ', ' are ', ' refers to ', ' means ']
            sentences = [s.strip() for s in all_text.split('.') if len(s.strip()) > 30]
            
            for sentence in sentences[:10]:
                for pattern in definition_patterns:
                    if pattern in sentence:
                        subject = sentence.split(pattern)[0].strip()
                        if len(subject) > 3 and len(subject) < 50 and subject not in seen:
                            questions.append(f"What is {subject}?")
                            seen.add(subject)
                            break
            
            # Return top questions
            if questions:
                return questions[:max_questions]
            
            # Fallback questions
            return [
                "What is the main topic of this document?",
                "What are the key takeaways?",
                "Can you summarize this document?",
            ]
            
        except Exception as e:
            logger.warning(f"Failed to generate questions: {e}")
            return []

# =========================
# QUICK TEST
# =========================

def quick_test():
    """Quick test of the complete pipeline."""
    print("\n" + "=" * 60)
    print("🧪 Quick Pipeline Test")
    print("=" * 60)
    
    # Test with ChromaDB backend (since it's working)
    print(f"\n📌 Testing ChromaDB backend")
    print("-" * 40)
    
    try:
        pipeline = RAGPipeline(
            llm_provider="mock",
            auto_load=False,
            vector_store_backend="chromadb",
            enable_hybrid_search=False,
            persist_stores=True,
        )
        
        # Check if we have PDFs
        pdf_files = list(Path("data").glob("*.pdf"))
        if pdf_files:
            print(f"   Found {len(pdf_files)} PDF files")
            result = pipeline.ingest_documents()
            if result["success"]:
                print(f"   ✅ ChromaDB backend working")
                print(f"   📊 Statistics: {result['documents']} docs, {result['chunks']} chunks")
                
                # Test texts property
                print(f"\n   📝 Testing .texts property...")
                texts = pipeline._get_vector_store_texts()
                print(f"   Retrieved {len(texts)} texts")
                
                # Test metadata property
                print(f"\n   📝 Testing .metadata property...")
                metadata = pipeline._get_vector_store_metadata()
                print(f"   Retrieved {len(metadata)} metadata entries")
            else:
                print(f"   ⚠️  Ingestion failed: {result.get('error', 'Unknown')}")
        else:
            print(f"   ⚠️  No PDF files found in data/ directory")
            print("   Please add a PDF to data/ and run again")
            
    except Exception as e:
        print(f"   ❌ ChromaDB backend failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()
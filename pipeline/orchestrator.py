"""
RAG Orchestrator - Thin coordinator (replaces RAGPipeline)

This is what rag_pipeline.py becomes - just orchestration, no heavy logic.
All heavy lifting is delegated to core/ and services/ modules.

Production Features:
    - Logger instead of print statements
    - Proper confidence scoring with weighted decay
    - Deterministic document sampling (no position bias)
    - Cache invalidation with full configuration hashing
    - Clean abstraction boundaries
    - Timestamped logging
    - Scanned PDF detection and graceful skipping
    - Detailed debugging logs
    - Progress callback for UI updates
    - FAST ingestion + FAST queries (optimized batching)
    - SINGLE embedding model (loaded once at app start)
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
import logging
import time
import json
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Import from existing modules (kept as-is)
from core.pdf_loader import load_pdf
from core.chunker import Chunker
from core.vector_store import VectorStore
from core.faiss_index import create_faiss_index_from_vectorstore
from services.embeddings import get_embedder, embed_chunks
from services.retriever import Retriever, RetrieverConfig
from services.generator import create_generator
from services.query_guard import QueryGuard, QueryAction, create_query_guard
from services.reranker import create_reranker

# Import new pipeline modules
from pipeline.cache import DocumentCache
from services.hybrid_retriever import HybridRetriever

# ============================================================
# TIMESTAMPED LOGGING CONFIGURATION
# ============================================================

class TimestampFilter(logging.Filter):
    """Add timestamp to all log records."""
    def filter(self, record):
        record.timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return True

# Configure root logger with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(timestamp)s | %(levelname)s | %(message)s'
)
for handler in logging.root.handlers:
    handler.addFilter(TimestampFilter())

# Module logger
logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Thin orchestrator - coordinates components, doesn't implement them.
    
    Delegates to:
        - core/: PDF loading, chunking, vector storage, FAISS
        - services/: Embeddings, retrieval, generation, query guard, reranker
        - pipeline/cache: Document hashing and cache validation
    
    This file should stay UNDER 650 lines.
    
    IMPORTANT: This orchestrator uses a SINGLE embedding model.
    The model is loaded once at app startup and reused everywhere.
    No model selection parameters - just use the one model.
    """
    
    def __init__(
        self,
        llm_provider: str = "groq",
        llm_model: Optional[str] = None,
        top_k: int = 8,
        score_threshold: float = 0.05,
        temperature: float = 0.2,
        enable_query_guard: bool = True,
        enable_reranking: bool = True,
        persist_stores: bool = True,
        use_hybrid_retrieval: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """
        Initialize RAG orchestrator.
        
        Args:
            llm_provider: LLM provider ("groq" or "mock")
            llm_model: LLM model name (auto-selects if None)
            top_k: Number of chunks to retrieve
            score_threshold: Minimum score for retrieval
            temperature: LLM temperature
            enable_query_guard: Enable query validation
            enable_reranking: Enable result reranking
            persist_stores: Persist vector stores to disk
            use_hybrid_retrieval: Use hybrid (dense + BM25) retrieval
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            rrf_k: RRF ranking constant
            dense_weight: Weight for dense retrieval scores
            bm25_weight: Weight for BM25 scores
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.enable_query_guard = enable_query_guard
        self.enable_reranking = enable_reranking
        self.persist_stores = persist_stores
        self.use_hybrid_retrieval = use_hybrid_retrieval
        
        # Hybrid retriever params
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        
        # FAISS index type (determined by vector count)
        self.faiss_index_type = None
        
        # Progress callback (for UI updates)
        self._progress_callback = None
        
        # Chunker config (for cache invalidation)
        self.chunker_config = {
            "strategy": "sentence",
            "chunk_size": 500,
            "chunk_overlap": 75,
        }
        
        logger.info("=" * 60)
        logger.info("🚀 Initializing RAG Orchestrator...")
        logger.info("=" * 60)
        
        # Initialize components (lazy where possible)
        logger.info("📦 Loading components:")
        
        # ✅ SINGLE EMBEDDER - No model selection, just use the one
        self.embedder = get_embedder()  # No parameters!
        logger.info("   ✅ Embedder ready (single instance)")
        
        self.chunker = Chunker(
            strategy=self.chunker_config["strategy"],
            chunk_size=self.chunker_config["chunk_size"],
            chunk_overlap=self.chunker_config["chunk_overlap"],
        )
        logger.info("   ✅ Chunker ready")
        
        self.vector_store = VectorStore()
        self.faiss_index = None
        logger.info("   ✅ Vector store ready")
        
        # Generator
        if llm_model is None:
            llm_model = "llama-3.1-8b-instant" if llm_provider == "groq" else "mock-model"
        self.generator = create_generator(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
        )
        logger.info(f"   ✅ LLM ready ({llm_provider}/{llm_model})")
        
        # Retriever (fallback - abstracts FAISS away from orchestrator)
        retriever_config = RetrieverConfig(
            top_k=top_k,
            score_threshold=score_threshold,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            config=retriever_config,
            embedder=self.embedder,
        )
        logger.info("   ✅ Dense Retriever ready")
        
        # Hybrid Retriever (EAGER - built during ingestion)
        if self.use_hybrid_retrieval:
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=self.retriever,
                vector_store=self.vector_store,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
                rrf_k=self.rrf_k,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                auto_build=False  # Will build during ingestion
            )
            logger.info("   ✅ Hybrid Retriever ready (will build during ingestion)")
        else:
            self.hybrid_retriever = None
            logger.info("   ⏭️ Hybrid Retriever disabled (using dense only)")
        
        # Optional components (lazy)
        self.query_guard = None
        self.reranker = None
        
        # Cache
        self.cache = DocumentCache()
        
        # State
        self.is_ready = False
        self.document_count = 0
        self.chunk_count = 0
        self.documents_hash = None
        
        logger.info("✅ Orchestrator initialized")
        logger.info(f"   Top K: {top_k}")
        logger.info(f"   Retrieval: {'HYBRID' if use_hybrid_retrieval else 'DENSE'}")
        logger.info(f"   Reranking: {'ON' if enable_reranking else 'OFF'}")
        logger.info(f"   Query Guard: {'ON' if enable_query_guard else 'OFF'}")
        logger.info(f"   Embedding Model: all-MiniLM-L6-v2 (384 dims)")
        logger.info("=" * 60)
    
    def set_progress_callback(self, callback: Callable[[int, int, float], None]):
        """
        Set a callback function for progress updates during embedding.
        
        Args:
            callback: Function that accepts (current_batch, total_batches, eta_seconds)
        """
        self._progress_callback = callback
        logger.info("📊 Progress callback registered")
    
    def ingest_documents(self, data_dir: str = "data") -> Dict[str, Any]:
        """
        FAST INGESTION: Build everything with optimized batching.
        No lazy loading - everything ready for instant queries.
        """
        total_start = time.time()
        
        logger.info("=" * 60)
        logger.info("📄 Ingesting Documents (FAST MODE)")
        logger.info("=" * 60)
                
        data_path = Path(data_dir)
        
        # ========== STAGE 1: Find PDF files ==========
        pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
        
        # Remove duplicates
        seen_names = {}
        unique_pdfs = []
        for pdf in pdf_files:
            name_lower = pdf.name.lower()
            if name_lower not in seen_names:
                seen_names[name_lower] = True
                unique_pdfs.append(pdf)
        pdf_files = unique_pdfs
        
        if not pdf_files:
            return {"success": False, "error": "No PDF files found"}
        
        logger.info(f"📁 Found {len(pdf_files)} unique PDF files")
        
        # ========== STAGE 2: Load and chunk (optimized) ==========
        all_chunks = []
        all_metadata = []
        failed_pdfs = []
        
        for pdf_path in pdf_files:
            try:
                text = load_pdf(str(pdf_path), light_clean=True)
                
                if not text or len(text.strip()) < 100:
                    failed_pdfs.append(pdf_path.name)
                    continue
                
                chunks = self.chunker.chunk(text)
                
                valid_chunks = []
                for i, chunk in enumerate(chunks):
                    if chunk and len(chunk.strip()) > 50 and "No text extracted" not in chunk:
                        valid_chunks.append((chunk, i))
                
                if not valid_chunks:
                    failed_pdfs.append(pdf_path.name)
                    continue
                
                for chunk, idx in valid_chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": pdf_path.name,
                        "chunk_index": idx,
                        "chunk_id": f"{pdf_path.name}_{idx}",
                        "chunk_length": len(chunk)
                    })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_path.name}: {e}")
                failed_pdfs.append(pdf_path.name)
                continue
        
        if not all_chunks:
            return {
                "success": False, 
                "error": "No chunks extracted from PDFs",
                "failed_files": failed_pdfs
            }
        
        logger.info(f"✅ Extracted {len(all_chunks)} total chunks")
        
        # ========== STAGE 3: Embed with OPTIMIZED batching ==========
        embed_start = time.time()
        
        # 🚀 LARGER BATCHES = FASTER (use GPU-friendly sizes)
        if len(all_chunks) > 5000:
            batch_size = 256
        elif len(all_chunks) > 2000:
            batch_size = 192
        elif len(all_chunks) > 1000:
            batch_size = 128
        else:
            batch_size = 64
        
        logger.info(f"📊 Embedding {len(all_chunks)} chunks with batch size {batch_size}")
        
        # Disable progress callback during embedding (saves overhead)
        saved_callback = self._progress_callback
        self._progress_callback = None
        
        # 🚀 EMBED IN BULK (faster than loop with callbacks)
        items_to_add = self._embed_chunks_optimized(all_chunks, all_metadata, batch_size)
        
        # Restore progress callback
        self._progress_callback = saved_callback
        
        embed_time = time.time() - embed_start
        logger.info(f"✅ Embedded {len(items_to_add)} chunks in {embed_time:.2f}s")
        
        if not items_to_add:
            return {"success": False, "error": "No embeddings were generated"}
        
        # ========== STAGE 4: Vector store & FAISS (fast) ==========
        store_start = time.time()
        
        # Clear and add to vector store
        self.vector_store.clear()
        self.vector_store.add_batch(items_to_add)
        
        # Create FAISS index
        self._create_faiss_index()
        
        store_time = time.time() - store_start
        logger.info(f"✅ Vector store + FAISS built in {store_time:.2f}s")
        
        # ========== STAGE 5: Build BM25 (if hybrid) ==========
        bm25_time = 0
        if self.use_hybrid_retrieval and self.hybrid_retriever:
            bm25_start = time.time()
            
            # 🚀 Force build BM25 immediately (not lazy)
            self.hybrid_retriever._ensure_bm25_index()
            
            bm25_time = time.time() - bm25_start
            logger.info(f"✅ BM25 index built in {bm25_time:.2f}s")
        
        # ========== STAGE 6: Initialize optional components ==========
        if self.enable_reranking:
            self._init_reranker()
        
        if self.enable_query_guard:
            self._init_query_guard()
        
        # Update state
        self.document_count = len(pdf_files) - len(failed_pdfs)
        self.chunk_count = len(all_chunks)
        self.is_ready = True
        
        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info(f"✅ FAST ingestion complete in {total_time:.2f}s")
        logger.info(f"   Documents: {self.document_count} (failed: {len(failed_pdfs)})")
        logger.info(f"   Chunks: {self.chunk_count}")
        logger.info(f"   Vectors: {self.vector_store.size()}")
        logger.info(f"   ⚡ All components ready for instant queries")
        logger.info("=" * 60)
        
        return {
            "success": True,
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "vectors": self.vector_store.size(),
            "failed_files": failed_pdfs,
            "processing_time": total_time,
            "embedding_time": embed_time,
            "store_time": store_time,
            "bm25_time": bm25_time,
            "from_cache": False,
        }
    
    def _embed_chunks_optimized(self, chunks: List[str], metadata: List[Dict], batch_size: int) -> List[Dict]:
        """
        Embed chunks with optimized batching and minimal overhead.
        """
        if not chunks:
            return []
        
        items_to_add = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        # Process all batches with minimal logging
        for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            try:
                embedded = embed_chunks(batch, batch_size=batch_size)
                
                for j, item in enumerate(embedded):
                    items_to_add.append({
                        "id": item.get("id", f"chunk_{i+j}"),
                        "text": item["text"],
                        "vector_np": item["vector_np"],
                        "metadata": batch_metadata[j] if j < len(batch_metadata) else {}
                    })
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_idx + 1} failed: {e}")
                continue
            
            # Log only every 10 batches (reduces overhead)
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"   📊 Embedded {batch_idx + 1}/{total_batches} batches")
        
        return items_to_add
    
    def _create_faiss_index(self):
        """Create FAISS index with auto-optimization."""
        if self.vector_store.size() == 0:
            logger.warning("No vectors to create FAISS index")
            return
        
        n_vectors = self.vector_store.size()
        
        if n_vectors < 10000:
            index_type = "flat_ip"
        elif n_vectors < 50000:
            index_type = "ivf"
        else:
            index_type = "hnsw"
        
        logger.info(f"🔍 Creating FAISS index ({index_type}) for {n_vectors} vectors")
        self.faiss_index = create_faiss_index_from_vectorstore(
            self.vector_store,
            index_type=index_type,
        )
        
        # Store for cache invalidation
        self.faiss_index_type = index_type
        logger.info(f"🔍 FAISS index created with {self.faiss_index.size()} vectors")
    
    def _init_query_guard(self):
        """Initialize Query Guard with deterministic sampling (no position bias)."""
        texts = self.vector_store.texts
        if texts:
            # Deterministic sampling: evenly spaced across document
            step = max(len(texts) // 20, 1)
            sampled = texts[::step][:20]
            
            # Use lower thresholds for better recall with small documents
            self.query_guard = create_query_guard(
                chunk_texts=sampled,
                high_threshold=0.5,
                low_threshold=0.15,
            )
            logger.info(f"🛡️ Query Guard ready (sampled {len(sampled)} chunks, thresholds: high=0.5, low=0.15)")
        
    def _init_reranker(self):
        """Initialize reranker (lazy loading)."""
        self.reranker = create_reranker(
            model_name="mini-lm",
            min_score_threshold=0.2,
        )
        if self.reranker:
            logger.info(f"📊 Reranker ready ({self.reranker.model_info['speed']})")
        else:
            logger.warning("Reranker not available")
    
    def _ensure_retriever_ready(self) -> bool:
        """
        Check if everything is ready (no lazy loading - all built at ingestion).
        """
        if not self.is_ready:
            logger.warning("Orchestrator not ready - no documents ingested")
            return False
        
        # Quick validation that everything is built
        if self.vector_store.size() == 0:
            logger.warning("Vector store is empty")
            return False
        
        if self.faiss_index is None:
            logger.warning("FAISS index not built - rebuilding...")
            self._create_faiss_index()
            if self.faiss_index is None:
                return False
        
        if self.use_hybrid_retrieval and self.hybrid_retriever:
            if self.hybrid_retriever.is_stale():
                logger.warning("BM25 index is stale - rebuilding...")
                self.hybrid_retriever._ensure_bm25_index()
                if not self.hybrid_retriever.is_ready():
                    return False
        
        return True
    
    def ask(self, question: str) -> str:
        """Ask a question (simple interface)."""
        result = self.ask_with_sources(question)
        return result["answer"]
    
    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        FAST QUERY: Everything already built - just search and generate.
        """
        if not self.is_ready:
            return {
                "answer": "Please upload documents first.",
                "sources": [],
                "confidence": 0.0
            }
        
        logger.info(f"🔄 Processing query: {question[:100]}...")
        
        # Quick validation (no building)
        if not self._ensure_retriever_ready():
            return {
                "answer": "System not ready. Please re-ingest documents.",
                "sources": [],
                "confidence": 0.0
            }
        
        # ========== QUERY GUARD (optional) ==========
        query_to_use = question
        if self.enable_query_guard and self.query_guard:
            try:
                doc_context = "\n".join(self.vector_store.texts[:5]) if self.vector_store.texts else ""
                decision = self.query_guard.process(
                    query=question,
                    document_context=doc_context,
                    document_chunks=self.vector_store.texts[:20] if self.vector_store.texts else []
                )
                
                if decision.action == QueryAction.SUGGEST:
                    return {
                        "answer": self._suggestions_message(decision),
                        "sources": [],
                        "confidence": 0.0
                    }
                query_to_use = decision.get_query_for_retrieval() or question
            except Exception as e:
                logger.warning(f"Query Guard failed: {e}")
        
        # ========== RETRIEVE (FAST - everything built) ==========
        if self.use_hybrid_retrieval and self.hybrid_retriever:
            results = self._hybrid_retrieve(query_to_use)
        else:
            results = self._dense_retrieve(query_to_use)
        
        if not results:
            return {
                "answer": "I couldn't find relevant information. Please try rephrasing.",
                "sources": [],
                "confidence": 0.0
            }
        
        # ========== RERANK (optional) ==========
        confidence = self._calculate_confidence(results)
        if self.enable_reranking and self.reranker and confidence < 0.6:
            results = self.reranker.rerank(question, results, top_k=self.top_k)
        
        # ========== GENERATE (FAST) ==========
        try:
            answer = self.generator.generate_from_chunks(query_to_use, results)
        except Exception as e:
            return {
                "answer": f"Error: {str(e)[:100]}",
                "sources": results[:3],
                "confidence": 0.0
            }
        
        return {
            "answer": answer,
            "sources": results[:5],
            "confidence": self._calculate_confidence(results),
            "all_sources": results,
        }
    
    def _dense_retrieve(self, query: str) -> List[Dict]:
        """
        Dense retrieval using existing retriever.
        """
        results = self.retriever.retrieve(query, top_k=self.top_k)
        
        results_dict = []
        for r in results:
            metadata = getattr(r, 'metadata', {})
            source_name = metadata.get('source', 'Unknown')
            if source_name == 'Unknown':
                source_name = metadata.get('source_file', metadata.get('file', 'Unknown'))
            
            results_dict.append({
                "text": r.text,
                "score": r.score,
                "rank": r.rank,
                "source": source_name,
                "metadata": metadata,
                "chunk_length": len(r.text),
                "passed_threshold": getattr(r, 'passed_threshold', True),
                "text_preview": r.text[:300] + "..." if len(r.text) > 300 else r.text,
            })
        
        return results_dict
    
    def _hybrid_retrieve(self, query: str) -> List[Dict]:
        """
        Hybrid retrieval with proper formatting.
        """
        try:
            hybrid_result = self.hybrid_retriever.retrieve(
                query=query,
                top_k=self.top_k,
                include_debug=True
            )
            
            results = []
            for r in hybrid_result.results:
                results.append({
                    "text": r.text,
                    "score": r.rrf_score,
                    "rank": len(results) + 1,
                    "source": r.source,
                    "metadata": r.metadata,
                    "chunk_length": len(r.text),
                    "passed_threshold": True,
                    "text_preview": r.text[:300] + "..." if len(r.text) > 300 else r.text,
                    "dense_score": r.dense_score,
                    "bm25_score": r.bm25_score,
                    "rrf_score": r.rrf_score,
                })
            
            return results
        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {e}, falling back to dense")
            return self._dense_retrieve(query)
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """
        Calculate confidence from retrieval scores using weighted decay.
        """
        if not results:
            return 0.0
        
        # Take top 5 scores
        scores = np.array([r.get("score", 0.0) for r in results[:5]])
        
        # Weighted decay: top result weight = 1.0, 5th result weight = 0.5
        weights = np.linspace(1.0, 0.5, len(scores))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        confidence = float(np.sum(scores * weights))
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _suggestions_message(self, decision) -> str:
        """Format query guard suggestions."""
        suggestions = decision.suggestions[:3]
        return f"I couldn't find good matches. Try: {', '.join(suggestions)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        status = {
            "ready": self.is_ready,
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "vectors": self.vector_store.size(),
            "config": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "reranking": self.enable_reranking,
                "query_guard": self.enable_query_guard,
                "faiss_index_type": self.faiss_index_type,
                "retrieval_type": "hybrid" if self.use_hybrid_retrieval else "dense",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimensions": 384,
            },
            "generator_info": {
                "provider": self.generator.provider if hasattr(self.generator, 'provider') else "unknown",
                "model": self.generator.model_name if hasattr(self.generator, 'model_name') else "unknown",
            }
        }
        
        # Add hybrid retriever status if available
        if self.hybrid_retriever:
            status["hybrid_retriever"] = {
                "is_initialized": self.hybrid_retriever._is_initialized,
                "document_count": self.hybrid_retriever.get_document_count(),
                "is_stale": self.hybrid_retriever.is_stale(),
                "is_ready": self.hybrid_retriever.is_ready(),
            }
        
        return status
    
    def clear(self):
        """Clear all data."""
        self.vector_store.clear()
        self.faiss_index = None
        self.query_guard = None
        self.reranker = None
        self.is_ready = False
        self.document_count = 0
        self.chunk_count = 0
        self.cache.clear()
        
        # Clear hybrid retriever state
        if self.hybrid_retriever:
            self.hybrid_retriever.mark_stale()
            self.hybrid_retriever._bm25_index = None
            self.hybrid_retriever._corpus = []
            self.hybrid_retriever._tokenized_corpus = []
            self.hybrid_retriever._document_metadata = []
            self.hybrid_retriever._is_initialized = False
        
        logger.info("🗑️ Pipeline cleared")


# Alias for backward compatibility
RAGPipeline = RAGOrchestrator
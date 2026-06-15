"""
Retriever Module - Task 6 of RAG Pipeline (OPTIMIZED: Tasks 2 & 3)

Responsibility: Bridge between embeddings, vector store, and LLM generation.
Single responsibility: query → relevant chunks (orchestration only).

OPTIMIZATIONS APPLIED:
- Task 2: Similarity threshold filtering (quality control gate)
- Task 3: Top-K optimization (focus control)

This module handles:
- Query embedding
- Vector store search
- Score thresholding (with intelligent fallback)
- Top-K optimization (3-5 chunks instead of 10+)
- Context formatting for LLM prompts
- Enhanced metadata for debugging (source tracking)

Architecture:
    User Query → Embed Query → Vector Search → Top-K Limit → Threshold Filter → Clean Context
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """
    Configuration for the retriever.
    
    OPTIMIZED DEFAULTS (Tasks 2 & 3):
    - top_k: 5 (optimal for RAG, was higher)
    - score_threshold: 0.3 (filters out noise below 30% similarity)
    - min_results: 2 (ensures at least 2 chunks for context)
    - include_scores: True (useful for debugging/threshold tuning)
    
    Attributes:
        top_k: Number of chunks to retrieve (OPTIMIZED: 5 is sweet spot)
        score_threshold: Minimum similarity score (0-1). Results below filtered.
        min_results: Minimum results to return (prevents empty context)
        include_scores: Include similarity scores in results
        context_separator: String used to join chunks in formatted context
        enable_threshold_filter: Master switch for threshold filtering
    """
    top_k: int = 5                      # TASK 3: Reduced from higher values
    score_threshold: float = 0.3        # TASK 2: Quality gate
    min_results: int = 2                # Ensure at least 2 chunks for context
    include_scores: bool = True
    context_separator: str = "\n\n---\n\n"
    enable_threshold_filter: bool = True  # Master switch for Task 2
    
    def __post_init__(self):
        """Validate configuration."""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.top_k > 20:
            print(f"⚠️  Warning: top_k={self.top_k} is high. Recommended: 3-7 for RAG")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError(f"score_threshold must be between 0 and 1, got {self.score_threshold}")
        if self.min_results < 1:
            raise ValueError(f"min_results must be >= 1, got {self.min_results}")
        if self.min_results > self.top_k:
            raise ValueError(f"min_results ({self.min_results}) cannot exceed top_k ({self.top_k})")


@dataclass
class RetrievalResult:
    """
    Single retrieval result with all relevant information.
    
    Attributes:
        text: The chunk text
        score: Similarity score (0-1, higher = more relevant)
        rank: Position in results (1 = most relevant)
        metadata: Optional metadata from original chunk
        passed_threshold: Whether this chunk met the score_threshold
        source: Convenience access to source document name
    """
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    passed_threshold: bool = True
    
    @property
    def source(self) -> str:
        """Convenience property to get source document name."""
        return self.metadata.get('source', self.metadata.get('source_file', 'Unknown'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict with enhanced debug info."""
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "rank": self.rank,
            "metadata": self.metadata,
            "source": self.source,
            "passed_threshold": self.passed_threshold,
            "chunk_length": len(self.text),
        }
    
    def to_debug_dict(self) -> Dict[str, Any]:
        """
        Enhanced debug output for UI panel.
        Includes first 500 chars and formatted score.
        """
        return {
            "text_preview": self.text[:500] + ("..." if len(self.text) > 500 else ""),
            "full_text": self.text,
            "score": round(self.score, 4),
            "score_percentage": f"{self.score * 100:.1f}%",
            "source": self.source,
            "rank": self.rank,
            "chunk_size": len(self.text),
            "passed_threshold": self.passed_threshold,
        }


class Retriever:
    """
    Orchestrates retrieval: query → relevant chunks.
    
    OPTIMIZED v1 Features:
    - Embed query using Task 4 embeddings
    - Search vector store (Task 5)
    - Top-K limit (Task 3: reduced to 3-7 chunks)
    - Score thresholding with intelligent fallback (Task 2)
    - Context formatting for LLM prompts
    - Enhanced metadata propagation for debugging
    
    v2 Upgrade Paths:
    - Hybrid search (keyword + semantic)
    - Reranking with cross-encoders
    - Metadata filtering
    - Multi-query retrieval
    
    Usage:
        retriever = Retriever(vector_store)
        results = retriever.retrieve("What is RAG?")
        context = retriever.format_context(results)
        
        # For debug panel
        debug_info = retriever.get_debug_info(results)
    """
    
    def __init__(
        self,
        vector_store,
        config: Optional[RetrieverConfig] = None,
        embedder = None,
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance from Task 5
            config: RetrieverConfig (defaults with optimized values)
            embedder: Optional embedder instance. If None, uses get_embedder()
        """
        self.vector_store = vector_store
        self.config = config or RetrieverConfig()
        
        # Lazy import to avoid circular dependency
        if embedder is None:
            from services.embeddings import get_embedder
            self._embedder = get_embedder()
        else:
            self._embedder = embedder
        
        # Track last retrieval for debugging
        self._last_query = None
        self._last_results = None
        self._last_timestamp = None
    
    def _apply_threshold_filter(
        self,
        results: List[Any],
        threshold: float,
        min_results: int,
    ) -> List[Any]:
        """
        TASK 2: Apply similarity threshold with intelligent fallback.
        
        This is the quality control gate.
        
        Strategy:
        1. Keep all results above threshold
        2. If not enough results, add top results below threshold
        3. Never return fewer than min_results (prevents empty context)
        
        Args:
            results: Search results from vector store
            threshold: Minimum similarity score (0-1)
            min_results: Minimum number of results to return
        
        Returns:
            Filtered results (guaranteed at least min_results)
        """
        if not results:
            return []
        
        # Separate high vs low quality results
        high_quality = [r for r in results if r.score >= threshold]
        low_quality = [r for r in results if r.score < threshold]
        
        # Start with high-quality results
        final_results = high_quality.copy()
        
        # If we need more to reach min_results, add best low-quality results
        if len(final_results) < min_results and low_quality:
            needed = min_results - len(final_results)
            final_results.extend(low_quality[:needed])
            logger.warning(f"Threshold fallback: {len(high_quality)} passed, added {needed} below threshold")
        
        # Log if we're returning low-quality results
        if final_results and final_results[-1].score < threshold:
            logger.debug(f"Last result score: {final_results[-1].score:.3f} (below threshold {threshold})")
        
        return final_results
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        apply_fallback: bool = True,
    ) -> List[RetrievalResult]:
        """
        Main retrieval interface: query → list of relevant chunks.
        
        TASK 2 & 3: This method now applies both Top-K limit and threshold filtering.
        
        Args:
            query: User question or search query
            top_k: Override config top_k for this call
            score_threshold: Override config score_threshold for this call
            apply_fallback: If True, ensures at least min_results are returned
        
        Returns:
            List of RetrievalResult objects, sorted by relevance descending
        
        Example:
            >>> retriever = Retriever(store)
            >>> results = retriever.retrieve("What is machine learning?")
            >>> for r in results:
            ...     print(f"[{r.rank}] Score {r.score:.3f}: {r.text[:50]}")
        """
        if not query or not query.strip():
            return []
        
        # Store query for debugging
        self._last_query = query
        
        # Use overrides or config defaults
        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        
        logger.debug(f"Retrieving for query: {query[:100]}... (top_k={k}, threshold={threshold})")
        
        # TASK 3: Top-K optimization - get results
        query_vector = self._embedder.embed_single(query)
        
        if len(query_vector) == 0:
            logger.warning("Empty query vector generated")
            return []
        
        # Get more results initially to allow filtering (2x top_k for fallback)
        search_k = k * 2 if apply_fallback else k
        search_results = self.vector_store.search(
            query_vector,
            top_k=search_k,
            score_threshold=0.0,  # Get all, we'll filter ourselves
        )
        
        # TASK 3: Apply Top-K limit first
        search_results = search_results[:search_k]
        
        # TASK 2: Apply threshold filtering
        if self.config.enable_threshold_filter and apply_fallback:
            filtered_results = self._apply_threshold_filter(
                search_results, 
                threshold, 
                self.config.min_results
            )
        else:
            # Strict mode: only results above threshold
            filtered_results = [r for r in search_results if r.score >= threshold]
        
        # Convert to RetrievalResult objects with enhanced metadata
        results = []
        for rank, result in enumerate(filtered_results[:k], 1):  # Ensure final Top-K limit
            
            # Ensure metadata has source info
            metadata = getattr(result, 'metadata', {})
            if 'source' not in metadata and 'source_file' in metadata:
                metadata['source'] = metadata['source_file']
            elif 'source' not in metadata and 'file' in metadata:
                metadata['source'] = metadata['file']
            elif 'source' not in metadata:
                metadata['source'] = 'Unknown'
            
            results.append(RetrievalResult(
                text=result.text,
                score=result.score,
                rank=rank,
                metadata=metadata,
                passed_threshold=result.score >= threshold,
            ))
        
        # Store results for debugging
        self._last_results = results
        self._last_timestamp = __import__('datetime').datetime.now()
        
        # Log retrieval quality (for debugging/monitoring)
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            logger.info(f"Retrieved {len(results)} chunks, avg score: {avg_score:.3f}")
            if avg_score < threshold:
                logger.warning(f"Low average similarity: {avg_score:.3f} (threshold={threshold})")
        
        return results
    
    def retrieve_texts(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        apply_fallback: bool = True,
    ) -> List[str]:
        """
        Simplified interface: returns only chunk texts.
        
        Useful when you only need the text content.
        """
        results = self.retrieve(query, top_k, score_threshold, apply_fallback)
        return [r.text for r in results]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        apply_fallback: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Returns (text, score) tuples.
        
        Convenient for debugging or when scores are needed with texts.
        """
        results = self.retrieve(query, top_k, score_threshold, apply_fallback)
        return [(r.text, r.score) for r in results]
    
    def retrieve_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        apply_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced interface: returns full dict with metadata.
        
        Ideal for UI debug panels that need source information.
        
        Returns:
            List of dicts with keys: text, score, source, metadata, rank
        """
        results = self.retrieve(query, top_k, score_threshold, apply_fallback)
        return [r.to_dict() for r in results]
    
    def retrieve_one(
        self,
        query: str,
        min_score: float = 0.3,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve the single most relevant chunk.
        
        Args:
            query: User question
            min_score: Minimum acceptable score (returns None if below)
        
        Returns:
            Best result or None if below threshold
        """
        results = self.retrieve(query, top_k=1, score_threshold=min_score, apply_fallback=False)
        return results[0] if results else None
    
    def format_context(
        self,
        results: List[RetrievalResult],
        include_scores: bool = False,
        include_metadata: bool = False,
        mark_below_threshold: bool = True,
    ) -> str:
        """
        Format retrieval results into a string for LLM prompts.
        
        Args:
            results: List of RetrievalResult objects
            include_scores: If True, include similarity scores in output
            include_metadata: If True, include metadata (source, page, etc.)
            mark_below_threshold: If True, mark chunks that didn't meet threshold
        
        Returns:
            Formatted context string ready for prompt injection
        """
        if not results:
            return "No relevant documents found."
        
        formatted_parts = []
        
        for result in results:
            # Build header
            header_parts = [f"Document {result.rank}"]
            
            if include_scores:
                header_parts.append(f"relevance: {result.score:.3f}")
            
            if mark_below_threshold and not result.passed_threshold:
                header_parts.append("⚠️ LOW RELEVANCE")
            
            if include_metadata and result.metadata:
                # Show source prominently if available
                if result.source != 'Unknown':
                    header_parts.append(f"source: {result.source}")
                # Show other metadata
                meta_items = [f"{k}={v}" for k, v in result.metadata.items() 
                             if k not in ['source', 'source_file']]
                if meta_items:
                    header_parts.append(f"meta: {', '.join(meta_items[:2])}")
            
            header = "=== " + " | ".join(header_parts) + " ==="
            
            # Add content
            formatted_parts.append(f"{header}\n{result.text}")
        
        return self.config.context_separator.join(formatted_parts)
    
    def format_context_simple(self, results: List[RetrievalResult]) -> str:
        """
        Minimal context formatting: just chunks with separators.
        """
        if not results:
            return ""
        
        return self.config.context_separator.join([r.text for r in results])
    
    def get_debug_info(self, results: Optional[List[RetrievalResult]] = None) -> Dict[str, Any]:
        """
        Get debug information for the UI panel.
        
        Args:
            results: Optional results to analyze. If None, uses last retrieval.
        
        Returns:
            Dict with debug info: query, chunks, scores, sources, timings
        """
        if results is None:
            results = self._last_results
        
        if not results:
            return {
                "query": self._last_query or "",
                "chunks": [],
                "scores": [],
                "sources": [],
                "avg_score": 0.0,
                "timestamp": self._last_timestamp,
                "total_chunks": 0,
            }
        
        chunks = [r.text for r in results]
        scores = [r.score for r in results]
        sources = [r.source for r in results]
        
        return {
            "query": self._last_query or "",
            "chunks": chunks,
            "scores": scores,
            "sources": sources,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "timestamp": self._last_timestamp,
            "total_chunks": len(results),
            "passed_threshold": [r.passed_threshold for r in results],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever configuration and statistics."""
        return {
            "config": {
                "top_k": self.config.top_k,
                "score_threshold": self.config.score_threshold,
                "min_results": self.config.min_results,
                "include_scores": self.config.include_scores,
                "context_separator": repr(self.config.context_separator),
                "enable_threshold_filter": self.config.enable_threshold_filter,
            },
            "vector_store_stats": self.vector_store.get_stats(),
            "last_query": self._last_query,
            "last_results_count": len(self._last_results) if self._last_results else 0,
        }
    
    def update_config(self, **kwargs):
        """Update retriever configuration dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")


# ========== CONVENIENCE FUNCTIONS ==========

def create_retriever(
    vector_store,
    top_k: int = 5,                    # TASK 3: Optimized default
    score_threshold: float = 0.3,      # TASK 2: Quality gate default
    min_results: int = 2,
) -> Retriever:
    """
    Quick retriever creation with optimized parameters.
    
    Args:
        vector_store: VectorStore instance
        top_k: Number of chunks to retrieve (optimized: 5)
        score_threshold: Minimum similarity score (optimized: 0.3)
        min_results: Minimum results to return
    
    Returns:
        Configured Retriever instance
    """
    config = RetrieverConfig(
        top_k=top_k,
        score_threshold=score_threshold,
        min_results=min_results,
    )
    return Retriever(vector_store, config=config)


# ========== MODULE SELF-TEST ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Retriever Module - Tasks 2 & 3 Test")
    print("=" * 60)
    
    # Create test data with varying similarity scores
    import numpy as np
    from datetime import datetime
    
    class MockVectorStore:
        """Mock for testing threshold behavior"""
        def __init__(self):
            self.texts = [
                "Chunk 1: Very relevant content about RAG",
                "Chunk 2: Somewhat relevant content",
                "Chunk 3: Slightly related content",
                "Chunk 4: Weakly related content",
                "Chunk 5: Barely relevant content",
            ]
            self.scores = [0.85, 0.65, 0.45, 0.25, 0.15]
            self.sources = ["doc1.pdf", "doc1.pdf", "doc2.pdf", "doc2.pdf", "doc3.pdf"]
        
        def search(self, query_vec, top_k, score_threshold):
            from core.vector_store import SearchResult
            results = []
            for i, (text, score) in enumerate(zip(self.texts, self.scores)):
                results.append(SearchResult(
                    text=text,
                    score=score,
                    metadata={"source": self.sources[i], "chunk_index": i},
                    rank=i+1
                ))
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        
        def get_stats(self):
            return {"total_vectors": len(self.texts)}
    
    class MockEmbedder:
        def embed_single(self, text):
            return np.array([1.0])
    
    mock_store = MockVectorStore()
    mock_embedder = MockEmbedder()
    
    # Create retriever with optimized settings
    retriever = Retriever(
        mock_store,
        embedder=mock_embedder,
        config=RetrieverConfig(top_k=5, score_threshold=0.3, min_results=2)
    )
    
    print("\n📌 OPTIMIZED SETTINGS (Tasks 2 & 3):")
    print(f"   - Top-K: {retriever.config.top_k} (reduced for focus)")
    print(f"   - Threshold: {retriever.config.score_threshold} (quality gate)")
    print(f"   - Min Results: {retriever.config.min_results} (prevents empty)")
    
    # Test 1: Retrieval with both optimizations
    print("\n📝 Test 1: Retrieval with Top-K + Threshold")
    results = retriever.retrieve("test query")
    
    print(f"\n   Results returned: {len(results)}")
    for r in results:
        status = "✅" if r.passed_threshold else "⚠️"
        print(f"   {status} Score {r.score:.2f} | Source: {r.source} | {r.text[:40]}...")
    
    # Test 2: Enhanced metadata output for debug panel
    print("\n📝 Test 2: Enhanced metadata for debug panel")
    debug_info = retriever.get_debug_info(results)
    print(f"   Query: {debug_info['query']}")
    print(f"   Avg Score: {debug_info['avg_score']:.3f}")
    print(f"   Sources: {debug_info['sources']}")
    
    # Test 3: retrieve_with_metadata method
    print("\n📝 Test 3: retrieve_with_metadata() for UI panels")
    metadata_results = retriever.retrieve_with_metadata("test query")
    for r in metadata_results:
        print(f"   - {r['source']}: {r['score']:.3f} ({r['chunk_length']} chars)")
    
    # Test 4: Formatted context with enhanced source tracking
    print("\n📝 Test 4: Formatted context with source tracking")
    context = retriever.format_context(results, include_scores=True, include_metadata=True)
    print(f"\n{context[:400]}...")
    
    print("\n" + "=" * 60)
    print("✅ Retriever enhanced with:")
    print("   - Task 2: Similarity threshold filtering (quality gate)")
    print("   - Task 3: Top-K optimization (focus control)")
    print("   - Enhanced metadata for debug panel (source tracking)")
    print("   - Intelligent fallback (prevents empty responses)")
    print("   - get_debug_info() for UI integration")
    print("=" * 60)
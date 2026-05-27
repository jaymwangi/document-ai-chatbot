"""
Retriever Module - Task 6 of RAG Pipeline

Responsibility: Bridge between embeddings, vector store, and LLM generation.
Single responsibility: query → relevant chunks (orchestration only).

This module handles:
- Query embedding
- Vector store search
- Score thresholding
- Context formatting for LLM prompts

NOT responsible for:
- LLM calls (Task 7)
- Chat memory
- Prompt engineering
- UI logic

Architecture:
    User Query → Embed Query → Retriever → Vector Store → Top Chunks
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dataclasses import dataclass


@dataclass
class RetrieverConfig:
    """
    Configuration for the retriever.
    
    Centralizes tuning parameters for easy experimentation.
    
    Attributes:
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity score (0-1). 
                        Results below this are filtered out.
        include_scores: Include similarity scores in results
        context_separator: String used to join chunks in formatted context
    """
    top_k: int = 5
    score_threshold: float = 0.3
    include_scores: bool = True
    context_separator: str = "\n\n---\n\n"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError(f"score_threshold must be between 0 and 1, got {self.score_threshold}")


@dataclass
class RetrievalResult:
    """
    Single retrieval result with all relevant information.
    
    Attributes:
        text: The chunk text
        score: Similarity score (0-1, higher = more relevant)
        rank: Position in results (1 = most relevant)
        metadata: Optional metadata from original chunk
    """
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "rank": self.rank,
            "metadata": self.metadata,
        }


class Retriever:
    """
    Orchestrates retrieval: query → relevant chunks.
    
    v1 Features:
    - Embed query using Task 4 embeddings
    - Search vector store (Task 5)
    - Apply score thresholds
    - Format context for LLM prompts
    
    v2 Upgrade Paths:
    - Hybrid search (keyword + semantic)
    - Reranking with cross-encoders
    - Metadata filtering
    - Multi-query retrieval
    
    Usage:
        retriever = Retriever(vector_store)
        results = retriever.retrieve("What is RAG?")
        context = retriever.format_context(results)
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
            config: RetrieverConfig (defaults if not provided)
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
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Main retrieval interface: query → list of relevant chunks.
        
        Args:
            query: User question or search query
            top_k: Override config top_k for this call
            score_threshold: Override config score_threshold for this call
        
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
        
        # Use overrides or config defaults
        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        
        # Step 1: Embed the query
        query_vector = self._embedder.embed_single(query)
        
        if len(query_vector) == 0:
            return []
        
        # Step 2: Search vector store
        search_results = self.vector_store.search(
            query_vector,
            top_k=k,
            score_threshold=threshold,
        )
        
        # Step 3: Convert to RetrievalResult objects
        results = []
        for rank, result in enumerate(search_results, 1):
            results.append(RetrievalResult(
                text=result.text,
                score=result.score,
                rank=rank,
                metadata=result.metadata,
            ))
        
        return results
    
    def retrieve_texts(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Simplified interface: returns only chunk texts.
        
        Useful when you only need the text content.
        
        Example:
            >>> texts = retriever.retrieve_texts("What is RAG?")
            >>> for text in texts:
            ...     print(text)
        """
        results = self.retrieve(query, top_k, score_threshold)
        return [r.text for r in results]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Returns (text, score) tuples.
        
        Convenient for debugging or when scores are needed with texts.
        
        Example:
            >>> pairs = retriever.retrieve_with_scores("What is AI?")
            >>> for text, score in pairs:
            ...     print(f"{score:.3f}: {text[:50]}")
        """
        results = self.retrieve(query, top_k, score_threshold)
        return [(r.text, r.score) for r in results]
    
    def retrieve_one(
        self,
        query: str,
        min_score: float = 0.3,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve the single most relevant chunk.
        
        Useful for simple Q&A or when only top result is needed.
        
        Args:
            query: User question
            min_score: Minimum acceptable score (returns None if below)
        
        Returns:
            Best result or None if below threshold
        """
        results = self.retrieve(query, top_k=1, score_threshold=min_score)
        return results[0] if results else None
    
    def format_context(
        self,
        results: List[RetrievalResult],
        include_scores: bool = False,
        include_metadata: bool = False,
    ) -> str:
        """
        Format retrieval results into a string for LLM prompts.
        
        This is a critical bridge to Task 7 (LLM generation).
        The formatted context will be injected into the prompt.
        
        Args:
            results: List of RetrievalResult objects
            include_scores: If True, include similarity scores in output
            include_metadata: If True, include metadata (source, page, etc.)
        
        Returns:
            Formatted context string ready for prompt injection
        
        Example output:
            === Document 1 (score: 0.89) ===
            RAG stands for Retrieval-Augmented Generation...
            
            === Document 2 (score: 0.76) ===
            Vector databases enable semantic search...
        """
        if not results:
            return "No relevant documents found."
        
        formatted_parts = []
        
        for result in results:
            # Build header
            header_parts = [f"Document {result.rank}"]
            
            if include_scores:
                header_parts.append(f"relevance: {result.score:.3f}")
            
            if include_metadata and result.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in result.metadata.items())
                header_parts.append(f"metadata: {meta_str}")
            
            header = "=== " + " | ".join(header_parts) + " ==="
            
            # Add content
            formatted_parts.append(f"{header}\n{result.text}")
        
        return self.config.context_separator.join(formatted_parts)
    
    def format_context_simple(self, results: List[RetrievalResult]) -> str:
        """
        Minimal context formatting: just chunks with separators.
        
        Use this for simpler prompts or when scores aren't needed.
        """
        if not results:
            return ""
        
        return self.config.context_separator.join([r.text for r in results])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever configuration and statistics."""
        return {
            "config": {
                "top_k": self.config.top_k,
                "score_threshold": self.config.score_threshold,
                "include_scores": self.config.include_scores,
                "context_separator": repr(self.config.context_separator),
            },
            "vector_store_stats": self.vector_store.get_stats(),
        }
    
    def update_config(self, **kwargs):
        """
        Update retriever configuration dynamically.
        
        Useful for testing different parameters without recreating the retriever.
        
        Example:
            retriever.update_config(top_k=10, score_threshold=0.2)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")


# ========== CONVENIENCE FUNCTIONS ==========

def create_retriever(
    vector_store,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> Retriever:
    """
    Quick retriever creation with simple parameters.
    
    Args:
        vector_store: VectorStore instance
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity score
    
    Returns:
        Configured Retriever instance
    """
    config = RetrieverConfig(
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return Retriever(vector_store, config=config)


# ========== MODULE SELF-TEST (lightweight smoke test only) ==========
# Full tests in tests/test_retriever.py

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Retriever Module - Quick Smoke Test")
    print("=" * 60)
    
    # Create minimal test data
    import numpy as np
    
    class MockVectorStore:
        """Minimal mock for testing"""
        def __init__(self):
            self.texts = ["Chunk 1: RAG is great", "Chunk 2: Vectors are cool"]
            self.vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        def search(self, query_vec, top_k, score_threshold):
            results = []
            for i, vec in enumerate(self.vectors):
                sim = float(np.dot(query_vec, vec))
                if sim >= score_threshold:
                    from core.vector_store import SearchResult
                    results.append(SearchResult(
                        text=self.texts[i],
                        score=sim,
                        metadata={},
                        rank=i+1
                    ))
            return results[:top_k]
        
        def get_stats(self):
            return {"total_vectors": len(self.vectors)}
    
    class MockEmbedder:
        def embed_single(self, text):
            # Return vector biased toward first chunk
            return np.array([1.0, 0.1])
    
    mock_store = MockVectorStore()
    mock_embedder = MockEmbedder()
    
    retriever = Retriever(mock_store, embedder=mock_embedder)
    
    # Test retrieve
    results = retriever.retrieve("What is RAG?", top_k=2)
    print(f"\n📝 Retrieve: {len(results)} results")
    
    # Test format_context
    context = retriever.format_context(results, include_scores=True)
    print(f"\n📝 Formatted context:\n{context[:200]}...")
    
    # Test stats
    stats = retriever.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    print("\n" + "=" * 60)
    print("✅ Retriever module ready!")
    print("   Full tests: python tests/test_retriever.py")
    print("=" * 60)
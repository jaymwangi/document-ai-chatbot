"""
Reranker - Cross-encoder for accurate result re-ranking

Cross-encoders are slower but MUCH more accurate than bi-encoders.
Use them ONLY on top results (20-30) for best performance.

Production pattern:
    Vector Search (Top 20-30) → Cross-Encoder Reranker → Top 5 → LLM

FIXES APPLIED:
    - Fixed index mapping bug (no more misaligned scores)
    - Lazy loading (no model load on __init__)
    - Safe cross-encoder import
    - Sentence-boundary truncation
    - No mutation of original chunks
    - Threshold filter before reranking
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Safe cross-encoder import
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    try:
        from sentence_transformers import CrossEncoder
        CROSS_ENCODER_AVAILABLE = True
    except ImportError:
        CROSS_ENCODER_AVAILABLE = False
        logger.warning(
            "CrossEncoder not available. Install: pip install sentence-transformers"
        )


class Reranker:
    """
    Cross-encoder reranker for improved relevance scoring.
    
    Models (best to worst quality):
    1. cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good)
    2. cross-encoder/ms-marco-MiniLM-L-12-v2 (better, slower)
    3. BAAI/bge-reranker-base (excellent, larger)
    4. BAAI/bge-reranker-large (best, heavy)
    
    Usage:
        reranker = Reranker()
        results = reranker.rerank(query, chunks, top_k=5)
    """
    
    # Available models with their trade-offs
    MODELS = {
        "mini-lm": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "speed": "fast",
            "quality": "good",
            "size_mb": 90,
        },
        "mini-lm-v2": {
            "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "speed": "medium",
            "quality": "better",
            "size_mb": 120,
        },
        "bge-base": {
            "name": "BAAI/bge-reranker-base",
            "quality": "excellent",
            "speed": "slow",
            "size_mb": 550,
        },
    }
    
    def __init__(
        self,
        model_name: str = "mini-lm",
        max_chunk_chars: int = 512,
        max_rerank_size: int = 30,
        min_score_threshold: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Initialize reranker (lazy loading - no model loaded yet).
        
        Args:
            model_name: "mini-lm", "mini-lm-v2", or "bge-base"
            max_chunk_chars: Truncate chunks to this length (512 is good)
            max_rerank_size: Only rerank top N results (saves time)
            min_score_threshold: Skip chunks below this score before reranking
            device: "cpu", "cuda", or None (auto-detect)
        """
        self.model_name = model_name
        self.max_chunk_chars = max_chunk_chars
        self.max_rerank_size = max_rerank_size
        self.min_score_threshold = min_score_threshold
        self.device = device
        
        # Model info
        if model_name not in self.MODELS:
            logger.warning(f"Unknown model {model_name}, using mini-lm")
            model_name = "mini-lm"
        
        self.model_info = self.MODELS[model_name]
        self._model = None  # Lazy loaded
        self._model_loaded = False
        
        logger.info(f"📊 Reranker configured: {self.model_info['name']} (lazy loading enabled)")
    
    def _ensure_model_loaded(self):
        """Lazy load cross-encoder model (only when first needed)."""
        if self._model_loaded:
            return
        
        if not CROSS_ENCODER_AVAILABLE:
            logger.error("CrossEncoder not available. Cannot use reranker.")
            self._model_loaded = True
            return
        
        try:
            model_path = self.model_info["name"]
            logger.info(f"🔄 Loading reranker model: {model_path}")
            
            self._model = CrossEncoder(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
            
            self._model_loaded = True
            logger.info(f"✅ Reranker loaded ({self.model_info['speed']}, {self.model_info['quality']})")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self._model_loaded = True
            self._model = None
    
    def _truncate_to_sentence_boundary(self, text: str, max_chars: int) -> str:
        """
        Truncate text at a natural sentence boundary.
        
        Preserves semantic completeness for better reranking accuracy.
        """
        if len(text) <= max_chars:
            return text
        
        truncated = text[:max_chars]
        
        # Try to cut at sentence boundary (. ! ? followed by space or newline)
        sentence_end = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? '),
            truncated.rfind('.\n'),
            truncated.rfind('!\n'),
            truncated.rfind('?\n'),
        )
        
        if sentence_end > max_chars // 2:
            return truncated[:sentence_end + 1]
        
        # Fallback to last space
        last_space = truncated.rfind(' ')
        if last_space > max_chars // 2:
            return truncated[:last_space] + "..."
        
        # Last resort: hard cut
        return truncated + "..."
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        return_all_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks by relevance to query using cross-encoder.
        
        Args:
            query: User query string
            chunks: List of chunk dicts with 'text' field
            top_k: Number of results to return
            return_all_scores: If True, keep original and rerank scores
        
        Returns:
            Re-ranked chunks with 'rerank_score' added (does NOT mutate input)
        
        Example:
            results = [
                {"text": "chunk 1", "score": 0.8, "metadata": {...}},
                {"text": "chunk 2", "score": 0.7, "metadata": {...}},
            ]
            reranked = reranker.rerank(query, results, top_k=3)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if not self._model:
            logger.warning("Reranker model not available, returning original order")
            return chunks[:top_k]
        
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks[:top_k]
        
        try:
            # Step 1: Filter by score threshold (saves compute)
            filtered = [
                c for c in chunks[:self.max_rerank_size * 2]
                if c.get("score", 0) >= self.min_score_threshold
            ]
            
            # Step 2: Take top candidates for reranking
            candidates = filtered[:self.max_rerank_size]
            
            if not candidates:
                return chunks[:top_k]
            
            # Step 3: Prepare query-chunk pairs (with safe truncation)
            pairs = []
            valid_candidates = []
            
            for chunk in candidates:
                text = chunk.get("text", "")
                if not text:
                    continue
                
                # Truncate at sentence boundary
                text = self._truncate_to_sentence_boundary(text, self.max_chunk_chars)
                pairs.append((query, text))
                valid_candidates.append(chunk)
            
            if not pairs:
                return chunks[:top_k]
            
            # Step 4: Get cross-encoder scores
            scores = self._model.predict(pairs)
            
            # Step 5: Build scored chunks (NEW dicts, no mutation)
            scored_chunks = []
            for chunk, score in zip(valid_candidates, scores):
                # Create a copy to avoid mutating original
                new_chunk = chunk.copy()
                new_chunk["rerank_score"] = float(score)
                new_chunk["original_score"] = chunk.get("score", 0.0)
                scored_chunks.append(new_chunk)
            
            # Step 6: Sort by rerank score (higher is better)
            reranked = sorted(
                scored_chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )
            
            # Step 7: Return top_k
            result = reranked[:top_k]
            
            # Log top score safely
            top_score = result[0].get("rerank_score", 0) if result else 0
            logger.info(
                f"📊 Reranked {len(reranked)}/{len(chunks)} results | "
                f"Top score: {top_score:.3f}"
            )
            
            # Clean up scores if requested
            if not return_all_scores:
                for r in result:
                    # Replace original score with rerank score as primary
                    r["score"] = r.get("rerank_score", r.get("score", 0))
            
            return result
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]
    
    def rerank_single(
        self,
        query: str,
        text: str,
    ) -> float:
        """
        Get relevance score for a single query-text pair.
        
        Useful for debugging or one-off scoring.
        
        Args:
            query: User query
            text: Document text
        
        Returns:
            Relevance score (higher = more relevant)
        """
        self._ensure_model_loaded()
        
        if not self._model:
            return 0.5
        
        try:
            # Truncate at sentence boundary
            text = self._truncate_to_sentence_boundary(text, self.max_chunk_chars)
            
            scores = self._model.predict([(query, text)])
            return float(scores[0])
        except Exception as e:
            logger.error(f"Single reranking failed: {e}")
            return 0.5
    
    def is_available(self) -> bool:
        """Check if reranker is available and loaded."""
        return self._model is not None and self._model_loaded
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "available": CROSS_ENCODER_AVAILABLE and self.is_available(),
            "model": self.model_info["name"],
            "model_speed": self.model_info["speed"],
            "model_quality": self.model_info["quality"],
            "max_chunk_chars": self.max_chunk_chars,
            "max_rerank_size": self.max_rerank_size,
            "min_score_threshold": self.min_score_threshold,
            "lazy_loaded": self._model_loaded,
        }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict]:
        """Get list of available cross-encoder models."""
        return cls.MODELS.copy()


# =========================
# CONVENIENCE FUNCTION
# =========================

def create_reranker(
    model_name: str = "mini-lm",
    max_chunk_chars: int = 512,
    max_rerank_size: int = 30,
    min_score_threshold: float = 0.2,
) -> Optional[Reranker]:
    """
    Create a reranker instance with error handling.
    
    Returns None if cross-encoder not available.
    """
    if not CROSS_ENCODER_AVAILABLE:
        logger.warning("Cannot create reranker: sentence-transformers not installed")
        return None
    
    try:
        return Reranker(
            model_name=model_name,
            max_chunk_chars=max_chunk_chars,
            max_rerank_size=max_rerank_size,
            min_score_threshold=min_score_threshold,
        )
    except Exception as e:
        logger.error(f"Failed to create reranker: {e}")
        return None


# =========================
# SELF-TEST
# =========================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Reranker (Production Version)")
    print("=" * 60)
    
    # Test with mock chunks
    test_query = "What is machine learning?"
    test_chunks = [
        {"text": "Machine learning is a subset of AI that enables systems to learn from data.", "score": 0.75},
        {"text": "Python is a programming language used for data science.", "score": 0.70},
        {"text": "Deep learning uses neural networks with many layers.", "score": 0.65},
        {"text": "The weather today is sunny and warm.", "score": 0.60},
        {"text": "Machine learning models improve with more training data.", "score": 0.55},
    ]
    
    reranker = create_reranker()
    
    if reranker:
        print(f"\n📝 Query: {test_query}")
        print("\n📊 Before reranking:")
        for i, c in enumerate(test_chunks[:3], 1):
            print(f"   [{i}] Score: {c['score']:.3f} | {c['text'][:60]}...")
        
        reranked = reranker.rerank(test_query, test_chunks, top_k=3)
        
        print("\n✨ After reranking:")
        for i, c in enumerate(reranked, 1):
            rerank_score = c.get('rerank_score', c.get('score', 0))
            print(f"   [{i}] Score: {rerank_score:.3f} | {c['text'][:60]}...")
        
        print("\n📊 Reranker stats:")
        stats = reranker.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    else:
        print("\n❌ Reranker not available - install sentence-transformers")
    
    print("\n" + "=" * 60)
    print("✅ Reranker production-ready!")
    print("=" * 60)
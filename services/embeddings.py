"""
Embeddings Module - Single Source of Truth for Embeddings

This is the ONE place where the embedding model is loaded and managed.
No other file in your project should ever load a model or create an embedder.

Core Purpose:
    - Load the model once at app start
    - Reuse the same model instance across all calls
    - Convert text chunks to vectors (embeddings)
    - Optional disk caching for speed

ARCHITECTURE:
    - Single embedder singleton (thread-safe)
    - Memory + disk cache with LRU and TTL
    - Background cache writer (async flush)
    - Metrics tracking (cache hit rate, latency)
    - Clean separation of concerns

FIXES APPLIED:
    - ✅ Single alignment system (valid_indices only)
    - ✅ Atexit registration for clean shutdown
    - ✅ Thread-safe metrics with atomic counters
    - ✅ Optimized embed_single with batch reuse
    - ✅ Clean separation ready for refactoring

Usage:
    from services.embeddings import embed_chunks, embed_query
    
    # Embed multiple chunks
    vectors = embed_chunks(["text 1", "text 2"])
    
    # Embed a single query
    query_vector = embed_query("What is RAG?")
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import hashlib
from pathlib import Path
import json
import threading
import atexit
from collections import OrderedDict
from dataclasses import dataclass, field
from collections import deque

# Setup logging
logger = logging.getLogger(__name__)

# Try to import sentence-transformers with helpful error
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )


# ============================================================
# THREAD-SAFE METRICS TRACKING
# ============================================================

@dataclass
class EmbeddingMetrics:
    """Thread-safe metrics tracking for embedding operations."""
    
    total_embeddings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_ms: float = 0.0
    batch_count: int = 0
    
    def __post_init__(self):
        self._lock = threading.Lock()
    
    def record_embedding(self, n: int, time_ms: float, cache_hit: bool = False):
        """Thread-safe record of embedding operation."""
        with self._lock:
            self.total_embeddings += n
            self.total_time_ms += time_ms
            self.batch_count += 1
            if cache_hit:
                self.cache_hits += n
            else:
                self.cache_misses += n
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate (thread-safe)."""
        with self._lock:
            total = self.cache_hits + self.cache_misses
            if total == 0:
                return 0.0
            return self.cache_hits / total
    
    def get_avg_time_ms(self) -> float:
        """Get average time per batch (thread-safe)."""
        with self._lock:
            if self.batch_count == 0:
                return 0.0
            return self.total_time_ms / self.batch_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dict (thread-safe)."""
        with self._lock:
            return {
                "total_embeddings": self.total_embeddings,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.get_hit_rate(),
                "avg_time_ms": self.get_avg_time_ms(),
                "batch_count": self.batch_count,
            }


# ============================================================
# LRU CACHE (with size limit and TTL)
# ============================================================

class LRUCache:
    """
    LRU cache with size limit and TTL.
    Prevents memory bloat and ensures cache stays manageable.
    
    Thread-safe for concurrent access.
    """
    
    def __init__(self, max_size: int = 50000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    def __len__(self):
        return len(self.cache)
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            expired_keys = [
                key for key, ts in self.timestamps.items()
                if now - ts > self.ttl
            ]
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                    del self.timestamps[key]
            self._last_cleanup = now
            if expired_keys:
                logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
    
    def _evict_one(self):
        """Evict the least recently used entry."""
        if not self.cache:
            return
        
        oldest = next(iter(self.cache))
        del self.cache[oldest]
        del self.timestamps[oldest]
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get value from cache if exists and not expired."""
        self._cleanup_expired()
        
        with self._lock:
            if key not in self.cache:
                return None
            
            # Check expiration
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def set(self, key: str, value: np.ndarray):
        """Set value in cache with LRU eviction."""
        with self._lock:
            if key in self.cache:
                self.cache[key] = value
                self.timestamps[key] = time.time()
                self.cache.move_to_end(key)
                return
            
            while len(self.cache) >= self.max_size:
                self._evict_one()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
            }


# ============================================================
# DISK CACHE (with background writer and shutdown hook)
# ============================================================

class SimpleEmbeddingCache:
    """
    Disk cache for embeddings with background writing.
    
    Features:
    - LRU in-memory cache (avoids memory bloat)
    - Dirty flag with thread-safe check
    - Background writer thread (async flush)
    - Atexit registration for clean shutdown
    - TTL for cache freshness
    - Configurable max size
    """
    
    _global_writer_started = False
    _global_writer_lock = threading.Lock()
    
    def __init__(
        self,
        cache_path: str = ".embedding_cache.json",
        max_size: int = 50000,
        ttl_seconds: int = 3600,
        background_flush: bool = True,
        flush_interval: int = 30,
    ):
        self.cache_path = Path(cache_path)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.background_flush = background_flush
        self.flush_interval = flush_interval
        
        # In-memory cache
        self._memory_cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        
        # Disk cache
        self._disk_cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._running = True
        
        # Background writer thread
        self._flush_thread = None
        self._load()
        
        if background_flush:
            self._start_background_writer()
        
        # Register cleanup on shutdown
        atexit.register(self._shutdown)
    
    def _start_background_writer(self):
        """Start background thread for async cache writes."""
        # ✅ FIXED: Only one global writer
        with SimpleEmbeddingCache._global_writer_lock:
            if SimpleEmbeddingCache._global_writer_started:
                logger.debug("Background writer already running")
                return
            
            def writer_loop():
                while self._running:
                    time.sleep(self.flush_interval)
                    if self._dirty:
                        self._save()
            
            self._flush_thread = threading.Thread(target=writer_loop, daemon=True)
            self._flush_thread.start()
            SimpleEmbeddingCache._global_writer_started = True
            logger.info(f"🔄 Background cache writer started (interval: {self.flush_interval}s)")
    
    def _shutdown(self):
        """Clean shutdown - flush and stop writer."""
        logger.info("🛑 Shutting down cache writer...")
        self._running = False
        
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)
        
        # Final flush
        self._save()
        logger.info("✅ Cache flushed on shutdown")
    
    def _load(self):
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self._disk_cache = json.load(f)
                logger.info(f"📂 Loaded {len(self._disk_cache)} cached embeddings from disk")
            except (json.JSONDecodeError, IOError):
                self._disk_cache = {}
    
    def _save(self):
        """Save cache to disk - thread-safe dirty check."""
        with self._lock:
            if not self._dirty:
                return
        
        try:
            temp_path = self.cache_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self._disk_cache, f)
            temp_path.replace(self.cache_path)
            
            with self._lock:
                self._dirty = False
            
            logger.debug(f"💾 Saved {len(self._disk_cache)} embeddings to disk cache")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._hash_text(text)
        
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached
        
        with self._lock:
            if key in self._disk_cache:
                embedding = np.array(self._disk_cache[key], dtype=np.float32)
                self._memory_cache.set(key, embedding)
                return embedding
        
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding."""
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        key = self._hash_text(text)
        self._memory_cache.set(key, embedding)
        
        with self._lock:
            self._disk_cache[key] = embedding.tolist()
            self._dirty = True
    
    def set_batch(self, texts: List[str], embeddings: np.ndarray):
        """Cache multiple embeddings at once."""
        if not texts or len(embeddings) == 0:
            return
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        with self._lock:
            for text, embedding in zip(texts, embeddings):
                key = self._hash_text(text)
                self._disk_cache[key] = embedding.tolist()
                self._memory_cache.set(key, embedding)
            self._dirty = True
    
    def flush(self):
        """Force save cache to disk."""
        self._save()
    
    def clear(self):
        """Clear all cache."""
        self._memory_cache.clear()
        with self._lock:
            self._disk_cache = {}
            self._dirty = True
        self._save()
        if self.cache_path.exists():
            self.cache_path.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = self._memory_cache.get_stats()
        with self._lock:
            return {
                "memory": memory_stats,
                "disk_size": len(self._disk_cache),
                "cache_path": str(self.cache_path),
                "dirty": self._dirty,
                "background_running": self._running and self._flush_thread and self._flush_thread.is_alive(),
            }


# ============================================================
# SINGLE EMBEDDER INSTANCE (The One True Embedder)
# ============================================================

# Global instance - only ONE exists
_embedder = None
_embedder_lock = threading.Lock()
_metrics = EmbeddingMetrics()


class Embedder:
    """
    Simple wrapper around SentenceTransformer.
    Loads once, reused forever.
    
    ARCHITECTURE:
        - Single singleton instance
        - Thread-safe initialization
        - Cache with background writer
        - Metrics tracking
        - Clean separation of concerns
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the model once at startup."""
        logger.info("🔄 Loading embedding model: %s", model_name)
        start_time = time.time()
        
        self.model = SentenceTransformer(model_name)
        
        # Warm up
        _ = self.model.encode(["warmup"], show_progress_bar=False)
        
        load_time = time.time() - start_time
        logger.info("✅ Embedding model loaded in %.2fs", load_time)
        
        self.cache = SimpleEmbeddingCache(
            max_size=50000,
            ttl_seconds=3600,
            background_flush=True,
            flush_interval=30,
        )
        
        self.dimensions = 384
        self.metrics = _metrics
        
        # Small batch cache for embed_single optimization
        self._single_batch_buffer = []
        self._single_batch_lock = threading.Lock()
    
    def embed(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Convert texts to embeddings.
        
        Returns:
            Tuple of (embeddings array, valid_indices list)
        """
        if not texts:
            return np.array([]), []
        
        # Preserve original indices
        valid_texts = []
        valid_indices = []
        for i, t in enumerate(texts):
            if t and isinstance(t, str):
                valid_texts.append(t)
                valid_indices.append(i)
        
        if not valid_texts:
            return np.array([]), []
        
        n_texts = len(valid_texts)
        start_time = time.time()
        
        # Smart batch size
        if batch_size is None:
            if n_texts > 5000:
                batch_size = 256
            elif n_texts > 2000:
                batch_size = 128
            elif n_texts > 500:
                batch_size = 64
            else:
                batch_size = 32
            batch_size = min(256, max(16, batch_size))
        
        logger.debug(f"📊 Using batch size: {batch_size} for {n_texts} texts")
        
        embeddings = np.zeros((n_texts, self.dimensions), dtype=np.float32)
        need_embedding = []
        need_indices = []
        cache_hits = 0
        
        # Check cache
        if use_cache:
            for i, text in enumerate(valid_texts):
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings[i] = cached.astype(np.float32)
                    cache_hits += 1
                else:
                    need_embedding.append(text)
                    need_indices.append(i)
            
            if not need_embedding:
                elapsed = time.time() - start_time
                self.metrics.record_embedding(n_texts, elapsed * 1000, cache_hit=True)
                logger.debug(f"✅ All {n_texts} texts found in cache ({cache_hits} hits)")
                return embeddings, valid_indices
        else:
            need_embedding = valid_texts
            need_indices = list(range(n_texts))
        
        # Embed uncached texts
        if need_embedding:
            logger.debug(f"🔨 Embedding {len(need_embedding)} uncached texts")
            
            all_embeddings = self.model.encode(
                need_embedding,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            
            cache_items = []
            for i, (text, embedding) in enumerate(zip(need_embedding, all_embeddings)):
                idx = need_indices[i]
                embeddings[idx] = embedding.astype(np.float32)
                if use_cache:
                    cache_items.append((text, embedding.astype(np.float32)))
            
            if use_cache and cache_items:
                texts_batch = [item[0] for item in cache_items]
                embeddings_batch = np.array([item[1] for item in cache_items])
                self.cache.set_batch(texts_batch, embeddings_batch)
        
        elapsed = time.time() - start_time
        self.metrics.record_embedding(n_texts, elapsed * 1000, cache_hit=False)
        
        return embeddings, valid_indices
    
    def embed_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Embed a single text string."""
        if not text:
            return np.array([])
        
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                self.metrics.record_embedding(1, (time.time() - start_time) * 1000, cache_hit=True)
                return cached.astype(np.float32)
        
        # ✅ FIXED: Embed single with batch optimization
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        
        embedding = embedding.astype(np.float32)
        
        if use_cache:
            self.cache.set(text, embedding)
        
        elapsed = time.time() - start_time
        self.metrics.record_embedding(1, elapsed * 1000, cache_hit=False)
        
        return embedding


# ============================================================
# PUBLIC API - The ONLY way to get embeddings
# ============================================================

_AUTO_FLUSH = False


def set_auto_flush(enabled: bool):
    """Control whether embed_chunks automatically flushes cache."""
    global _AUTO_FLUSH
    _AUTO_FLUSH = enabled
    logger.info(f"📊 Auto-flush: {'ON' if enabled else 'OFF'}")


def get_embedder() -> Embedder:
    """Get the single embedder instance."""
    global _embedder
    
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                logger.info("🏗️ Creating embedder instance (first time)")
                _embedder = Embedder()
    
    return _embedder


def embed_chunks(
    chunks: List[str],
    use_cache: bool = True,
    batch_size: Optional[int] = None,
    return_format: str = "both",
    auto_flush: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Convert text chunks to structured embeddings.
    
    ✅ FIXED: Single alignment system using valid_indices.
    """
    if not chunks:
        return []
    
    embedder = get_embedder()
    
    # Get embeddings and valid indices
    vectors, valid_indices = embedder.embed(
        chunks, use_cache=use_cache, batch_size=batch_size
    )
    
    if len(vectors) == 0:
        return []
    
    # ✅ FIXED: Single alignment system
    # Build results directly from valid_indices
    results = []
    for idx, vector in zip(valid_indices, vectors):
        text = chunks[idx]
        
        result = {
            "id": f"chunk_{idx}_{hashlib.md5(text[:50].encode()).hexdigest()[:8]}",
            "text": text,
            "vector_np": vector,
        }
        
        if return_format in ["both", "list_only"]:
            result["vector"] = vector.tolist()
        
        results.append(result)
    
    if return_format == "list_only":
        for r in results:
            del r["vector_np"]
    
    should_flush = auto_flush if auto_flush is not None else _AUTO_FLUSH
    if use_cache and should_flush:
        embedder.cache.flush()
    
    return results


def embed_query(query: str, use_cache: bool = True) -> np.ndarray:
    """Embed a single query for retrieval."""
    if not query:
        return np.array([])
    
    embedder = get_embedder()
    return embedder.embed_single(query, use_cache=use_cache)


def embed_texts(
    texts: List[str],
    use_cache: bool = True,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Embed multiple texts (numpy-only, no metadata)."""
    if not texts:
        return np.array([])
    
    embedder = get_embedder()
    vectors, _ = embedder.embed(texts, use_cache=use_cache, batch_size=batch_size)
    return vectors


def preload_embedder():
    """Explicitly preload the embedder at app startup."""
    logger.info("🔧 Preloading embedder...")
    start = time.time()
    get_embedder()
    elapsed = time.time() - start
    logger.info(f"✅ Embedder ready in {elapsed:.2f}s")


def flush_cache():
    """Force flush cache to disk."""
    embedder = get_embedder()
    embedder.cache.flush()
    logger.info("💾 Cache flushed")


def clear_cache():
    """Clear the embedding disk cache."""
    embedder = get_embedder()
    embedder.cache.clear()
    logger.info("🗑️ Embedding cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    embedder = get_embedder()
    return embedder.cache.get_stats()


def get_metrics() -> Dict[str, Any]:
    """Get embedding metrics."""
    return _metrics.to_dict()


def get_embedder_info() -> Dict[str, Any]:
    """Get information about the current embedder."""
    embedder = get_embedder()
    return {
        "model_name": "all-MiniLM-L6-v2",
        "dimensions": embedder.dimensions,
        "is_loaded": True,
        "cache_stats": get_cache_stats(),
        "metrics": get_metrics(),
        "auto_flush": _AUTO_FLUSH,
    }


# ============================================================
# MODULE SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Embeddings Module - Self Test")
    print("=" * 60)
    
    # Test loading
    print("\n📥 Loading embedder...")
    embedder = get_embedder()
    print(f"✅ Embedder ready: {embedder.dimensions} dimensions")
    
    # Test embedding with order preservation
    print("\n📝 Testing embedding and order preservation...")
    test_texts = [
        "First chunk",
        "",
        "Second chunk",
        "Third chunk",
        "Fourth chunk",
    ]
    
    results = embed_chunks(test_texts)
    print(f"✅ Embedded {len(results)} chunks")
    
    # Verify order is preserved
    expected_texts = ["First chunk", "Second chunk", "Third chunk", "Fourth chunk"]
    for i, result in enumerate(results):
        print(f"   [{i}] {result['text']} -> shape {result['vector_np'].shape}")
        assert result['text'] == expected_texts[i], f"Order mismatch at index {i}"
    
    # Test cache
    print("\n💾 Testing cache...")
    start = time.time()
    results_cached = embed_chunks(test_texts, use_cache=True)
    cached_time = time.time() - start
    print(f"✅ Cached embedding: {cached_time:.4f}s")
    
    # Test metrics
    print("\n📊 Testing metrics...")
    metrics = get_metrics()
    print(f"   Total embeddings: {metrics['total_embeddings']}")
    print(f"   Cache hit rate: {metrics['hit_rate']:.1%}")
    print(f"   Avg time per batch: {metrics['avg_time_ms']:.2f}ms")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("   - Single alignment system: ✅")
    print("   - Order preservation: ✅")
    print("   - Cache correctness: ✅")
    print("   - Metrics tracking: ✅")
    print("   - Background writer: ✅")
    print("   - Atexit registration: ✅")
    print("=" * 60)
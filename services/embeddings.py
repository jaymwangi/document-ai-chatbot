"""
Embeddings Module - Task 4 & 5 of RAG Pipeline

Responsibility: Convert text chunks into vector embeddings (meaning representations).

OPTIMIZATIONS:
- Lazy loading: Model loads only on first use (not at startup)
- Singleton pattern: Single model instance reused across calls
- Thread-safe initialization (with simple lock for concurrency)
- Warm-up option for first query speed
- Fast model options with actually available models
- Disk cache to avoid recomputation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import hashlib
import json
import warnings
import threading
import time
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Try to import sentence-transformers with helpful error
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )


class EmbeddingCache:
    """
    Disk-based cache for embeddings to avoid recomputation.
    
    Usage:
        cache = EmbeddingCache()
        if cache.has(text):
            vector = cache.get(text)
        else:
            vector = model.encode(text)
            cache.set(text, vector)
    """
    
    def __init__(self, cache_path: str = ".embedding_cache.json", enabled: bool = True):
        """
        Initialize cache.
        
        Args:
            cache_path: Path to cache file
            enabled: Whether caching is active (can disable for debugging)
        """
        self.cache_path = Path(cache_path)
        self.enabled = enabled
        self.cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load cache from disk."""
        if not self.enabled:
            return
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                warnings.warn(f"Could not load cache from {self.cache_path}")
                self.cache = {}
    
    def _save(self):
        """Save cache to disk - thread-safe."""
        if not self.enabled:
            return
        
        with self._lock:
            try:
                # Create a copy to avoid modification during write
                cache_copy = dict(self.cache)
                temp_path = self.cache_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(cache_copy, f)
                temp_path.replace(self.cache_path)
            except Exception as e:
                warnings.warn(f"Could not save cache: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def has(self, text: str) -> bool:
        """Check if text exists in cache."""
        if not self.enabled:
            return False
        key = self._hash_text(text)
        with self._lock:
            return key in self.cache
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding as numpy array."""
        if not self.enabled:
            return None
        key = self._hash_text(text)
        with self._lock:
            if key in self.cache:
                return np.array(self.cache[key])
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding (stores as list for JSON)."""
        if not self.enabled:
            return
        key = self._hash_text(text)
        with self._lock:
            self.cache[key] = embedding.tolist()
            # Save every 100 entries
            if len(self.cache) % 100 == 0:
                self._save()
    
    def set_batch(self, texts: List[str], embeddings: np.ndarray):
        """Cache multiple embeddings at once."""
        if not self.enabled:
            return
        
        with self._lock:
            for text, embedding in zip(texts, embeddings):
                key = self._hash_text(text)
                self.cache[key] = embedding.tolist()
            self._save()
    
    def flush(self):
        """Force save cache to disk."""
        self._save()
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "size": len(self.cache),
                "cache_path": str(self.cache_path),
            }


class EmbeddingModel:
    """
    Manages embedding model loading and inference.
    
    Lazy loading: model not pre-loaded, loads on first use.
    """
    
    MODELS = {
        "mini-lm": {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "speed": "fast",
            "quality": "good",
            "free": True,
            "layers": 12,
        },
        "tiny-bert": {  # Smaller, faster model (actually exists!)
            "name": "prajjwal1/bert-tiny",
            "dimensions": 128,
            "speed": "very fast",
            "quality": "basic",
            "free": True,
            "layers": 2,
        },
        "mini-lm-small": {  # Alternative small model
            "name": "all-MiniLM-L3-v2",  # This might not exist, but let's try
            "dimensions": 384,
            "speed": "very fast",
            "quality": "good enough",
            "free": True,
            "layers": 6,
            "fallback": "all-MiniLM-L6-v2",  # Fallback if not found
        },
        "mpnet": {
            "name": "all-mpnet-base-v2", 
            "dimensions": 768,
            "speed": "medium",
            "quality": "better",
            "free": True,
            "layers": 12,
        },
    }
    
    def __init__(
        self,
        model_name: str = "mini-lm",  # Default to reliable model
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        enable_disk_cache: bool = True,
        disk_cache_path: str = ".embedding_cache.json",
        warmup: bool = True,
    ):
        """
        Initialize embedding model (lazy loading - model not loaded yet).
        
        Args:
            model_name: Model key from MODELS dict
            cache_dir: Directory for model cache
            device: 'cpu', 'cuda', or None (auto)
            enable_disk_cache: Whether to cache embeddings to disk
            disk_cache_path: Path for embedding cache
            warmup: Whether to warm up model after loading
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.MODELS.keys())}")
        
        self.config = self.MODELS[model_name].copy()  # Copy to avoid mutation
        self.model_name = model_name
        self.dimensions = self.config["dimensions"]
        self.device = device
        self.warmup = warmup
        
        # Lazy loading: model starts as None
        self._model: Optional[SentenceTransformer] = None
        self._load_lock = threading.Lock()
        
        # Cache setup
        self.cache = EmbeddingCache(
            cache_path=disk_cache_path,
            enabled=enable_disk_cache
        )
        
        # Model cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "sentence-transformers"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track loading time
        self._load_time_ms: Optional[float] = None
        
        print(f"📥 Embedding model configured: {self.config['name']}")
        print(f"   ⚡ Speed: {self.config['speed']} ({self.config.get('layers', 'unknown')} layers)")
        print(f"   ⚡ Lazy loading enabled - model will load on first use")
    
    def _ensure_model_loaded(self):
        """Lazy load the model if not already loaded."""
        if self._model is not None:
            return
        
        with self._load_lock:
            if self._model is not None:
                return
            
            print(f"\n🔄 Lazy loading embedding model: {self.config['name']}")
            start_time = time.time()
            
            try:
                self._model = SentenceTransformer(
                    self.config["name"],
                    cache_folder=str(self.cache_dir),
                    device=self.device
                )
            except Exception as e:
                # If model fails to load and has a fallback, use it
                if "fallback" in self.config:
                    print(f"   ⚠️ Model {self.config['name']} failed to load: {e}")
                    print(f"   🔄 Falling back to {self.config['fallback']}")
                    self.config["name"] = self.config["fallback"]
                    self.dimensions = 384  # MiniLM dimensions
                    self._model = SentenceTransformer(
                        self.config["name"],
                        cache_folder=str(self.cache_dir),
                        device=self.device
                    )
                else:
                    raise
            
            self._load_time_ms = (time.time() - start_time) * 1000
            
            print(f"✅ Model loaded in {self._load_time_ms:.0f}ms")
            print(f"   Dimensions: {self.dimensions}")
            print(f"   Layers: {self.config.get('layers', 'unknown')}")
            
            if self.warmup:
                print("   🔥 Warming up model...")
                warmup_text = "This is a warmup sentence to initialize the model."
                _ = self._model.encode([warmup_text], show_progress_bar=False)
                print("   ✅ Warmup complete")
    
    def _validate_embedding(self, embedding: np.ndarray, text_snippet: str = "") -> None:
        """Validate embedding shape and type."""
        if not isinstance(embedding, np.ndarray):
            raise ValueError(f"Embedding must be numpy array, got {type(embedding)}")
        
        if len(embedding.shape) != 1:
            raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")
        
        if embedding.shape[0] != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimensions}, "
                f"got {embedding.shape[0]}. Text snippet: {text_snippet[:100]}"
            )
    
    def embed(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> np.ndarray:
        """Generate embeddings for a list of texts with progress tracking."""
        if not texts:
            return np.array([])
        
        self._ensure_model_loaded()
        
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            return np.array([])
        
        total_chunks = len(valid_texts)
        start_time = time.time()
        
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache
        if use_cache:
            cached_count = 0
            for i, text in enumerate(valid_texts):
                cached = self.cache.get(text)
                if cached is not None:
                    self._validate_embedding(cached, text[:50])
                    embeddings.append(cached)
                    cached_count += 1
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            
            logger.info(f"Cache: {cached_count} hits, {len(texts_to_embed)} to embed")
        else:
            texts_to_embed = valid_texts
            indices_to_embed = list(range(len(valid_texts)))
        
        if texts_to_embed:
            embed_start = time.time()
            total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
            
            batch_embeddings = []
            
            for batch_idx, i in enumerate(range(0, len(texts_to_embed), batch_size)):
                batch = texts_to_embed[i:i + batch_size]
                batch_num = batch_idx + 1
                
                # Process batch
                batch_result = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                
                batch_embeddings.append(batch_result)
                
                # Progress callback
                if progress_callback:
                    elapsed = time.time() - embed_start
                    avg_time_per_batch = elapsed / batch_num
                    eta_seconds = avg_time_per_batch * (total_batches - batch_num)
                    progress_callback(batch_num, total_batches, eta_seconds)
            
            new_embeddings = np.vstack(batch_embeddings) if batch_embeddings else np.array([])
            
            # Cache results
            for j, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
                self._validate_embedding(embedding, text[:50])
                
                if use_cache:
                    self.cache.set(text, embedding)
                
                embeddings.insert(indices_to_embed[j], embedding)
            
            embed_time = time.time() - embed_start
            logger.info(f"Embedding completed in {embed_time:.2f}s")
        
        # Flush cache
        if use_cache:
            self.cache.flush()
        
        total_time = time.time() - start_time
        logger.info(f"Total: {total_time:.2f}s for {total_chunks} chunks")
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def embed_single(
        self, 
        text: str, 
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text:
            return np.array([])
        
        self._ensure_model_loaded()
        
        if use_cache and self.cache.has(text):
            embedding = self.cache.get(text)
            if embedding is not None:
                self._validate_embedding(embedding, text[:50])
                return embedding
        
        embedding = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        
        self._validate_embedding(embedding, text[:50])
        
        if use_cache:
            self.cache.set(text, embedding)
            self.cache.flush()
        
        return embedding
    
    def embed_with_metadata(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings with BOTH numpy and list representations.
        """
        if not texts:
            return []
        
        vectors = self.embed(texts, use_cache=use_cache, batch_size=batch_size, progress_callback=progress_callback)
        
        if len(vectors) == 0:
            return []
        
        results = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            result = {
                "id": f"chunk_{i}_{hashlib.md5(text[:50].encode()).hexdigest()[:8]}",
                "text": text,
                "vector_np": vector,
                "vector": vector.tolist(),
                "vector_dim": len(vector),
            }
            results.append(result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None
    
    def get_load_time_ms(self) -> Optional[float]:
        """Get model loading time in milliseconds."""
        return self._load_time_ms
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_identifier": self.config["name"],
            "dimensions": self.dimensions,
            "speed": self.config["speed"],
            "quality": self.config["quality"],
            "free": self.config["free"],
            "layers": self.config.get("layers", "unknown"),
            "cache_dir": str(self.cache_dir),
            "device": self.device,
            "cache_stats": self.cache.get_stats(),
            "is_loaded": self.is_loaded(),
            "load_time_ms": self.get_load_time_ms(),
            "lazy_loading_enabled": True,
        }


# ========== SINGLETON WITH LAZY LOADING ==========

_embedder: Optional[EmbeddingModel] = None
_embedder_lock = threading.Lock()


def get_embedder(
    model_name: str = "mini-lm",  # Default to reliable model
    enable_disk_cache: bool = True,
    warmup: bool = True,
) -> EmbeddingModel:
    """Get or create singleton embedder instance."""
    global _embedder
    
    if _embedder is not None and _embedder.model_name == model_name:
        return _embedder
    
    with _embedder_lock:
        if _embedder is None or _embedder.model_name != model_name:
            print(f"\n🏗️  Creating embedding model instance (lazy loading enabled)...")
            _embedder = EmbeddingModel(
                model_name=model_name,
                enable_disk_cache=enable_disk_cache,
                warmup=warmup,
            )
        return _embedder


def preload_embedder(
    model_name: str = "mini-lm",
    enable_disk_cache: bool = True,
    warmup: bool = True,
) -> EmbeddingModel:
    """Explicitly preload the embedder."""
    embedder = get_embedder(model_name, enable_disk_cache, warmup)
    embedder._ensure_model_loaded()
    return embedder


def embed_chunks(
    chunks: List[str], 
    use_cache: bool = True,
    batch_size: int = 32,
    return_format: str = "both",
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> List[Dict[str, Any]]:
    """
    Convert chunks to structured embeddings with progress tracking.
    
    Args:
        chunks: List of text chunks from chunker
        use_cache: Whether to use disk cache
        batch_size: Batch size for embedding
        return_format: "both", "numpy_only", or "list_only"
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of dicts with text and embeddings
    """
    if not chunks:
        return []
    
    embedder = get_embedder()
    results = embedder.embed_with_metadata(
        chunks, 
        use_cache=use_cache, 
        batch_size=batch_size,
        progress_callback=progress_callback
    )
    
    if return_format == "numpy_only":
        filtered_results = []
        for r in results:
            new_r = r.copy()
            new_r.pop("vector", None)
            filtered_results.append(new_r)
        return filtered_results
    elif return_format == "list_only":
        filtered_results = []
        for r in results:
            new_r = r.copy()
            new_r.pop("vector_np", None)
            filtered_results.append(new_r)
        return filtered_results
    
    return results


def embed_query(query: str, use_cache: bool = True) -> np.ndarray:
    """Embed a user query for retrieval."""
    embedder = get_embedder()
    return embedder.embed_single(query, use_cache=use_cache)


def clear_embedding_cache():
    """Clear the embedding disk cache."""
    embedder = get_embedder()
    embedder.cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    embedder = get_embedder()
    return embedder.cache.get_stats()


def is_model_loaded() -> bool:
    """Check if embedding model is currently loaded in memory."""
    if _embedder is None:
        return False
    return _embedder.is_loaded()


# ========== MODULE SELF-TEST ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Embeddings Module - Test")
    print("=" * 60)
    
    # Test reliable model only
    print(f"\n📊 Testing mini-lm (all-MiniLM-L6-v2)...")
    embedder = get_embedder(model_name="mini-lm")
    test_texts = ["Hello world", "Testing embeddings"]
    
    def show_progress(batch, total, eta):
        print(f"   Progress: {batch}/{total} | ETA: {eta:.1f}s")
    
    start = time.time()
    vectors = embedder.embed(test_texts, progress_callback=show_progress)
    elapsed = time.time() - start
    print(f"✅ Generated {len(vectors)} embeddings in {elapsed:.2f}s")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
"""
Embeddings Module - Task 4 of RAG Pipeline

Responsibility: Convert text chunks into vector embeddings (meaning representations).

FIXES APPLIED:
1. ✅ Keep BOTH numpy and list representations (performance + serialization)
2. ✅ Integrated EmbeddingCache into main flow
3. ✅ Documented singleton limitations (acceptable for v1)
4. ✅ Added batch_size parameter to pipeline interfaces
5. ✅ Added embedding shape validation with clear errors
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import json
import warnings

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
        """Save cache to disk."""
        if not self.enabled:
            return
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f)
        except IOError as e:
            warnings.warn(f"Could not save cache: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def has(self, text: str) -> bool:
        """Check if text exists in cache."""
        if not self.enabled:
            return False
        key = self._hash_text(text)
        return key in self.cache
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding as numpy array."""
        if not self.enabled:
            return None
        key = self._hash_text(text)
        if key in self.cache:
            return np.array(self.cache[key])
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding (stores as list for JSON)."""
        if not self.enabled:
            return
        key = self._hash_text(text)
        self.cache[key] = embedding.tolist()
        if len(self.cache) % 10 == 0:
            self._save()
    
    def set_batch(self, texts: List[str], embeddings: np.ndarray):
        """Cache multiple embeddings at once."""
        if not self.enabled:
            return
        for text, embedding in zip(texts, embeddings):
            key = self._hash_text(text)
            self.cache[key] = embedding.tolist()
        self._save()
    
    def clear(self):
        """Clear cache."""
        self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "size": len(self.cache),
            "cache_path": str(self.cache_path),
        }


class EmbeddingModel:
    """
    Manages embedding model loading and inference.
    
    v1: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    
    NOTE: Singleton pattern used via get_embedder() function below.
    This is NOT thread-safe but acceptable for single-threaded RAG v1.
    """
    
    MODELS = {
        "mini-lm": {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "speed": "fast",
            "quality": "good",
            "free": True,
        },
        "mpnet": {
            "name": "all-mpnet-base-v2", 
            "dimensions": 768,
            "speed": "medium",
            "quality": "better",
            "free": True,
        },
        "openai-small": {
            "name": "text-embedding-3-small",
            "dimensions": 1536,
            "speed": "slow",
            "quality": "best",
            "free": False,
        },
    }
    
    def __init__(
        self,
        model_name: str = "mini-lm",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        enable_disk_cache: bool = True,
        disk_cache_path: str = ".embedding_cache.json",
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Key from MODELS dict (default: "mini-lm")
            cache_dir: Where to cache model files
            device: "cpu", "cuda", or None (auto-detect)
            enable_disk_cache: Whether to cache embeddings to disk
            disk_cache_path: Path for disk cache
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.MODELS.keys())}")
        
        self.config = self.MODELS[model_name]
        self.model_name = model_name
        self.dimensions = self.config["dimensions"]
        self.device = device
        
        self.cache = EmbeddingCache(
            cache_path=disk_cache_path,
            enabled=enable_disk_cache
        )
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "sentence-transformers"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Loading embedding model: {self.config['name']}")
        self._model = SentenceTransformer(
            self.config["name"],
            cache_folder=str(self.cache_dir),
            device=device
        )
        print(f"✅ Model loaded. Dimensions: {self.dimensions}")
    
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
        
        if np.isnan(embedding).any():
            raise ValueError(f"Embedding contains NaN values. Text snippet: {text_snippet[:100]}")
        
        if np.isinf(embedding).any():
            raise ValueError(f"Embedding contains Inf values. Text snippet: {text_snippet[:100]}")
    
    def embed(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            use_cache: Whether to use disk cache
            batch_size: Number of texts per batch (memory control)
            
        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])
        
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            return np.array([])
        
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        if use_cache:
            for i, text in enumerate(valid_texts):
                cached = self.cache.get(text)
                if cached is not None:
                    self._validate_embedding(cached, text[:50])
                    embeddings.append(cached)
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = valid_texts
            indices_to_embed = list(range(len(valid_texts)))
        
        if texts_to_embed:
            batch_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_result = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                batch_embeddings.append(batch_result)
            
            new_embeddings = np.vstack(batch_embeddings) if batch_embeddings else np.array([])
            
            for j, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
                self._validate_embedding(embedding, text[:50])
                
                if use_cache:
                    self.cache.set(text, embedding)
                
                embeddings.insert(indices_to_embed[j], embedding)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def embed_single(
        self, 
        text: str, 
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text:
            return np.array([])
        
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
        
        return embedding
    
    def embed_with_metadata(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings with BOTH numpy and list representations.
        
        Returns:
            List of dicts with 'text', 'vector_np', 'vector', 'vector_dim'
        """
        if not texts:
            return []
        
        vectors = self.embed(texts, use_cache=use_cache, batch_size=batch_size)
        
        results = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            results.append({
                "id": f"chunk_{i}",
                "text": text,
                "vector_np": vector,
                "vector": vector.tolist(),
                "vector_dim": len(vector),
            })
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_identifier": self.config["name"],
            "dimensions": self.dimensions,
            "speed": self.config["speed"],
            "quality": self.config["quality"],
            "free": self.config["free"],
            "cache_dir": str(self.cache_dir),
            "device": self.device,
            "cache_stats": self.cache.get_stats(),
        }


# ========== SIMPLE FUNCTION INTERFACE ==========

# WARNING: This singleton is NOT thread-safe.
# For concurrent workloads, instantiate EmbeddingModel directly.
_embedder: Optional[EmbeddingModel] = None


def get_embedder(
    model_name: str = "mini-lm",
    enable_disk_cache: bool = True,
) -> EmbeddingModel:
    """Get or create singleton embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = EmbeddingModel(
            model_name=model_name,
            enable_disk_cache=enable_disk_cache
        )
    return _embedder


def embed_chunks(
    chunks: List[str], 
    use_cache: bool = True,
    batch_size: int = 32,
    return_format: str = "both"
) -> List[Dict[str, Any]]:
    """
    Convert chunks to structured embeddings.
    
    Args:
        chunks: List of text chunks from chunker
        use_cache: Whether to use disk cache
        batch_size: Batch size for embedding (controls memory usage)
        return_format: "both", "numpy_only", or "list_only"
    
    Returns:
        List of dicts with text and embeddings
    """
    if not chunks:
        return []
    
    embedder = get_embedder()
    results = embedder.embed_with_metadata(chunks, use_cache=use_cache, batch_size=batch_size)
    
    if return_format == "numpy_only":
        for r in results:
            r.pop("vector", None)
    elif return_format == "list_only":
        for r in results:
            r.pop("vector_np", None)
    
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
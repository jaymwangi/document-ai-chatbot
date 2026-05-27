"""
Vector Store Module - Task 5 of RAG Pipeline

Responsibility: Store embeddings and retrieve semantically similar chunks.
This is where your RAG becomes a search engine.

v1 (Current):
- In-memory storage with linear search O(n)
- Cosine similarity (dot product on normalized vectors)
- Perfect for small/medium datasets (under 10k chunks)

v2 Upgrade Paths:
- FAISS (approximate nearest neighbors for speed)
- ChromaDB (persistent storage)
- Hybrid search (keyword + semantic)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class SearchResult:
    """Single search result with metadata."""
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
            "rank": self.rank,
        }


class VectorStore:
    """
    Simple in-memory vector store with cosine similarity.
    
    v1 Design:
    - Linear search: O(n) complexity
    - All vectors stored in RAM
    - Suitable for portfolio projects, prototypes
    """
    
    def __init__(self, normalize_vectors: bool = True):
        """
        Initialize empty vector store.
        
        Args:
            normalize_vectors: If True, ensures all vectors are unit length.
        """
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._normalize = normalize_vectors
        self._next_id = 0
    
    def _generate_id(self) -> str:
        """Generate unique ID for each chunk."""
        self._next_id += 1
        return f"chunk_{self._next_id - 1}"
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector is unit length for cosine similarity."""
        if not self._normalize:
            return vector
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def add(
        self,
        vector: np.ndarray,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> str:
        """
        Add a single embedding to the store.
        
        Args:
            vector: Embedding vector (numpy array)
            text: Original chunk text
            metadata: Optional additional data
            vector_id: Optional custom ID
        
        Returns:
            ID of the stored vector
        """
        if vector is None or len(vector) == 0:
            raise ValueError("Cannot add empty vector")
        
        if len(vector.shape) != 1:
            raise ValueError(f"Vector must be 1D, got shape {vector.shape}")
        
        normalized_vector = self._normalize_vector(vector)
        
        vid = vector_id or self._generate_id()
        
        self.vectors.append(normalized_vector)
        self.texts.append(text)
        self.metadata.append(metadata or {})
        self.ids.append(vid)
        
        return vid
    
    def add_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple embeddings at once.
        
        Expected item format (from embed_chunks):
        {
            "id": str (optional),
            "text": str,
            "vector_np": np.ndarray,
            "metadata": dict (optional)
        }
        
        Returns:
            List of IDs for stored vectors
        """
        ids = []
        for item in items:
            vid = self.add(
                vector=item["vector_np"],
                text=item["text"],
                metadata=item.get("metadata", {}),
                vector_id=item.get("id"),
            )
            ids.append(vid)
        return ids
    
    def _cosine_similarity(
        self,
        query_vector: np.ndarray,
        stored_vector: np.ndarray,
    ) -> float:
        """Compute cosine similarity via dot product (normalized vectors)."""
        return float(np.dot(query_vector, stored_vector))
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Retrieve top-K most similar chunks.
        
        Args:
            query_vector: Embedded user query (numpy array)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of SearchResult objects, sorted by score descending
        """
        if len(self.vectors) == 0:
            return []
        
        query_norm = self._normalize_vector(query_vector)
        
        scores = []
        for i, vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_norm, vector)
            
            if score_threshold is not None and similarity < score_threshold:
                continue
            
            scores.append({
                "index": i,
                "score": similarity,
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "id": self.ids[i],
            })
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        results = []
        for rank, item in enumerate(scores[:top_k]):
            results.append(SearchResult(
                text=item["text"],
                score=item["score"],
                metadata=item["metadata"],
                rank=rank + 1,
            ))
        
        return results
    
    def search_texts(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[str]:
        """Return only chunk texts (simplified interface)."""
        results = self.search(query_vector, top_k)
        return [r.text for r in results]
    
    def search_with_scores(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return (text, score) tuples for debugging."""
        results = self.search(query_vector, top_k)
        return [(r.text, r.score) for r in results]
    
    def get_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a vector by its ID."""
        for i, vid in enumerate(self.ids):
            if vid == vector_id:
                return {
                    "id": vid,
                    "text": self.texts[i],
                    "vector": self.vectors[i],
                    "metadata": self.metadata[i],
                }
        return None
    
    def delete_by_id(self, vector_id: str) -> bool:
        """Delete a vector by its ID."""
        for i, vid in enumerate(self.ids):
            if vid == vector_id:
                del self.vectors[i]
                del self.texts[i]
                del self.metadata[i]
                del self.ids[i]
                return True
        return False
    
    def size(self) -> int:
        """Number of stored vectors."""
        return len(self.vectors)
    
    def is_empty(self) -> bool:
        """Check if store has any vectors."""
        return len(self.vectors) == 0
    
    def clear(self) -> None:
        """Remove all stored vectors."""
        self.vectors.clear()
        self.texts.clear()
        self.metadata.clear()
        self.ids.clear()
        self._next_id = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics for debugging."""
        dimensions = 0
        if self.vectors:
            dimensions = len(self.vectors[0])
        
        return {
            "total_vectors": len(self.vectors),
            "dimensions": dimensions,
            "has_vectors": len(self.vectors) > 0,
            "normalization_enabled": self._normalize,
        }
    
    def save_to_disk(self, filepath: str) -> None:
        """Save vector store to disk (v1 simple format)."""
        data = {
            "texts": self.texts,
            "metadata": self.metadata,
            "ids": self.ids,
            "vectors": [v.tolist() for v in self.vectors] if self.vectors else [],
            "normalize": self._normalize,
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_disk(self, filepath: str) -> None:
        """Load vector store from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No save file found at {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.clear()
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        self.ids = data["ids"]
        self.vectors = [np.array(v) for v in data["vectors"]]
        self._normalize = data.get("normalize", True)
        self._next_id = len(self.ids)


# ========== CONVENIENCE FUNCTION ==========

def create_vector_store_from_chunks(
    chunks: List[str],
    embeddings_results: List[Dict[str, Any]],
) -> VectorStore:
    """
    Convenience function: Create and populate vector store.
    
    Args:
        chunks: Original chunk texts
        embeddings_results: Output from embed_chunks()
    
    Returns:
        Populated VectorStore instance
    """
    store = VectorStore()
    store.add_batch(embeddings_results)
    return store
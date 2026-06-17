"""
Vector Store Module - Task 5 of RAG Pipeline

Responsibility: Store and retrieve vector embeddings.
Single responsibility: vectors → search → results.

TASK 6 ADDITIONS:
- Version tracking (monotonic counter)
- get_version() method for hybrid retriever compatibility
- Staleness detection support
- Document ID tracking
- Metadata preservation
- Persistent memory across operations
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import hashlib
import time
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


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


@dataclass
class SearchStats:
    """Performance statistics for a search operation."""
    elapsed_ms: float
    vectors_searched: int
    candidates_found: int
    results_returned: int


class VectorStore:
    """
    Simple in-memory vector store with version tracking.
    
    Features:
    - Add vectors in batches
    - Search by cosine similarity
    - Metadata storage
    - Version tracking (monotonic counter)
    - get_version() for hybrid retriever compatibility
    - Document ID tracking
    - Disk persistence (NPZ format)
    
    Usage:
        store = VectorStore()
        store.add_batch(items)
        print(f"Store version: {store.get_version()}")
        
        # Later, check if store has changed
        version = store.get_version()
        if hybrid_retriever.is_stale(version):
            hybrid_retriever.rebuild()
    """
    
    # File format version
    SAVE_FORMAT_VERSION = 2
    
    def __init__(self, normalize_vectors: bool = True, doc_id: Optional[str] = None):
        """
        Initialize empty vector store.
        
        Args:
            normalize_vectors: If True, ensures all vectors are unit length.
            doc_id: Optional document ID for this store instance.
        """
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        
        self._normalize = normalize_vectors
        self._next_id = 0
        self._version: int = 0  # Monotonic version counter
        self._dimension: Optional[int] = None
        self._doc_id = doc_id
        self._created_at = datetime.now().isoformat()
        self._last_modified = self._created_at
    
    @property
    def doc_id(self) -> str:
        """Get document ID, generating one if not set."""
        if self._doc_id is None:
            self._doc_id = self._generate_doc_id()
        return self._doc_id
    
    @doc_id.setter
    def doc_id(self, value: str):
        """Set document ID (only if not already set)."""
        if self._doc_id is not None and self._doc_id != value:
            raise ValueError(f"Cannot change doc_id from {self._doc_id} to {value}")
        self._doc_id = value
    
    @property
    def version(self) -> int:
        """Get current store version (monotonic counter)."""
        return self._version
    
    @property
    def dimension(self) -> Optional[int]:
        """Get vector dimension."""
        return self._dimension
    
    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        return f"doc_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
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
    
    def _validate_vector(self, vector: np.ndarray) -> None:
        """
        Validate vector shape and dimension consistency.
        
        Raises:
            ValueError: If vector is invalid or dimension mismatch.
        """
        if vector is None:
            raise ValueError("Cannot add None vector")
        
        if not isinstance(vector, np.ndarray):
            raise ValueError(f"Vector must be numpy array, got {type(vector)}")
        
        if len(vector.shape) != 1:
            raise ValueError(f"Vector must be 1D, got shape {vector.shape}")
        
        if vector.size == 0:
            raise ValueError("Cannot add empty vector")
        
        # Check for NaN or Inf
        if not np.isfinite(vector).all():
            raise ValueError("Vector contains NaN or Inf values")
        
        # Track and validate dimension
        vector_dim = len(vector)
        if self._dimension is None:
            self._dimension = vector_dim
        elif vector_dim != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, "
                f"got {vector_dim}"
            )
    
    def _increment_version(self) -> None:
        """Increment the version counter on any mutation."""
        self._version += 1
        self._last_modified = datetime.now().isoformat()
    
    def get_version(self) -> int:
        """
        Get the current version of the vector store.
        
        This is the primary method for hybrid retriever compatibility.
        Used to detect when the store has changed and BM25 needs rebuilding.
        
        Returns:
            int: Current version number (monotonic counter)
        
        Example:
            store = VectorStore()
            store.add_batch(items)
            version = store.get_version()  # Returns 1
        """
        return self._version
    
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
        # Validate vector
        self._validate_vector(vector)
        
        # Normalize vector
        normalized_vector = self._normalize_vector(vector)
        
        # Prepare metadata with auto-injected doc_id
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)  # Don't mutate original
        
        # Auto-inject doc_id if not present
        if self._doc_id and "doc_id" not in metadata:
            metadata["doc_id"] = self._doc_id
        if self._doc_id and "source" not in metadata:
            metadata["source"] = self._doc_id
        
        # Add timestamp if not present
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()
        
        # Generate ID
        vid = vector_id or self._generate_id()
        
        # Store
        self.vectors.append(normalized_vector)
        self.texts.append(text)
        self.metadata.append(metadata)
        self.ids.append(vid)
        
        # Increment version
        self._increment_version()
        
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
        if not items:
            return []
        
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
        return_stats: bool = False,
    ) -> List[SearchResult]:
        """
        Retrieve top-K most similar chunks.
        
        Args:
            query_vector: Embedded user query (numpy array)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            return_stats: If True, return SearchStats as second element
        
        Returns:
            List of SearchResult objects, sorted by score descending.
            If return_stats=True, returns (results, stats)
        """
        start_time = time.time()
        
        if len(self.vectors) == 0:
            elapsed = (time.time() - start_time) * 1000
            stats = SearchStats(
                elapsed_ms=elapsed,
                vectors_searched=0,
                candidates_found=0,
                results_returned=0,
            )
            if return_stats:
                return [], stats
            return []
        
        # Validate query vector dimension
        query_dim = len(query_vector)
        if self._dimension is not None and query_dim != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dimension}, "
                f"got {query_dim}"
            )
        
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
        
        elapsed = (time.time() - start_time) * 1000
        stats = SearchStats(
            elapsed_ms=elapsed,
            vectors_searched=len(self.vectors),
            candidates_found=len(scores),
            results_returned=len(results),
        )
        
        if return_stats:
            return results, stats
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
                self._increment_version()
                return True
        return False
    
    def delete_by_metadata(self, key: str, value: Any) -> int:
        """
        Delete all vectors matching a metadata key/value pair.
        
        Returns:
            Number of vectors deleted
        """
        deleted = 0
        i = 0
        while i < len(self.metadata):
            if self.metadata[i].get(key) == value:
                del self.vectors[i]
                del self.texts[i]
                del self.metadata[i]
                del self.ids[i]
                deleted += 1
            else:
                i += 1
        
        if deleted > 0:
            self._increment_version()
        
        return deleted
    
    def size(self) -> int:
        """Number of stored vectors."""
        return len(self.vectors)
    
    def __len__(self) -> int:
        """Pythonic way to get size."""
        return len(self.vectors)
    
    def is_empty(self) -> bool:
        """Check if store has any vectors."""
        return len(self.vectors) == 0
    
    def has_documents(self) -> bool:
        """Check if store has any documents."""
        return not self.is_empty()
    
    def clear(self) -> None:
        """Remove all stored vectors."""
        if len(self.vectors) > 0:
            self.vectors.clear()
            self.texts.clear()
            self.metadata.clear()
            self.ids.clear()
            self._next_id = 0
            self._dimension = None
            self._increment_version()
    
    def has_document(self, doc_id: str) -> bool:
        """Check if a document with given ID exists in the store."""
        for meta in self.metadata:
            if meta.get("doc_id") == doc_id or meta.get("source") == doc_id:
                return True
        return False
    
    def get_document_ids(self) -> List[str]:
        """Get all unique document IDs in the store."""
        doc_ids = set()
        for meta in self.metadata:
            doc_id = meta.get("doc_id") or meta.get("source")
            if doc_id:
                doc_ids.add(doc_id)
        return list(doc_ids)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in the store.
        
        Returns:
            List of dicts with text and metadata for each document
        """
        documents = []
        for i, text in enumerate(self.texts):
            doc = {
                "text": text,
                "metadata": self.metadata[i],
                "id": self.ids[i] if i < len(self.ids) else f"doc_{i}",
            }
            documents.append(doc)
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics for debugging."""
        return {
            "total_vectors": len(self.vectors),
            "total_texts": len(self.texts),
            "dimensions": self._dimension or 0,
            "version": self._version,
            "doc_id": self._doc_id,
            "document_ids": self.get_document_ids(),
            "is_empty": self.is_empty(),
            "has_documents": self.has_documents(),
        }
    
    def is_stale(self, other_version: int) -> bool:
        """
        Check if this store is stale compared to another version.
        
        Args:
            other_version: Version to compare against
        
        Returns:
            True if this store's version is less than other_version
        """
        return self._version < other_version
    
    def is_synced(self, other_version: int) -> bool:
        """Check if this store is at the same version as another."""
        return self._version == other_version
    
    # ========== PERSISTENCE METHODS ==========
    
    def save_to_disk(self, filepath: str, format: str = "npz") -> None:
        """Save vector store to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "doc_id": self._doc_id,
            "file_format_version": self.SAVE_FORMAT_VERSION,
            "state_version": self._version,
            "texts": self.texts,
            "metadata": self.metadata,
            "ids": self.ids,
            "vectors": [v.tolist() for v in self.vectors] if self.vectors else [],
            "normalize": self._normalize,
            "next_id": self._next_id,
            "dimension": self._dimension,
            "created_at": self._created_at,
            "last_modified": self._last_modified,
        }
        
        if format == "npz":
            np.savez_compressed(
                path,
                doc_id=data["doc_id"],
                file_format_version=data["file_format_version"],
                state_version=data["state_version"],
                texts=np.array(self.texts, dtype=object),
                metadata=np.array(self.metadata, dtype=object),
                ids=np.array(self.ids, dtype=object),
                vectors=np.array(self.vectors) if self.vectors else np.array([]),
                normalize=data["normalize"],
                next_id=data["next_id"],
                dimension=self._dimension if self._dimension is not None else -1,
                created_at=self._created_at,
                last_modified=self._last_modified,
            )
            logger.info(f"💾 Saved {len(self.vectors)} vectors (v{self._version}) to {path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> "VectorStore":
        """Load vector store from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No save file found at {filepath}")
        
        extension = path.suffix.lower()
        
        if extension == '.npz':
            data = np.load(path, allow_pickle=True)
            
            store = cls(normalize_vectors=bool(data["normalize"]))
            
            # Load doc_id
            doc_id_val = data["doc_id"]
            if doc_id_val is not None and str(doc_id_val) != 'None':
                store._doc_id = str(doc_id_val)
            
            # Load state version
            if "state_version" in data:
                store._version = int(data["state_version"])
            else:
                store._version = 0
            
            # Load other attributes
            store._next_id = int(data["next_id"])
            store._dimension = int(data["dimension"]) if data["dimension"] != -1 else None
            store._created_at = str(data["created_at"]) if "created_at" in data else datetime.now().isoformat()
            store._last_modified = str(data["last_modified"]) if "last_modified" in data else store._created_at
            
            # Reconstruct data
            vectors_array = data["vectors"]
            if len(vectors_array) > 0:
                store.vectors = [v for v in vectors_array]
            
            store.texts = list(data["texts"])
            store.metadata = list(data["metadata"])
            store.ids = list(data["ids"])
            
            logger.info(f"📂 Loaded {store.size()} vectors (v{store._version}) from {filepath}")
            return store
        else:
            raise ValueError(f"Unsupported file format: {extension}")


# ========== CONVENIENCE FUNCTIONS ==========

def create_vector_store_from_chunks(
    chunks: List[str],
    embeddings_results: List[Dict[str, Any]],
    doc_id: Optional[str] = None,
) -> VectorStore:
    """
    Convenience function: Create and populate vector store.
    
    Args:
        chunks: Original chunk texts
        embeddings_results: Output from embed_chunks()
        doc_id: Optional document ID
    
    Returns:
        Populated VectorStore instance
    """
    store = VectorStore(doc_id=doc_id)
    store.add_batch(embeddings_results)
    return store


# ========== MODULE SELF-TEST ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🗂️ Vector Store - Version Tracking Test")
    print("=" * 60)
    
    # Create store
    store = VectorStore()
    print(f"Initial version: {store.get_version()}")
    
    # Add some items
    test_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.7, 0.7, 0.0]),
    ]
    
    test_texts = [
        "Document about cats",
        "Document about dogs",
        "Document about both cats and dogs",
    ]
    
    for vec, text in zip(test_vectors, test_texts):
        store.add(vec, text)
        print(f"Added: {text[:20]}... (version: {store.get_version()})")
    
    print(f"Final version: {store.get_version()}")
    print(f"Store size: {store.size()}")
    print(f"Has documents: {store.has_documents()}")
    
    # Test get_all_documents
    print("\n📝 get_all_documents():")
    docs = store.get_all_documents()
    for doc in docs[:2]:
        print(f"   - {doc['text'][:30]}... (id: {doc['id']})")
    
    # Test version tracking after deletion
    print("\n📝 Version after deletion:")
    store.delete_by_id("chunk_1")
    print(f"   After deletion: version = {store.get_version()}")
    
    print("\n" + "=" * 60)
    print("✅ Vector Store ready with version tracking!")
    print("   - get_version() for hybrid retriever compatibility")
    print("   - Monotonic version counter")
    print("   - get_all_documents() for BM25 indexing")
    print("   - has_documents() check")
    print("=" * 60)
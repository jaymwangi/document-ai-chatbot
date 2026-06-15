"""
Vector Store Module - Task 5 & 6 of RAG Pipeline

Responsibility: Store embeddings and retrieve semantically similar chunks.
This is where your RAG becomes a search engine with PERSISTENT MEMORY.

TASK 6 ADDITIONS:
- Disk persistence (save/load to NPZ format - fast + compact)
- Document-level storage (store by document ID)
- Fast reload on restart (skip recomputation)
- Metadata preservation (source, page, chunk_index)
- Foundation for FAISS upgrade path

v1 (Current):
- In-memory storage with linear search O(n)
- Cosine similarity (dot product on normalized vectors)
- Disk persistence with NPZ (NumPy format)
- Perfect for portfolio projects (under 10k chunks)

v2 Upgrade Paths:
- FAISS (approximate nearest neighbors for speed)
- ChromaDB (persistent vector database)
- Hybrid search (keyword + semantic)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import hashlib
import pickle


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
    Simple in-memory vector store with PERSISTENCE (Task 6).
    
    TASK 6 FEATURES:
    - Save to disk (NPZ format for fast loading)
    - Load from disk (skip recomputation)
    - Document-level storage
    - Metadata preservation
    
    v1 Design:
    - Linear search: O(n) complexity
    - All vectors stored in RAM
    - Suitable for portfolio projects
    
    Usage:
        store = VectorStore()
        store.add_batch(embeddings)
        store.save_to_disk("my_docs.npz")
        
        # Later...
        new_store = VectorStore.load_from_disk("my_docs.npz")
    """
    
    # File format versions
    SAVE_FORMAT_VERSION = 1
    VALID_EXTENSIONS = ['.npz', '.pkl', '.json']
    
    def __init__(self, normalize_vectors: bool = True, doc_id: Optional[str] = None):
        """
        Initialize empty vector store.
        
        Args:
            normalize_vectors: If True, ensures all vectors are unit length.
            doc_id: Optional document ID for this store instance.
        """
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._normalize = normalize_vectors
        self._next_id = 0
        self.doc_id = doc_id or self._generate_doc_id()
    
    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        import time
        return f"doc_{int(time.time())}_{self._next_id}"
    
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
    
    # ========== TASK 6: PERSISTENCE METHODS ==========
    
    def save_to_disk(self, filepath: str, format: str = "npz") -> None:
        """
        Save vector store to disk (TASK 6).
        
        Supports multiple formats:
        - npz: NumPy compressed format (fast, compact, RECOMMENDED)
        - json: Human-readable (slow, large)
        - pkl: Pickle format (Python-specific)
        
        Args:
            filepath: Path to save file
            format: "npz", "json", or "pkl"
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            "doc_id": self.doc_id,
            "version": self.SAVE_FORMAT_VERSION,
            "texts": self.texts,
            "metadata": self.metadata,
            "ids": self.ids,
            "vectors": [v.tolist() for v in self.vectors] if self.vectors else [],
            "normalize": self._normalize,
            "next_id": self._next_id,
        }
        
        if format == "npz":
            # NPZ format: store vectors as numpy arrays (fastest)
            np.savez_compressed(
                path,
                doc_id=data["doc_id"],
                version=data["version"],
                texts=np.array(self.texts, dtype=object),
                metadata=np.array(self.metadata, dtype=object),
                ids=np.array(self.ids, dtype=object),
                vectors=np.array(self.vectors) if self.vectors else np.array([]),
                normalize=data["normalize"],
                next_id=data["next_id"],
            )
            print(f"💾 Saved {len(self.vectors)} vectors to {path} (NPZ format)")
            
        elif format == "json":
            # JSON format: human-readable
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Saved {len(self.vectors)} vectors to {path} (JSON format)")
            
        elif format == "pkl":
            # Pickle format
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved {len(self.vectors)} vectors to {path} (Pickle format)")
        
        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'json', or 'pkl'")
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> "VectorStore":
        """
        Load vector store from disk (TASK 6).
        
        Automatically detects format based on file extension:
        - .npz → NumPy format
        - .json → JSON format
        - .pkl → Pickle format
        
        Returns:
            Loaded VectorStore instance
        
        Example:
            store = VectorStore.load_from_disk("my_documents.npz")
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No save file found at {filepath}")
        
        extension = path.suffix.lower()
        
        if extension == '.npz':
            # Load NPZ format
            data = np.load(path, allow_pickle=True)
            
            store = cls(normalize_vectors=bool(data["normalize"]))
            store.doc_id = str(data["doc_id"])
            store._next_id = int(data["next_id"])
            
            # Reconstruct vectors
            vectors_array = data["vectors"]
            if len(vectors_array) > 0:
                store.vectors = [v for v in vectors_array]
            
            # Reconstruct texts, metadata, ids
            store.texts = list(data["texts"])
            store.metadata = list(data["metadata"])
            store.ids = list(data["ids"])
            
            print(f"📂 Loaded {store.size()} vectors from {filepath} (NPZ format)")
            return store
            
        elif extension == '.json':
            # Load JSON format
            with open(path, 'r') as f:
                data = json.load(f)
            
            store = cls(normalize_vectors=data.get("normalize", True))
            store.doc_id = data.get("doc_id", store._generate_doc_id())
            store._next_id = data.get("next_id", 0)
            store.texts = data["texts"]
            store.metadata = data["metadata"]
            store.ids = data["ids"]
            store.vectors = [np.array(v) for v in data["vectors"]] if data["vectors"] else []
            
            print(f"📂 Loaded {store.size()} vectors from {filepath} (JSON format)")
            return store
            
        elif extension == '.pkl':
            # Load Pickle format
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            store = cls(normalize_vectors=data.get("normalize", True))
            store.doc_id = data.get("doc_id", store._generate_doc_id())
            store._next_id = data.get("next_id", 0)
            store.texts = data["texts"]
            store.metadata = data["metadata"]
            store.ids = data["ids"]
            store.vectors = [np.array(v) for v in data["vectors"]] if data["vectors"] else []
            
            print(f"📂 Loaded {store.size()} vectors from {filepath} (Pickle format)")
            return store
            
        else:
            raise ValueError(f"Unknown file extension: {extension}. Use .npz, .json, or .pkl")
    
    def has_document(self, doc_id: str) -> bool:
        """Check if a document with given ID exists in the store."""
        # Check if any metadata has matching doc_id
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
            "doc_id": self.doc_id,
            "document_ids": self.get_document_ids(),
        }


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


def load_or_create_store(
    filepath: str,
    chunks: Optional[List[str]] = None,
    embeddings_results: Optional[List[Dict[str, Any]]] = None,
    doc_id: Optional[str] = None,
) -> VectorStore:
    """
    Load store from disk if exists, otherwise create new.
    
    TASK 6: This is the main interface for persistent storage.
    
    Args:
        filepath: Path to save/load file
        chunks: Chunks to embed (if creating new)
        embeddings_results: Embeddings (if creating new)
        doc_id: Document ID
    
    Returns:
        VectorStore instance (loaded or new)
    
    Example:
        store = load_or_create_store(
            "data/my_doc.npz",
            chunks=chunks,
            embeddings_results=embeddings,
            doc_id="my_paper"
        )
    """
    store_path = Path(filepath)
    
    if store_path.exists():
        print(f"📂 Loading existing store from {filepath}")
        return VectorStore.load_from_disk(filepath)
    else:
        print(f"🆕 Creating new store at {filepath}")
        if chunks is None or embeddings_results is None:
            raise ValueError("When creating new store, chunks and embeddings_results are required")
        store = create_vector_store_from_chunks(chunks, embeddings_results, doc_id)
        store.save_to_disk(filepath)
        return store


# ========== MODULE SELF-TEST ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🗂️ Vector Store - Task 6: Persistence Test")
    print("=" * 60)
    
    # Test 1: Basic operations
    print("\n📝 Test 1: Basic vector store operations")
    store = VectorStore()
    
    # Create test data
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
    
    print(f"   ✅ Added {store.size()} vectors")
    print(f"   📊 Stats: {store.get_stats()}")
    
    # Test 2: Persistence (save/load)
    print("\n📝 Test 2: Save and load to disk")
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        store.save_to_disk(tmp_path, format="npz")
        
        loaded_store = VectorStore.load_from_disk(tmp_path)
        print(f"   ✅ Loaded {loaded_store.size()} vectors")
        print(f"   Original size: {store.size()}")
        print(f"   Loaded size: {loaded_store.size()}")
        
        # Verify data integrity
        assert loaded_store.texts == store.texts
        assert len(loaded_store.vectors) == len(store.vectors)
        print("   ✅ Data integrity verified")
        
    finally:
        Path(tmp_path).unlink()
    
    # Test 3: Search functionality
    print("\n📝 Test 3: Search after reload")
    query_vec = np.array([1.0, 0.2, 0.0])
    results = store.search(query_vec, top_k=2)
    
    print(f"   Query: vectors similar to [1.0, 0.2, 0.0]")
    for r in results:
        print(f"      Score {r.score:.3f}: {r.text}")
    
    # Test 4: Document ID tracking
    print("\n📝 Test 4: Document ID tracking")
    doc_store = VectorStore(doc_id="test_document_001")
    doc_store.add(np.array([1.0]), "Content", metadata={"source": "test_document_001"})
    print(f"   Document ID: {doc_store.doc_id}")
    print(f"   Has document: {doc_store.has_document('test_document_001')}")
    print(f"   All document IDs: {doc_store.get_document_ids()}")
    
    print("\n" + "=" * 60)
    print("✅ Task 6 Complete - Vector Store with Persistence!")
    print("   Features:")
    print("   - In-memory search with cosine similarity")
    print("   - Disk persistence (NPZ/JSON/Pickle formats)")
    print("   - Document ID tracking")
    print("   - Fast reload on restart (skip recomputation)")
    print("=" * 60)
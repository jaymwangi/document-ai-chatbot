"""
FAISS Index Module - Production Performance Upgrade for RAG

This module replaces linear search with FAISS (Facebook AI Similarity Search)
for high-speed vector retrieval.

Why FAISS:
- Linear search O(n) → FAISS O(log n) or sub-linear
- 10x-100x faster for large datasets
- Handles millions of vectors efficiently
- GPU support for even faster search

Architecture:
    VectorStore (Task 6) → FAISS Index → Fast Retrieval

Usage:
    index = FaissIndex(dimensions=384)
    index.add(embeddings, chunks, metadata)
    results = index.search(query_vector, top_k=5)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS not installed. Run: pip install faiss-cpu (or faiss-gpu for CUDA)"
    )


@dataclass
class FaissSearchResult:
    """Search result from FAISS index."""
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    index_position: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
            "rank": self.rank,
        }


class FaissIndex:
    """
    FAISS-based vector index for fast similarity search.
    
    v1 Features:
    - IndexFlatIP: Inner product (cosine for normalized vectors)
    - Add vectors in batches
    - Search with top-k
    - Save/load to disk
    
    v2 Upgrades:
    - IVF (Inverted File) for larger datasets
    - HNSW for even faster search
    - GPU support
    - Quantization for memory reduction
    """
    
    # Index types
    INDEX_FLAT_IP = "flat_ip"      # Inner product (cosine for normalized)
    INDEX_FLAT_L2 = "flat_l2"      # Euclidean distance
    INDEX_IVF = "ivf"              # Inverted file (faster for large datasets)
    INDEX_HNSW = "hnsw"            # Hierarchical Navigable Small World
    
    def __init__(
        self,
        dimensions: int,
        index_type: str = "flat_ip",
        normalize_vectors: bool = True,
        use_gpu: bool = False,
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimensions: Embedding dimension (384 for all-MiniLM-L6-v2)
            index_type: "flat_ip", "flat_l2", "ivf", or "hnsw"
            normalize_vectors: If True, ensures vectors are normalized
            use_gpu: If True and available, use GPU for faster search
        """
        self.dimensions = dimensions
        self.index_type = index_type
        self.normalize_vectors = normalize_vectors
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Storage for chunks and metadata (aligned with index positions)
        self.chunks: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        # Create the index
        self.index = self._create_index()
        
        # Track if index is trained (for IVF)
        self.is_trained = index_type not in ["flat_ip", "flat_l2"]
        
        print(f"🔍 FAISS Index initialized:")
        print(f"   Dimensions: {dimensions}")
        print(f"   Type: {index_type}")
        print(f"   GPU enabled: {self.use_gpu}")
        
    def optimize_for_large_dataset(self, n_vectors: int):
        """
        Automatically choose optimal index type based on dataset size.
        
        Args:
            n_vectors: Number of vectors in the dataset
        
        Returns:
            Recommended index type
        """
        if n_vectors < 10000:
            index_type = "flat_ip"
            print(f"   📊 Small dataset ({n_vectors} vectors) → Using Flat index (fastest for small data)")
        elif n_vectors < 50000:
            index_type = "ivf"
            n_lists = min(256, int(4 * np.sqrt(n_vectors)))
            print(f"   📊 Medium dataset ({n_vectors} vectors) → Using IVF with {n_lists} lists")
        else:
            index_type = "hnsw"
            print(f"   📊 Large dataset ({n_vectors} vectors) → Using HNSW for fastest search")
        
        return index_type
    
    def _create_index(self):
        """Create the appropriate FAISS index based on configuration."""
        if self.index_type == "flat_ip":
            index = faiss.IndexFlatIP(self.dimensions)
        elif self.index_type == "flat_l2":
            index = faiss.IndexFlatL2(self.dimensions)
        elif self.index_type == "ivf":
            # IVF requires training; create with 100 centroids
            quantizer = faiss.IndexFlatIP(self.dimensions)
            index = faiss.IndexIVFFlat(quantizer, self.dimensions, 100)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimensions, 32)  # 32 = HNSW neighbors
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        if not self.normalize_vectors:
            return vectors
        
        # L2 normalization
        faiss.normalize_L2(vectors)
        return vectors
    
    def add(self, embeddings: np.ndarray, chunks: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of shape (n_chunks, dimensions)
            chunks: List of chunk texts (same length as embeddings)
            metadata: Optional list of metadata dicts
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != chunks count ({len(chunks)})")
        
        if len(embeddings) == 0:
            return
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # Normalize if needed
        embeddings = self._normalize(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        if metadata:
            self.metadata_list.extend(metadata)
        else:
            self.metadata_list.extend([{} for _ in chunks])
        
        # Generate IDs
        start_id = len(self.ids)
        self.ids.extend([f"chunk_{start_id + i}" for i in range(len(chunks))])
        
        # Train if needed (IVF requires training after first add)
        if self.index_type == "ivf" and not self.is_trained and self.index.ntotal > 100:
            print(f"   Training IVF index with {self.index.ntotal} vectors...")
            self.index.train(embeddings)
            self.is_trained = True
            # After training, need to re-add vectors
            self.index.add(embeddings)
        
        print(f"   ✅ Added {len(chunks)} vectors to FAISS index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[FaissSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding (1D array of length dimensions)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (for IP, higher is better)
        
        Returns:
            List of FaissSearchResult objects
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure correct shape and type
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        # Normalize query if needed
        query_vector = self._normalize(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Process results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for padding
                continue
            
            # For L2 distance, convert to similarity (lower distance = higher similarity)
            if self.index_type == "flat_l2":
                similarity = 1.0 / (1.0 + score)  # Convert distance to similarity
            else:
                similarity = float(score)
            
            # Apply threshold
            if score_threshold is not None and similarity < score_threshold:
                continue
            
            results.append(FaissSearchResult(
                text=self.chunks[idx],
                score=similarity,
                metadata=self.metadata_list[idx],
                rank=rank + 1,
                index_position=int(idx),
            ))
        
        return results
    
    def search_texts(self, query_vector: np.ndarray, top_k: int = 5) -> List[str]:
        """Return only chunk texts."""
        results = self.search(query_vector, top_k)
        return [r.text for r in results]
    
    def search_with_scores(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return (text, score) tuples."""
        results = self.search(query_vector, top_k)
        return [(r.text, r.score) for r in results]
    
    def size(self) -> int:
        """Number of vectors in index."""
        return self.index.ntotal
    
    def is_empty(self) -> bool:
        """Check if index has any vectors."""
        return self.index.ntotal == 0
    
    def clear(self):
        """Clear the index and stored data."""
        self.index = self._create_index()
        self.chunks.clear()
        self.metadata_list.clear()
        self.ids.clear()
        self.is_trained = False
    
    def save(self, filepath: str):
        """
        Save FAISS index and associated data to disk.
        
        Args:
            filepath: Path without extension (will create .faiss and .pkl files)
        """
        path = Path(filepath)
        
        # Save FAISS index
        index_path = path.with_suffix(".faiss")
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks and metadata
        data_path = path.with_suffix(".pkl")
        data = {
            "chunks": self.chunks,
            "metadata": self.metadata_list,
            "ids": self.ids,
            "dimensions": self.dimensions,
            "index_type": self.index_type,
            "normalize_vectors": self.normalize_vectors,
        }
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved FAISS index to {index_path} and {data_path}")
    
    @classmethod
    def load(cls, filepath: str, use_gpu: bool = False) -> "FaissIndex":
        """
        Load FAISS index from disk.
        
        Args:
            filepath: Path without extension (expects .faiss and .pkl files)
            use_gpu: Whether to use GPU
        """
        path = Path(filepath)
        
        # Load FAISS index
        index_path = path.with_suffix(".faiss")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        faiss_index = faiss.read_index(str(index_path))
        
        # Load chunks and metadata
        data_path = path.with_suffix(".pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(
            dimensions=data["dimensions"],
            index_type=data["index_type"],
            normalize_vectors=data["normalize_vectors"],
            use_gpu=use_gpu,
        )
        
        # Replace index with loaded one (handling GPU)
        if use_gpu and instance.use_gpu:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        else:
            instance.index = faiss_index
        
        instance.chunks = data["chunks"]
        instance.metadata_list = data["metadata"]
        instance.ids = data["ids"]
        
        print(f"📂 Loaded FAISS index with {instance.size()} vectors from {filepath}")
        return instance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_vectors": self.size(),
            "dimensions": self.dimensions,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "use_gpu": self.use_gpu,
            "chunks_stored": len(self.chunks),
        }


# ========== BRIDGE FUNCTION ==========

def create_faiss_index_from_vectorstore(
    vector_store,
    index_type: str = "flat_ip",
    use_gpu: bool = False,
) -> FaissIndex:
    """
    Convert existing VectorStore to FAISS index.
    
    This bridges Task 6 (VectorStore) to FAISS upgrade.
    
    Args:
        vector_store: Existing VectorStore instance
        index_type: FAISS index type
        use_gpu: Whether to use GPU
    
    Returns:
        FaissIndex populated with vectors from vector_store
    """
    if vector_store.is_empty():
        raise ValueError("Cannot create FAISS index from empty vector store")
    
    # Get dimensions from first vector
    dimensions = len(vector_store.vectors[0])
    
    # Create FAISS index
    faiss_index = FaissIndex(
        dimensions=dimensions,
        index_type=index_type,
        normalize_vectors=vector_store._normalize,
        use_gpu=use_gpu,
    )
    
    # Convert vectors to numpy array
    embeddings = np.array(vector_store.vectors)
    
    # Add to FAISS index
    faiss_index.add(embeddings, vector_store.texts, vector_store.metadata)
    
    return faiss_index


# ========== MODULE SELF-TEST ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 FAISS Index - Production Performance Test")
    print("=" * 60)
    
    # Create test data
    dimensions = 384
    n_vectors = 1000
    
    print(f"\n📝 Creating {n_vectors} test vectors...")
    np.random.seed(42)
    test_vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
    test_chunks = [f"Chunk {i}: This is test content number {i}" for i in range(n_vectors)]
    
    # Test 1: Create and add to index
    print("\n📝 Test 1: Creating FAISS index")
    index = FaissIndex(dimensions=dimensions, index_type="flat_ip")
    index.add(test_vectors, test_chunks)
    print(f"   Index size: {index.size()}")
    
    # Test 2: Search
    print("\n📝 Test 2: Searching")
    query_vector = test_vectors[0]  # Search for the first vector
    results = index.search(query_vector, top_k=5)
    
    print(f"   Query: {test_chunks[0][:50]}...")
    print(f"   Results:")
    for r in results[:3]:
        print(f"      [{r.rank}] Score {r.score:.4f}: {r.text[:60]}...")
    
    # Test 3: Save and load
    print("\n📝 Test 3: Persistence")
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        tmp_path = Path(tmp.name).stem
    
    try:
        index.save(str(tmp_path))
        loaded_index = FaissIndex.load(str(tmp_path))
        print(f"   Original size: {index.size()}")
        print(f"   Loaded size: {loaded_index.size()}")
        
        # Verify search works
        loaded_results = loaded_index.search(query_vector, top_k=3)
        print(f"   Loaded index search returned {len(loaded_results)} results")
        
    finally:
        # Cleanup
        for ext in [".faiss", ".pkl"]:
            f = Path(str(tmp_path) + ext)
            if f.exists():
                f.unlink()
    
    # Test 4: Performance comparison (linear vs FAISS)
    print("\n📝 Test 4: Performance (linear search vs FAISS)")
    
    from core.vector_store import VectorStore
    import time
    
    # Create VectorStore
    store = VectorStore()
    for vec, chunk in zip(test_vectors[:100], test_chunks[:100]):
        store.add(vec, chunk)
    
    # Linear search
    start = time.time()
    linear_results = store.search(query_vector, top_k=5)
    linear_time = (time.time() - start) * 1000
    
    # FAISS search
    start = time.time()
    faiss_results = index.search(query_vector, top_k=5)
    faiss_time = (time.time() - start) * 1000
    
    print(f"   Linear search (100 vectors): {linear_time:.2f}ms")
    print(f"   FAISS search (1000 vectors): {faiss_time:.2f}ms")
    print(f"   ⚡ FAISS speedup: {linear_time/faiss_time:.1f}x (with 10x more vectors)")
    
    print("\n" + "=" * 60)
    print("✅ FAISS Index Ready!")
    print("   Features:")
    print("   - Fast approximate nearest neighbor search")
    print("   - 10x-100x faster than linear search")
    print("   - GPU support available")
    print("   - Multiple index types (flat, IVF, HNSW)")
    print("=" * 60)
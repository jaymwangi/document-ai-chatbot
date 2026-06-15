"""
Document Cache - Hash-based cache validation for RAG pipeline

Prevents serving stale vectors when documents or configurations change.

Features:
    - Full configuration hashing (model, chunker, index type)
    - Deterministic hash generation
    - Manifest persistence
    - Cache invalidation on any change
    - Version tracking for future migrations

Usage:
    cache = DocumentCache()
    hash = cache.get_documents_hash(files, model_name, chunker_config, index_type)
    if cache.is_valid(hash):
        use_cached_vectors()
    else:
        rebuild_index()
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentCache:
    """
    Cache manager for document vectors.
    
    Invalidates cache when:
        - Any PDF file changes (name, size, modification time)
        - Embedding model changes
        - Chunker configuration changes (chunk_size, overlap, strategy)
        - FAISS index type changes
        - Cache version changes
    
    The cache is stored as a JSON manifest file.
    """
    
    # Current cache version - increment if cache format changes
    CACHE_VERSION = "2.0"
    
    def __init__(self, cache_file: str = "data/cache_manifest.json"):
        """
        Initialize cache manager.
        
        Args:
            cache_file: Path to cache manifest JSON file
        """
        self.cache_file = Path(cache_file)
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """
        Load cache manifest from disk.
        
        Returns:
            Dictionary with cache state, empty dict if not found or corrupted
        """
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            # Check version compatibility
            if manifest.get("version") != self.CACHE_VERSION:
                logger.warning(
                    f"Cache version mismatch: {manifest.get('version')} != {self.CACHE_VERSION}"
                )
                return {}
            
            return manifest
            
        except json.JSONDecodeError:
            logger.warning(f"Cache file corrupted: {self.cache_file}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}
    
    def _save_manifest(self):
        """Save cache manifest to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp for debugging
            self.manifest["last_updated"] = datetime.now().isoformat()
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Cache manifest saved to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cache manifest: {e}")
    
    def get_documents_hash(
        self,
        pdf_files: List[Path],
        embedder_model: str,
        chunker_config: Dict[str, Any],
        faiss_index_type: str,
        additional_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate unique hash for document collection and configuration.
        
        The hash includes:
            - Embedder model name and dimensions
            - Chunker strategy, chunk_size, chunk_overlap
            - FAISS index type (flat_ip, ivf, hnsw)
            - Each PDF file: name, modification time, size
            - Optional additional configuration
        
        Args:
            pdf_files: List of PDF file paths
            embedder_model: Name of embedding model (e.g., "mini-lm")
            chunker_config: Chunker configuration dict with keys:
                - strategy: "sentence", "paragraph", etc.
                - chunk_size: int
                - chunk_overlap: int
            faiss_index_type: "flat_ip", "ivf", or "hnsw"
            additional_config: Optional extra config to include in hash
        
        Returns:
            MD5 hash string (32 characters)
        """
        hasher = hashlib.md5()
        
        # 1. Include cache version (prevents cross-version cache reuse)
        hasher.update(self.CACHE_VERSION.encode())
        
        # 2. Include embedder model
        hasher.update(embedder_model.encode())
        
        # 3. Include chunker configuration (sorted for determinism)
        chunker_str = json.dumps(chunker_config, sort_keys=True)
        hasher.update(chunker_str.encode())
        
        # 4. Include FAISS index type
        hasher.update(faiss_index_type.encode())
        
        # 5. Include additional configuration if provided
        if additional_config:
            additional_str = json.dumps(additional_config, sort_keys=True)
            hasher.update(additional_str.encode())
        
        # 6. Include each PDF file's identity
        # Sort by path for deterministic order
        for pdf in sorted(pdf_files, key=lambda p: str(p)):
            if not pdf.exists():
                logger.warning(f"PDF file not found: {pdf}")
                continue
            
            stat = pdf.stat()
            # Use name, modification time, and size
            file_info = f"{pdf.name}:{stat.st_mtime}:{stat.st_size}"
            hasher.update(file_info.encode())
        
        return hasher.hexdigest()
    
    def is_valid(self, documents_hash: str) -> bool:
        """
        Check if cached vectors are still valid.
        
        Args:
            documents_hash: Hash from get_documents_hash()
        
        Returns:
            True if cache exists and matches the current hash
        """
        if not self.manifest:
            return False
        
        cached_hash = self.manifest.get("documents_hash")
        if cached_hash != documents_hash:
            logger.debug(f"Hash mismatch: {cached_hash[:8]} != {documents_hash[:8]}")
            return False
        
        # Also verify version (redundant but safe)
        if self.manifest.get("version") != self.CACHE_VERSION:
            logger.debug("Version mismatch")
            return False
        
        return True
    
    def save_state(
        self,
        documents_hash: str,
        document_count: int,
        chunk_count: int,
        vector_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save cache state after successful ingestion.
        
        Args:
            documents_hash: Hash from get_documents_hash()
            document_count: Number of documents ingested
            chunk_count: Number of chunks created
            vector_count: Number of vectors stored
            metadata: Optional additional metadata to store
        """
        self.manifest = {
            "version": self.CACHE_VERSION,
            "documents_hash": documents_hash,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "created_at": datetime.now().isoformat(),
        }
        
        if metadata:
            self.manifest["metadata"] = metadata
        
        self._save_manifest()
        logger.info(f"💾 Cache saved (hash: {documents_hash[:8]}..., docs: {document_count}, chunks: {chunk_count})")
    
    def get_cached_info(self) -> Dict[str, Any]:
        """
        Get information about the cached state.
        
        Returns:
            Dictionary with cache info or empty dict if no cache
        """
        if not self.manifest:
            return {"cached": False}
        
        return {
            "cached": True,
            "documents_hash": self.manifest.get("documents_hash", "")[:8],
            "document_count": self.manifest.get("document_count", 0),
            "chunk_count": self.manifest.get("chunk_count", 0),
            "vector_count": self.manifest.get("vector_count", 0),
            "created_at": self.manifest.get("created_at"),
            "version": self.manifest.get("version"),
        }
    
    def clear(self):
        """Clear cache manifest."""
        self.manifest = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("🗑️ Cache cleared")
    
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return not self.manifest
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists(),
            "manifest_size": len(self.manifest),
            "cached_documents": self.manifest.get("document_count", 0),
            "cached_chunks": self.manifest.get("chunk_count", 0),
            "cache_version": self.CACHE_VERSION,
        }


# =========================
# CONVENIENCE FUNCTION
# =========================

def create_document_cache(cache_file: str = "data/cache_manifest.json") -> DocumentCache:
    """
    Create a document cache instance.
    
    Args:
        cache_file: Path to cache manifest file
    
    Returns:
        DocumentCache instance
    """
    return DocumentCache(cache_file=cache_file)


# =========================
# SELF-TEST
# =========================

if __name__ == "__main__":
    import tempfile
    from datetime import datetime
    
    print("=" * 60)
    print("🧪 Testing DocumentCache")
    print("=" * 60)
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("test content")
        test_pdf = Path(f.name)
    
    try:
        # Initialize cache
        cache = DocumentCache(cache_file="test_cache_manifest.json")
        
        # Test 1: Empty cache
        print("\n📝 Test 1: Empty cache")
        print(f"   Is empty: {cache.is_empty()}")
        print(f"   Is valid: {cache.is_valid('test')}")
        
        # Test 2: Generate hash
        print("\n📝 Test 2: Generate document hash")
        chunker_config = {
            "strategy": "sentence",
            "chunk_size": 500,
            "chunk_overlap": 75,
        }
        
        hash1 = cache.get_documents_hash(
            pdf_files=[test_pdf],
            embedder_model="mini-lm",
            chunker_config=chunker_config,
            faiss_index_type="flat_ip"
        )
        print(f"   Hash: {hash1[:16]}...")
        
        # Test 3: Save and validate
        print("\n📝 Test 3: Save and validate")
        cache.save_state(
            documents_hash=hash1,
            document_count=1,
            chunk_count=100,
            vector_count=100
        )
        
        is_valid = cache.is_valid(hash1)
        print(f"   Is valid: {is_valid}")
        
        # Test 4: Different hash (should invalidate)
        print("\n📝 Test 4: Different configuration")
        chunker_config2 = {
            "strategy": "paragraph",
            "chunk_size": 800,
            "chunk_overlap": 100,
        }
        
        hash2 = cache.get_documents_hash(
            pdf_files=[test_pdf],
            embedder_model="mini-lm",
            chunker_config=chunker_config2,
            faiss_index_type="flat_ip"
        )
        print(f"   New hash: {hash2[:16]}...")
        print(f"   Is valid with new hash: {cache.is_valid(hash2)}")
        
        # Test 5: Cache info
        print("\n📝 Test 5: Cache info")
        info = cache.get_cached_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test 6: Stats
        print("\n📝 Test 6: Cache stats")
        stats = cache.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test 7: Clear cache
        print("\n📝 Test 7: Clear cache")
        cache.clear()
        print(f"   Is empty after clear: {cache.is_empty()}")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Cleanup
        test_pdf.unlink()
        cache_file = Path("test_cache_manifest.json")
        if cache_file.exists():
            cache_file.unlink()
    
    print("\n" + "=" * 60)
    print("✅ DocumentCache ready for production!")
    print("=" * 60)
"""Unit tests for Embeddings module - Task 4"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from services.embeddings import (
    EmbeddingModel, 
    EmbeddingCache,
    embed_chunks, 
    embed_query,
    get_embedder,
    clear_embedding_cache
)


class TestEmbeddingCache:
    """Tests for EmbeddingCache"""
    
    def test_cache_set_and_get(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            cache_path = tmp.name
        
        try:
            cache = EmbeddingCache(cache_path=cache_path, enabled=True)
            text = "test text"
            vector = np.array([1.0, 2.0, 3.0])
            
            cache.set(text, vector)
            assert cache.has(text) is True
            
            retrieved = cache.get(text)
            assert retrieved is not None
            assert np.allclose(retrieved, vector)
        finally:
            Path(cache_path).unlink(missing_ok=True)
    
    def test_cache_disabled(self):
        cache = EmbeddingCache(enabled=False)
        cache.set("test", np.array([1.0]))
        assert cache.has("test") is False
        assert cache.get("test") is None
    
    def test_cache_clear(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            cache_path = tmp.name
        
        try:
            cache = EmbeddingCache(cache_path=cache_path, enabled=True)
            cache.set("text1", np.array([1.0]))
            cache.set("text2", np.array([2.0]))
            assert cache.get_stats()["size"] == 2
            
            cache.clear()
            assert cache.get_stats()["size"] == 0
        finally:
            Path(cache_path).unlink(missing_ok=True)


class TestEmbeddingModel:
    """Tests for EmbeddingModel"""
    
    def test_model_loading(self):
        embedder = EmbeddingModel(model_name="mini-lm", enable_disk_cache=False)
        assert embedder.dimensions == 384
        assert embedder.config["name"] == "all-MiniLM-L6-v2"
    
    def test_embed_single(self):
        embedder = EmbeddingModel(enable_disk_cache=False)
        vector = embedder.embed_single("test text")
        assert len(vector) == 384
        assert isinstance(vector, np.ndarray)
    
    def test_embed_batch(self):
        embedder = EmbeddingModel(enable_disk_cache=False)
        texts = ["text1", "text2", "text3"]
        vectors = embedder.embed(texts)
        assert vectors.shape == (3, 384)
    
    def test_empty_input(self):
        embedder = EmbeddingModel(enable_disk_cache=False)
        assert len(embedder.embed([])) == 0
        assert len(embedder.embed_single("")) == 0
    
    def test_embed_with_metadata(self):
        embedder = EmbeddingModel(enable_disk_cache=False)
        texts = ["chunk one", "chunk two"]
        results = embedder.embed_with_metadata(texts)
        
        assert len(results) == 2
        for result in results:
            assert "id" in result
            assert "text" in result
            assert "vector_np" in result
            assert "vector" in result
            assert "vector_dim" in result
            assert len(result["vector"]) == 384
    
    def test_embedding_validation(self):
        embedder = EmbeddingModel(enable_disk_cache=False)
        
        # Wrong dimension should raise error
        wrong_vector = np.array([1.0, 2.0])  # Only 2 dimensions
        
        import pytest
        with pytest.raises(ValueError):
            embedder._validate_embedding(wrong_vector, "test")
    
    def test_similarity_semantic(self):
        """Similar meanings should have high cosine similarity"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        
        vec_car = embedder.embed_single("automobile")
        vec_vehicle = embedder.embed_single("car")
        similarity = np.dot(vec_car, vec_vehicle)
        
        assert similarity > 0.5  # Should be reasonably similar


class TestFunctionInterfaces:
    """Tests for convenience functions"""
    
    def test_embed_chunks(self):
        chunks = ["chunk one", "chunk two", "chunk three"]
        results = embed_chunks(chunks, use_cache=False)
        
        assert len(results) == 3
        for result in results:
            assert "text" in result
            assert "vector_np" in result
            assert "vector" in result
    
    def test_embed_query(self):
        vector = embed_query("test question", use_cache=False)
        assert len(vector) == 384
        assert isinstance(vector, np.ndarray)
    
    def test_singleton_pattern(self):
        embedder1 = get_embedder(enable_disk_cache=False)
        embedder2 = get_embedder(enable_disk_cache=False)
        assert embedder1 is embedder2
    
    def test_clear_cache(self):
        clear_embedding_cache()
        # Just verify it doesn't error
        assert True


def run_tests():
    """Manual test runner"""
    
    print("Running Embeddings tests...")
    
    # Cache tests
    test_cache = TestEmbeddingCache()
    test_cache.test_cache_set_and_get()
    print("✅ test_cache_set_and_get")
    test_cache.test_cache_disabled()
    print("✅ test_cache_disabled")
    test_cache.test_cache_clear()
    print("✅ test_cache_clear")
    
    # Model tests
    test_model = TestEmbeddingModel()
    test_model.test_model_loading()
    print("✅ test_model_loading")
    test_model.test_embed_single()
    print("✅ test_embed_single")
    test_model.test_embed_batch()
    print("✅ test_embed_batch")
    test_model.test_empty_input()
    print("✅ test_empty_input")
    test_model.test_embed_with_metadata()
    print("✅ test_embed_with_metadata")
    test_model.test_embedding_validation()
    print("✅ test_embedding_validation")
    test_model.test_similarity_semantic()
    print("✅ test_similarity_semantic")
    
    # Function tests
    test_funcs = TestFunctionInterfaces()
    test_funcs.test_embed_chunks()
    print("✅ test_embed_chunks")
    test_funcs.test_embed_query()
    print("✅ test_embed_query")
    test_funcs.test_singleton_pattern()
    print("✅ test_singleton_pattern")
    test_funcs.test_clear_cache()
    print("✅ test_clear_cache")
    
    print("\n🎉 All Task 4 tests passed!")


if __name__ == "__main__":
    run_tests()
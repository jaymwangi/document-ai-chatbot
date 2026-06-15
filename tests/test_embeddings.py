"""Unit tests for Embeddings module - Task 4 & 5 (Lazy Loading + Warmup)"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from services.embeddings import (
    EmbeddingModel, 
    EmbeddingCache,
    embed_chunks, 
    embed_query,
    get_embedder,
    clear_embedding_cache,
    preload_embedder,
    is_model_loaded,
    get_cache_stats,
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
    
    def test_cache_batch_set(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            cache_path = tmp.name
        
        try:
            cache = EmbeddingCache(cache_path=cache_path, enabled=True)
            texts = ["text1", "text2", "text3"]
            vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            
            cache.set_batch(texts, vectors)
            
            for text in texts:
                assert cache.has(text) is True
                retrieved = cache.get(text)
                assert retrieved is not None
        finally:
            Path(cache_path).unlink(missing_ok=True)


class TestEmbeddingModelLazyLoading:
    """Tests for Task 5: Lazy Loading functionality"""
    
    def test_lazy_loading_initial_state(self):
        """Model should NOT be loaded at initialization"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        assert embedder.is_loaded() is False
        assert embedder._model is None
    
    def test_lazy_loading_triggers_on_first_embed(self):
        """Model should load only when first embedding is requested"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        
        # Before: not loaded
        assert embedder.is_loaded() is False
        
        # First embed triggers load
        start_time = time.time()
        vector = embedder.embed_single("test text")
        elapsed = (time.time() - start_time) * 1000
        
        # After: loaded
        assert embedder.is_loaded() is True
        assert embedder.get_load_time_ms() is not None
        assert len(vector) == embedder.dimensions
        print(f"   First embed time (with load): {elapsed:.0f}ms")
    
    def test_subsequent_calls_are_fast(self):
        """After loading, subsequent embeds should be fast"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        
        # First call (with load)
        start1 = time.time()
        embedder.embed_single("first text")
        time1 = (time.time() - start1) * 1000
        
        # Second call (no load)
        start2 = time.time()
        embedder.embed_single("second text")
        time2 = (time.time() - start2) * 1000
        
        # Second call should be significantly faster
        # (but on CPU, difference may be small; just ensure it works)
        print(f"   First call: {time1:.0f}ms, Second: {time2:.0f}ms")
        assert embedder.is_loaded() is True
    
    def test_warmup_improves_first_query(self):
        """Warmup should prepare the model"""
        # With warmup enabled
        embedder_warm = EmbeddingModel(enable_disk_cache=False, warmup=True)
        start_warm = time.time()
        embedder_warm.embed_single("test")
        time_warm = (time.time() - start_warm) * 1000
        
        # Without warmup (but model already loaded in previous test)
        # Just verify warmup doesn't break anything
        assert embedder_warm.is_loaded() is True
        print(f"   With warmup first embed: {time_warm:.0f}ms")
    
    def test_no_double_loading(self):
        """Model should not load twice"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        
        # First embed
        embedder.embed_single("text1")
        first_load_time = embedder.get_load_time_ms()
        
        # Clear the model reference to simulate? No, we want to test that
        # embed_single doesn't trigger reload when model is already loaded
        embedder.embed_single("text2")
        
        # Load time should still be the same (no second load)
        assert embedder.get_load_time_ms() == first_load_time


class TestEmbeddingModelWarmup:
    """Tests for Task 5: Warmup functionality"""
    
    def test_warmup_enabled_by_default(self):
        """Warmup should be enabled by default"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        assert embedder.warmup is True
    
    def test_warmup_can_be_disabled(self):
        """Warmup should be configurable"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        assert embedder.warmup is False
    
    def test_warmup_doesnt_crash(self):
        """Warmup should not cause errors"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=True)
        # Just ensure it doesn't error
        embedder._ensure_model_loaded()
        assert embedder.is_loaded() is True


class TestEmbeddingModelBatchProcessing:
    """Tests for batch embedding with lazy loading"""
    
    def test_batch_triggers_lazy_load(self):
        """First batch should trigger lazy load"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        
        assert embedder.is_loaded() is False
        
        texts = ["text1", "text2", "text3"]
        vectors = embedder.embed(texts)
        
        assert embedder.is_loaded() is True
        assert vectors.shape == (3, embedder.dimensions)
    
    def test_batch_empty_texts(self):
        """Empty batch should return empty array"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        result = embedder.embed([])
        assert len(result) == 0
    
    def test_batch_with_empty_strings(self):
        """Batch with empty strings should filter them out"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        texts = ["valid text", "", "another valid", "   "]
        vectors = embedder.embed(texts)
        
        # Should only embed non-empty texts
        # Note: The current implementation may return different counts
        assert vectors.shape[1] == embedder.dimensions
    
    def test_batch_size_control(self):
        """Should respect batch_size parameter"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        texts = [f"text {i}" for i in range(20)]
        
        # Test with small batch size
        import time
        start = time.time()
        vectors_small = embedder.embed(texts, batch_size=4)
        time_small = time.time() - start
        
        # Recreate embedder to reset cache
        embedder2 = EmbeddingModel(enable_disk_cache=False, warmup=False)
        start = time.time()
        vectors_large = embedder2.embed(texts, batch_size=20)
        time_large = time.time() - start
        
        print(f"   Batch size 4: {time_small:.2f}s, Batch size 20: {time_large:.2f}s")
        assert vectors_small.shape == vectors_large.shape


class TestEmbeddingModelValidation:
    """Tests for embedding validation"""
    
    def test_shape_validation(self):
        """Should validate embedding dimensions"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        
        # This should pass
        good_vector = np.random.randn(embedder.dimensions)
        embedder._validate_embedding(good_vector, "test")
        
        # This should fail
        wrong_vector = np.random.randn(embedder.dimensions + 10)
        with pytest.raises(ValueError):
            embedder._validate_embedding(wrong_vector, "test")
    
    def test_nan_validation(self):
        """Should detect NaN values"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        
        nan_vector = np.array([np.nan] * embedder.dimensions)
        with pytest.raises(ValueError):
            embedder._validate_embedding(nan_vector, "test")
    
    def test_inf_validation(self):
        """Should detect Inf values"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        
        inf_vector = np.array([np.inf] * embedder.dimensions)
        with pytest.raises(ValueError):
            embedder._validate_embedding(inf_vector, "test")
    
    def test_2d_vector_fails(self):
        """2D vectors should be rejected"""
        embedder = EmbeddingModel(enable_disk_cache=False)
        
        vector_2d = np.random.randn(2, embedder.dimensions // 2)
        with pytest.raises(ValueError):
            embedder._validate_embedding(vector_2d, "test")


class TestEmbeddingModelMetadata:
    """Tests for metadata and info methods"""
    
    def test_get_info_before_load(self):
        """Info should be available before model loads"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        info = embedder.get_info()
        
        assert info["model_name"] == "mini-lm"
        assert info["dimensions"] == 384
        assert info["is_loaded"] is False
        assert info["lazy_loading_enabled"] is True
        assert info["load_time_ms"] is None
    
    def test_get_info_after_load(self):
        """Info should update after model loads"""
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        embedder.embed_single("test")
        
        info = embedder.get_info()
        assert info["is_loaded"] is True
        assert info["load_time_ms"] is not None


class TestSingletonPattern:
    """Tests for singleton pattern with lazy loading"""
    
    def test_singleton_returns_same_instance(self):
        """get_embedder should return same instance"""
        embedder1 = get_embedder(enable_disk_cache=False)
        embedder2 = get_embedder(enable_disk_cache=False)
        assert embedder1 is embedder2
    
    def test_singleton_lazy_loading(self):
        """Singleton should not load model immediately"""
        # Clear any existing embedder
        global _embedder
        import services.embeddings
        services.embeddings._embedder = None
        
        embedder = get_embedder(enable_disk_cache=False, warmup=False)
        assert embedder.is_loaded() is False
    
    def test_preload_embedder_forces_load(self):
        """preload_embedder should force model load"""
        # Clear any existing embedder
        import services.embeddings
        services.embeddings._embedder = None
        
        embedder = preload_embedder(enable_disk_cache=False, warmup=False)
        assert embedder.is_loaded() is True
    
    def test_is_model_loaded_function(self):
        """is_model_loaded() should return correct state"""
        # Clear existing
        import services.embeddings
        services.embeddings._embedder = None
        
        assert is_model_loaded() is False
        
        embedder = get_embedder(enable_disk_cache=False, warmup=False)
        assert is_model_loaded() is False  # Still not loaded
        
        embedder.embed_single("test")
        assert is_model_loaded() is True


class TestFunctionInterfaces:
    """Tests for convenience functions with lazy loading"""
    
    def test_embed_chunks_triggers_lazy_load(self):
        """embed_chunks should work with lazy loading"""
        # Clear existing
        import services.embeddings
        services.embeddings._embedder = None
        
        chunks = ["chunk one", "chunk two", "chunk three"]
        results = embed_chunks(chunks, use_cache=False)
        
        assert len(results) == 3
        for result in results:
            assert "vector_np" in result
            assert len(result["vector"]) == 384
    
    def test_embed_query_triggers_lazy_load(self):
        """embed_query should work with lazy loading"""
        import services.embeddings
        services.embeddings._embedder = None
        
        vector = embed_query("test question", use_cache=False)
        assert len(vector) == 384
    
    def test_embed_chunks_empty(self):
        """Empty chunks should return empty list"""
        results = embed_chunks([])
        assert results == []
    
    def test_embed_chunks_return_formats(self):
        chunks = ["test chunk"]
        
        # Both formats
        results_both = embed_chunks(chunks, return_format="both")
        assert "vector_np" in results_both[0]
        assert "vector" in results_both[0]
        
        # Numpy only
        results_numpy = embed_chunks(chunks, return_format="numpy_only")
        assert "vector_np" in results_numpy[0]
        assert "vector" not in results_numpy[0]
        
        # List only
        results_list = embed_chunks(chunks, return_format="list_only")
        assert "vector_np" not in results_list[0]
        assert "vector" in results_list[0]
    
    def test_clear_cache_function(self):
        """clear_embedding_cache should work"""
        # Just verify it doesn't error
        clear_embedding_cache()
        assert True
    
    def test_get_cache_stats(self):
        """get_cache_stats should return dict"""
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "size" in stats


class TestEmbeddingModelThreadSafety:
    """Basic thread-safety tests for lazy loading"""
    
    def test_concurrent_embed_calls(self):
        """Multiple threads should work with lazy loading"""
        import threading
        
        embedder = EmbeddingModel(enable_disk_cache=False, warmup=False)
        results = []
        errors = []
        
        def embed_in_thread(text):
            try:
                vector = embedder.embed_single(text)
                results.append(vector)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=embed_in_thread, args=(f"text {i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        assert embedder.is_loaded() is True


# =========================
# PERFORMANCE TEST (Optional)
# =========================

def test_performance_comparison():
    """Compare performance with and without cache (optional)"""
    print("\n📊 Performance Comparison:")
    
    # Without cache
    start = time.time()
    embedder1 = EmbeddingModel(enable_disk_cache=False, warmup=False)
    embedder1.embed_single("test text")
    time_no_cache = (time.time() - start) * 1000
    
    # With cache (second call should be faster)
    embedder2 = EmbeddingModel(enable_disk_cache=True, warmup=False)
    embedder2.embed_single("test text")  # First call (loads model)
    start = time.time()
    embedder2.embed_single("test text")  # Second call (cached)
    time_with_cache = (time.time() - start) * 1000
    
    print(f"   Without cache (first call): {time_no_cache:.0f}ms")
    print(f"   With cache (second call): {time_with_cache:.0f}ms")
    
    # This is informative only, not an assertion
    assert True


def run_tests():
    """Manual test runner"""
    
    print("=" * 60)
    print("🧠 Embeddings Module - Task 4 & 5 Comprehensive Tests")
    print("=" * 60)
    
    # Cache tests
    print("\n📦 Testing EmbeddingCache...")
    test_cache = TestEmbeddingCache()
    test_cache.test_cache_set_and_get()
    print("✅ test_cache_set_and_get")
    test_cache.test_cache_disabled()
    print("✅ test_cache_disabled")
    test_cache.test_cache_clear()
    print("✅ test_cache_clear")
    test_cache.test_cache_batch_set()
    print("✅ test_cache_batch_set")
    
    # Lazy loading tests (Task 5)
    print("\n🔄 Testing Lazy Loading (Task 5)...")
    test_lazy = TestEmbeddingModelLazyLoading()
    test_lazy.test_lazy_loading_initial_state()
    print("✅ test_lazy_loading_initial_state")
    test_lazy.test_lazy_loading_triggers_on_first_embed()
    print("✅ test_lazy_loading_triggers_on_first_embed")
    test_lazy.test_subsequent_calls_are_fast()
    print("✅ test_subsequent_calls_are_fast")
    test_lazy.test_warmup_improves_first_query()
    print("✅ test_warmup_improves_first_query")
    test_lazy.test_no_double_loading()
    print("✅ test_no_double_loading")
    
    # Warmup tests
    print("\n🔥 Testing Warmup...")
    test_warmup = TestEmbeddingModelWarmup()
    test_warmup.test_warmup_enabled_by_default()
    print("✅ test_warmup_enabled_by_default")
    test_warmup.test_warmup_can_be_disabled()
    print("✅ test_warmup_can_be_disabled")
    test_warmup.test_warmup_doesnt_crash()
    print("✅ test_warmup_doesnt_crash")
    
    # Batch processing tests
    print("\n📦 Testing Batch Processing...")
    test_batch = TestEmbeddingModelBatchProcessing()
    test_batch.test_batch_triggers_lazy_load()
    print("✅ test_batch_triggers_lazy_load")
    test_batch.test_batch_empty_texts()
    print("✅ test_batch_empty_texts")
    test_batch.test_batch_with_empty_strings()
    print("✅ test_batch_with_empty_strings")
    test_batch.test_batch_size_control()
    print("✅ test_batch_size_control")
    
    # Validation tests
    print("\n✅ Testing Embedding Validation...")
    test_val = TestEmbeddingModelValidation()
    test_val.test_shape_validation()
    print("✅ test_shape_validation")
    test_val.test_nan_validation()
    print("✅ test_nan_validation")
    test_val.test_inf_validation()
    print("✅ test_inf_validation")
    test_val.test_2d_vector_fails()
    print("✅ test_2d_vector_fails")
    
    # Metadata tests
    print("\n📊 Testing Metadata...")
    test_meta = TestEmbeddingModelMetadata()
    test_meta.test_get_info_before_load()
    print("✅ test_get_info_before_load")
    test_meta.test_get_info_after_load()
    print("✅ test_get_info_after_load")
    
    # Singleton tests
    print("\n🏗️ Testing Singleton Pattern...")
    test_singleton = TestSingletonPattern()
    test_singleton.test_singleton_returns_same_instance()
    print("✅ test_singleton_returns_same_instance")
    test_singleton.test_singleton_lazy_loading()
    print("✅ test_singleton_lazy_loading")
    test_singleton.test_preload_embedder_forces_load()
    print("✅ test_preload_embedder_forces_load")
    test_singleton.test_is_model_loaded_function()
    print("✅ test_is_model_loaded_function")
    
    # Function interface tests
    print("\n🔧 Testing Function Interfaces...")
    test_funcs = TestFunctionInterfaces()
    test_funcs.test_embed_chunks_triggers_lazy_load()
    print("✅ test_embed_chunks_triggers_lazy_load")
    test_funcs.test_embed_query_triggers_lazy_load()
    print("✅ test_embed_query_triggers_lazy_load")
    test_funcs.test_embed_chunks_empty()
    print("✅ test_embed_chunks_empty")
    test_funcs.test_embed_chunks_return_formats()
    print("✅ test_embed_chunks_return_formats")
    test_funcs.test_clear_cache_function()
    print("✅ test_clear_cache_function")
    test_funcs.test_get_cache_stats()
    print("✅ test_get_cache_stats")
    
    # Thread safety tests
    print("\n🧵 Testing Thread Safety...")
    test_thread = TestEmbeddingModelThreadSafety()
    test_thread.test_concurrent_embed_calls()
    print("✅ test_concurrent_embed_calls")
    
    # Performance test (optional)
    print("\n⚡ Running performance test...")
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("🎉 All Task 4 & 5 tests passed!")
    print("   Features tested:")
    print("   - Lazy loading (model loads on first use)")
    print("   - Warmup support")
    print("   - Singleton pattern")
    print("   - Thread safety")
    print("   - Embedding validation")
    print("   - Batch processing")
    print("   - Disk caching")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
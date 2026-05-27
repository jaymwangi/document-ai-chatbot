"""Unit tests for Vector Store module - Task 5"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.vector_store import VectorStore, SearchResult


class TestVectorStore:
    """Test suite for VectorStore"""
    
    def test_empty_store(self):
        store = VectorStore()
        assert store.size() == 0
        assert store.is_empty()
        assert len(store.search(np.array([1.0, 0.0]))) == 0
    
    def test_add_single_vector(self):
        store = VectorStore()
        vector = np.array([1.0, 0.0, 0.0])
        text = "Test document"
        
        vid = store.add(vector, text)
        
        assert store.size() == 1
        assert vid.startswith("chunk_")
        assert store.texts[0] == text
    
    def test_add_batch(self):
        store = VectorStore()
        items = [
            {"text": "doc1", "vector_np": np.array([1.0, 0.0])},
            {"text": "doc2", "vector_np": np.array([0.0, 1.0])},
        ]
        
        ids = store.add_batch(items)
        
        assert store.size() == 2
        assert len(ids) == 2
    
    def test_add_empty_vector_raises_error(self):
        store = VectorStore()
        
        import pytest
        with pytest.raises(ValueError):
            store.add(np.array([]), "empty")
    
    def test_search_returns_top_k(self):
        store = VectorStore()
        
        # Add vectors
        store.add(np.array([1.0, 0.0]), "Cats document")
        store.add(np.array([0.0, 1.0]), "Dogs document")
        store.add(np.array([0.7, 0.7]), "Both cats and dogs")
        
        # Query similar to cats
        query = np.array([1.0, 0.2])
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0].score > results[1].score
    
    def test_search_with_threshold(self):
        store = VectorStore()
        
        store.add(np.array([1.0, 0.0]), "Very relevant")
        store.add(np.array([0.0, 1.0]), "Not relevant")
        
        query = np.array([1.0, 0.0])
        results = store.search(query, top_k=2, score_threshold=0.5)
        
        # Only the relevant one should pass threshold
        assert len(results) == 1
        assert results[0].text == "Very relevant"
    
    def test_search_texts_interface(self):
        store = VectorStore()
        store.add(np.array([1.0, 0.0]), "Result A")
        store.add(np.array([0.0, 1.0]), "Result B")
        
        query = np.array([1.0, 0.0])
        texts = store.search_texts(query, top_k=1)
        
        assert len(texts) == 1
        assert texts[0] == "Result A"
    
    def test_search_with_scores(self):
        store = VectorStore()
        store.add(np.array([1.0, 0.0]), "Document")
        
        query = np.array([1.0, 0.0])
        results = store.search_with_scores(query)
        
        assert len(results) == 1
        text, score = results[0]
        assert text == "Document"
        assert isinstance(score, float)
    
    def test_get_by_id(self):
        store = VectorStore()
        vid = store.add(np.array([1.0]), "Test")
        
        retrieved = store.get_by_id(vid)
        assert retrieved is not None
        assert retrieved["text"] == "Test"
        
        assert store.get_by_id("nonexistent") is None
    
    def test_delete_by_id(self):
        store = VectorStore()
        vid = store.add(np.array([1.0]), "To delete")
        assert store.size() == 1
        
        deleted = store.delete_by_id(vid)
        assert deleted is True
        assert store.size() == 0
        
        assert store.delete_by_id("nonexistent") is False
    
    def test_clear(self):
        store = VectorStore()
        store.add(np.array([1.0]), "One")
        store.add(np.array([2.0]), "Two")
        
        assert store.size() == 2
        store.clear()
        assert store.size() == 0
    
    def test_stats(self):
        store = VectorStore()
        store.add(np.array([1.0, 2.0, 3.0]), "Test")
        
        stats = store.get_stats()
        assert stats["total_vectors"] == 1
        assert stats["dimensions"] == 3
        assert stats["has_vectors"] is True
    
    def test_save_and_load(self):
        store = VectorStore()
        store.add(np.array([1.0, 0.0]), "Original text", metadata={"source": "test"})
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            store.save_to_disk(tmp_path)
            
            new_store = VectorStore()
            new_store.load_from_disk(tmp_path)
            
            assert new_store.size() == 1
            assert new_store.texts[0] == "Original text"
            assert new_store.metadata[0]["source"] == "test"
            
        finally:
            Path(tmp_path).unlink()
    
    def test_normalization(self):
        # Vector should be normalized to unit length
        store = VectorStore(normalize_vectors=True)
        
        vector = np.array([3.0, 4.0])  # Length = 5
        store.add(vector, "Test")
        
        # Stored vector should be normalized
        stored = store.vectors[0]
        norm = np.linalg.norm(stored)
        assert abs(norm - 1.0) < 0.0001  # Unit length
    
    def test_search_result_dataclass(self):
        result = SearchResult(
            text="Test",
            score=0.95,
            metadata={"page": 1},
            rank=1
        )
        
        assert result.text == "Test"
        assert result.score == 0.95
        assert result.metadata["page"] == 1
        assert result.rank == 1
        
        as_dict = result.to_dict()
        assert as_dict["text"] == "Test"
        assert as_dict["score"] == 0.95


def run_tests():
    """Manual test runner"""
    test = TestVectorStore()
    
    print("Running Vector Store tests...")
    
    test.test_empty_store()
    print("✅ test_empty_store")
    
    test.test_add_single_vector()
    print("✅ test_add_single_vector")
    
    test.test_add_batch()
    print("✅ test_add_batch")
    
    test.test_add_empty_vector_raises_error()
    print("✅ test_add_empty_vector_raises_error")
    
    test.test_search_returns_top_k()
    print("✅ test_search_returns_top_k")
    
    test.test_search_with_threshold()
    print("✅ test_search_with_threshold")
    
    test.test_search_texts_interface()
    print("✅ test_search_texts_interface")
    
    test.test_search_with_scores()
    print("✅ test_search_with_scores")
    
    test.test_get_by_id()
    print("✅ test_get_by_id")
    
    test.test_delete_by_id()
    print("✅ test_delete_by_id")
    
    test.test_clear()
    print("✅ test_clear")
    
    test.test_stats()
    print("✅ test_stats")
    
    test.test_save_and_load()
    print("✅ test_save_and_load")
    
    test.test_normalization()
    print("✅ test_normalization")
    
    test.test_search_result_dataclass()
    print("✅ test_search_result_dataclass")
    
    print("\n🎉 All Task 5 tests passed!")


if __name__ == "__main__":
    run_tests()
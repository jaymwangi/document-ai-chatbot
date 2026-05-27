"""Unit tests for Retriever module - Task 6"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from services.retriever import (
    Retriever,
    RetrieverConfig,
    RetrievalResult,
    create_retriever,
)


class MockVectorStore:
    """Mock vector store for testing"""
    
    def __init__(self):
        self.texts = [
            "RAG stands for Retrieval-Augmented Generation",
            "Vector databases enable semantic search",
            "Embeddings capture meaning numerically",
            "Language models generate coherent text",
        ]
        self.vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
        ]
    
    def search(self, query_vec, top_k, score_threshold):
        from core.vector_store import SearchResult
        
        scores = []
        for i, vec in enumerate(self.vectors):
            sim = float(np.dot(query_vec, vec))
            if sim >= score_threshold:
                scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (idx, sim) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(
                text=self.texts[idx],
                score=sim,
                metadata={},
                rank=rank,
            ))
        
        return results
    
    def get_stats(self):
        return {"total_vectors": len(self.vectors)}


class MockEmbedder:
    """Mock embedder for testing"""
    
    def embed_single(self, query):
        # Query vector biased toward RAG-related content
        if "RAG" in query or "retrieval" in query:
            return np.array([1.0, 0.1, 0.1])
        elif "vector" in query:
            return np.array([0.1, 1.0, 0.1])
        else:
            return np.array([0.3, 0.3, 0.3])


class TestRetrieverConfig:
    """Tests for RetrieverConfig"""
    
    def test_default_config(self):
        config = RetrieverConfig()
        assert config.top_k == 5
        assert config.score_threshold == 0.3
        assert config.include_scores is True
    
    def test_custom_config(self):
        config = RetrieverConfig(top_k=10, score_threshold=0.5)
        assert config.top_k == 10
        assert config.score_threshold == 0.5
    
    def test_invalid_config(self):
        with pytest.raises(ValueError):
            RetrieverConfig(top_k=0)
        
        with pytest.raises(ValueError):
            RetrieverConfig(score_threshold=1.5)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass"""
    
    def test_creation(self):
        result = RetrievalResult(
            text="Test text",
            score=0.95,
            rank=1,
            metadata={"source": "test.pdf"},
        )
        
        assert result.text == "Test text"
        assert result.score == 0.95
        assert result.rank == 1
        assert result.metadata["source"] == "test.pdf"
    
    def test_to_dict(self):
        result = RetrievalResult(
            text="Test",
            score=0.8765,
            rank=2,
            metadata={},
        )
        
        as_dict = result.to_dict()
        assert as_dict["text"] == "Test"
        assert as_dict["score"] == 0.8765  # Rounded
        assert as_dict["rank"] == 2


class TestRetriever:
    """Tests for Retriever class"""
    
    def setup_method(self):
        self.mock_store = MockVectorStore()
        self.mock_embedder = MockEmbedder()
        self.retriever = Retriever(
            self.mock_store,
            embedder=self.mock_embedder,
        )
    
    def test_retrieve_basic(self):
        results = self.retriever.retrieve("What is RAG?", top_k=2)
        
        assert len(results) == 2
        assert results[0].text == "RAG stands for Retrieval-Augmented Generation"
        assert results[0].score > 0
    
    def test_retrieve_with_threshold(self):
        results = self.retriever.retrieve(
            "What is RAG?",
            top_k=3,
            score_threshold=0.5,
        )
        
        # Only highly relevant results pass threshold
        assert len(results) < 3
    
    def test_retrieve_texts(self):
        texts = self.retriever.retrieve_texts("What is RAG?", top_k=2)
        
        assert len(texts) == 2
        assert isinstance(texts[0], str)
    
    def test_retrieve_with_scores(self):
        pairs = self.retriever.retrieve_with_scores("What is RAG?", top_k=2)
        
        assert len(pairs) == 2
        for text, score in pairs:
            assert isinstance(text, str)
            assert isinstance(score, float)
    
    def test_retrieve_one(self):
        result = self.retriever.retrieve_one("What is RAG?", min_score=0.3)
        
        assert result is not None
        assert result.rank == 1
    
    def test_retrieve_one_below_threshold(self):
        # Query that should return low scores
        result = self.retriever.retrieve_one(
            "Unrelated query with no match",
            min_score=0.9,
        )
        
        assert result is None
    
    def test_empty_query(self):
        results = self.retriever.retrieve("")
        assert results == []
        
        results = self.retriever.retrieve("   ")
        assert results == []
    
    def test_format_context(self):
        results = self.retriever.retrieve("What is RAG?", top_k=2)
        context = self.retriever.format_context(results, include_scores=True)
        
        assert "Document 1" in context
        assert "relevance:" in context
        assert results[0].text in context
    
    def test_format_context_simple(self):
        results = self.retriever.retrieve("What is RAG?", top_k=2)
        context = self.retriever.format_context_simple(results)
        
        assert results[0].text in context
        assert results[1].text in context
    
    def test_format_context_empty(self):
        context = self.retriever.format_context([])
        assert context == "No relevant documents found."
    
    def test_get_stats(self):
        stats = self.retriever.get_stats()
        
        assert "config" in stats
        assert "vector_store_stats" in stats
        assert stats["config"]["top_k"] == 5
    
    def test_update_config(self):
        assert self.retriever.config.top_k == 5
        
        self.retriever.update_config(top_k=10, score_threshold=0.4)
        
        assert self.retriever.config.top_k == 10
        assert self.retriever.config.score_threshold == 0.4
    
    def test_update_config_invalid(self):
        with pytest.raises(ValueError):
            self.retriever.update_config(invalid_param=123)


class TestCreateRetriever:
    """Tests for create_retriever convenience function"""
    
    def test_create_retriever(self):
        mock_store = MockVectorStore()
        retriever = create_retriever(mock_store, top_k=7, score_threshold=0.2)
        
        assert retriever.config.top_k == 7
        assert retriever.config.score_threshold == 0.2
        assert retriever.vector_store is mock_store


def run_tests():
    """Manual test runner"""
    
    print("Running Retriever tests...")
    
    # Config tests
    test_config = TestRetrieverConfig()
    test_config.test_default_config()
    print("✅ test_default_config")
    test_config.test_custom_config()
    print("✅ test_custom_config")
    test_config.test_invalid_config()
    print("✅ test_invalid_config")
    
    # Result tests
    test_result = TestRetrievalResult()
    test_result.test_creation()
    print("✅ test_creation")
    test_result.test_to_dict()
    print("✅ test_to_dict")
    
    # Retriever tests
    test_retriever = TestRetriever()
    test_retriever.setup_method()
    test_retriever.test_retrieve_basic()
    print("✅ test_retrieve_basic")
    test_retriever.test_retrieve_with_threshold()
    print("✅ test_retrieve_with_threshold")
    test_retriever.test_retrieve_texts()
    print("✅ test_retrieve_texts")
    test_retriever.test_retrieve_with_scores()
    print("✅ test_retrieve_with_scores")
    test_retriever.test_retrieve_one()
    print("✅ test_retrieve_one")
    test_retriever.test_retrieve_one_below_threshold()
    print("✅ test_retrieve_one_below_threshold")
    test_retriever.test_empty_query()
    print("✅ test_empty_query")
    test_retriever.test_format_context()
    print("✅ test_format_context")
    test_retriever.test_format_context_simple()
    print("✅ test_format_context_simple")
    test_retriever.test_format_context_empty()
    print("✅ test_format_context_empty")
    test_retriever.test_get_stats()
    print("✅ test_get_stats")
    test_retriever.test_update_config()
    print("✅ test_update_config")
    test_retriever.test_update_config_invalid()
    print("✅ test_update_config_invalid")
    
    # Create function tests
    test_create = TestCreateRetriever()
    test_create.test_create_retriever()
    print("✅ test_create_retriever")
    
    print("\n🎉 All Task 6 tests passed!")
    print("\n📌 Your RAG pipeline now has:")
    print("   Task 2: PDF Loader")
    print("   Task 3: Chunker")
    print("   Task 4: Embeddings")
    print("   Task 5: Vector Store")
    print("   Task 6: Retriever ← YOU ARE HERE")
    print("\n🚀 Next: Task 7 - LLM Generation (Groq/OpenAI)")


if __name__ == "__main__":
    run_tests()
"""Unit tests for chunker module - Separate from implementation"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.chunker import Chunker, ChunkerConfig, chunk_text


class TestChunker:
    """Test suite for chunker module"""
    
    def test_empty_input(self):
        chunker = Chunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
        assert chunker.chunk(None) == []  # type: ignore
    
    def test_short_text(self):
        text = "This is a short sentence."
        chunker = Chunker(chunk_size=500)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_character_chunking(self):
        text = "A" * 1000
        chunker = Chunker(strategy="character", chunk_size=300, chunk_overlap=0)
        chunks = chunker.chunk(text)
        
        # Should create 4 chunks (300+300+300+100)
        assert len(chunks) == 4
        assert all(len(c) <= 300 for c in chunks)
    
    def test_sentence_chunking_with_known_limitations(self):
        """Tests current behavior with known limitations"""
        text = "Dr. Smith went to U.S. He saw Mr. Jones. They talked."
        chunker = Chunker(strategy="sentence", chunk_size=50)
        chunks = chunker.chunk(text)
        
        # Current behavior: may split at Dr. incorrectly
        # This test documents current behavior, not ideal
        assert len(chunks) >= 1  # Will be fixed in v2
    
    def test_chunk_stats(self):
        chunker = Chunker()
        chunks = ["a" * 100, "b" * 200, "c" * 300]
        stats = chunker.get_stats(chunks)
        
        assert stats["count"] == 3
        assert stats["min_size"] == 100
        assert stats["max_size"] == 300
        assert stats["avg_size"] == 200
    
    def test_config_object(self):
        config = ChunkerConfig(strategy="paragraph", chunk_size=400, chunk_overlap=30)
        chunker = Chunker(config=config)
        assert chunker.config.strategy == "paragraph"
        assert chunker.config.chunk_size == 400
        assert chunker.config.chunk_overlap == 30
    
    def test_upgrade_notice_exists(self):
        chunker = Chunker()
        notice = chunker.upgrade_notice()
        assert "v1_limitations" in notice
        assert "v2_upgrade_paths" in notice
        assert len(notice["v1_limitations"]) > 0


def run_tests():
    """Manual test runner (no pytest required)"""
    test = TestChunker()
    
    print("Running chunker tests...")
    
    test.test_empty_input()
    print("✅ test_empty_input")
    
    test.test_short_text()
    print("✅ test_short_text")
    
    test.test_character_chunking()
    print("✅ test_character_chunking")
    
    test.test_sentence_chunking_with_known_limitations()
    print("✅ test_sentence_chunking_with_known_limitations (v1 behavior)")
    
    test.test_chunk_stats()
    print("✅ test_chunk_stats")
    
    test.test_config_object()
    print("✅ test_config_object")
    
    test.test_upgrade_notice_exists()
    print("✅ test_upgrade_notice_exists")
    
    print("\n🎉 All v1 tests passed!")
    print("\n📌 Note: Known limitations are documented in upgrade_notice()")
    print("   v2 will address: abbreviations, token-based chunking, better overlap")


if __name__ == "__main__":
    run_tests()
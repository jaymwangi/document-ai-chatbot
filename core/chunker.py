"""
Chunker Module - Task 3 of RAG Pipeline

Responsibility: Split large text into small, searchable chunks.
Single responsibility: Raw text → List[chunks]. Nothing more.

Design Philosophy:
- Keep it simple for v1 (Day 1-3)
- Provide clear upgrade paths for v2
- No external dependencies beyond standard library
- Tests are separate (tests/test_chunker.py)

Upgrade Paths (for v2/v3):
- Replace simple sentence splitter with nltk or spacy
- Add token-based chunking (better for LLM context windows)
- Add semantic chunking (embedding-based boundaries)
"""

import re
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    CHARACTER = "character"      # Simple: split by character count
    SENTENCE = "sentence"        # Better: respect sentence boundaries
    PARAGRAPH = "paragraph"      # Preserves natural document structure


class ChunkerConfig:
    """Centralized configuration for chunker"""
    def __init__(
        self,
        strategy: str = "sentence",
        chunk_size: int = 600,          # Unified parameter name
        chunk_overlap: int = 50,         # Unified parameter name
        min_chunk_size: int = 100,
        respect_sentence_boundaries: bool = True,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundaries = respect_sentence_boundaries


class Chunker:
    """
    Configurable chunker with multiple strategies.
    
    v1: Simple character and sentence chunking
    v2: Will add NLTK/spacy integration
    v3: Will add semantic chunking
    
    Usage:
        chunker = Chunker(strategy="sentence", chunk_size=600)
        chunks = chunker.chunk(text)
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None, **kwargs):
        """
        Initialize chunker with configuration.
        
        Args:
            config: ChunkerConfig object (preferred)
            **kwargs: Individual config parameters (for simplicity)
                - strategy: "character", "sentence", or "paragraph"
                - chunk_size: Target size in characters (unified)
                - chunk_overlap: Overlap between chunks
                - min_chunk_size: Minimum chunk size for merging
        """
        if config:
            self.config = config
        else:
            self.config = ChunkerConfig(**kwargs)
        
        # Strategy routing
        self._strategies: Dict[str, Callable] = {
            "character": self._chunk_by_characters,
            "sentence": self._chunk_by_sentences,
            "paragraph": self._chunk_by_paragraphs,
        }
    
    def chunk(self, text: str) -> List[str]:
        """
        Main chunking interface.
        
        Args:
            text: Raw input text
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
        
        # Clean text first (deterministic, non-semantic)
        text = self._clean_text(text)
        
        # Route to selected strategy
        chunk_func = self._strategies.get(self.config.strategy, self._chunk_by_sentences)
        return chunk_func(text)
    
    def _clean_text(self, text: str) -> str:
        """Light, deterministic cleaning before chunking"""
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Collapse excessive newlines (but preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive spaces (but preserve intentional indentation)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Fixed-size character chunking with overlap.
        
        v1: Simple character-based
        v2: Will add token-based overlap
        """
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        overlap = self.config.chunk_overlap
        
        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)
            
            # Find natural break near the end (v1: simple spaces/punctuation)
            if end < text_len and self.config.respect_sentence_boundaries:
                lookback = min(50, self.config.chunk_size // 5)
                for i in range(end, max(start, end - lookback), -1):
                    if text[i] in [' ', '\n', '.', '!', '?', ';', ':']:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            if overlap > 0 and end < text_len:
                start = end - overlap
            else:
                start = end
        
        return chunks
    
    def _split_sentences_v1(self, text: str) -> List[str]:
        """
        v1: Simple sentence splitter.
        
        KNOWN LIMITATIONS (will be fixed in v2):
        - Fails on abbreviations (Dr., U.S., etc.)
        - Requires capital letters after punctuation
        - Doesn't handle quotes properly
        - Breaks on bullet points
        
        v2 UPGRADE PATH:
        - Replace with: nltk.sent_tokenize(text)
        - Or with: spacy.load("en_core_web_sm").create_pipe("sentencizer")
        """
        # Simple pattern: punctuation + space + capital letter
        # This works for ~85% of cases. v2 will fix the rest.
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Sentence-aware chunking.
        
        v1: Simple sentence splitter (fast, no dependencies)
        v2: Will add NLTK/spacy for better accuracy
        v3: Will add token-based chunking for LLM context windows
        """
        sentences = self._split_sentences_v1(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Handle long sentences (rare with good splitter)
            if sentence_size > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Fallback to character chunking for this sentence
                sub_chunks = self._chunk_by_characters(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Add to current chunk
            if current_size + sentence_size + 1 > self.config.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
        
        # Handle remaining
        if current_chunk:
            final_chunk = ' '.join(current_chunk)
            # Merge tiny last chunk with previous
            if len(final_chunk) < self.config.min_chunk_size and chunks:
                chunks[-1] = chunks[-1] + ' ' + final_chunk
            else:
                chunks.append(final_chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Paragraph-aware chunking"""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if para_size > self.config.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Use sentence chunking for long paragraphs
                sub_chunks = self._chunk_by_sentences(para)
                chunks.extend(sub_chunks)
                continue
            
            if current_size + para_size + 2 > self.config.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def get_stats(self, chunks: List[str]) -> Dict[str, Any]:
        """Get statistics about chunks for debugging"""
        if not chunks:
            return {
                "count": 0,
                "avg_size": 0.0,
                "min_size": 0,
                "max_size": 0,
                "total_chars": 0,
                "strategy": self.config.strategy,
                "chunk_size": self.config.chunk_size,
            }
        
        sizes = [len(c) for c in chunks]
        return {
            "count": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes), 1),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes),
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
        }
    
    def upgrade_notice(self) -> Dict[str, List[str]]:
        """
        Returns known limitations and upgrade paths.
        Used for documentation and future planning.
        """
        return {
            "v1_limitations": [
                "Sentence splitter fails on abbreviations (Dr., U.S.)",
                "Requires capital letters after punctuation",
                "No token-based chunking for LLM context windows",
                "Overlap only applies to character strategy",
            ],
            "v2_upgrade_paths": [
                "Replace _split_sentences_v1() with nltk.sent_tokenize",
                "Add token-based chunking (tiktoken for OpenAI models)",
                "Add configurable overlap for sentence strategy",
                "Add semantic chunking (embedding-based boundaries)",
            ],
            "v3_enhancements": [
                "Add recursive chunking for nested structures",
                "Add metadata preservation (source, page number, section)",
                "Add chunk deduplication",
                "Add streaming chunking for very large documents",
            ],
        }


# ========== SIMPLE FUNCTION INTERFACE ==========

def chunk_text(
    text: str,
    strategy: str = "sentence",
    chunk_size: int = 600,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Simple function interface for chunking.
    
    Args:
        text: Raw text to chunk
        strategy: "character", "sentence", or "paragraph"
        chunk_size: Target chunk size in characters (unified)
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunker = Chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunker.chunk(text)


# ========== MODULE SELF-TEST (quick verification only) ==========
# For full tests, run: pytest tests/test_chunker.py
# This self-test is minimal and does not replace unit tests

if __name__ == "__main__":
    # This runs when you do: python -m core.chunker
    # It's for quick verification, not production testing
    
    print("=" * 60)
    print("✂️  Chunker - Quick Self-Test")
    print("=" * 60)
    
    # Simple test (no external dependencies)
    test_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    
    print(f"\n📝 Test: '{test_text}'")
    
    chunker = Chunker(strategy="sentence", chunk_size=30)
    chunks = chunker.chunk(test_text)
    stats = chunker.get_stats(chunks)
    
    print(f"\n✅ Created {stats['count']} chunks")
    print(f"   Avg size: {stats['avg_size']} chars")
    print(f"   Strategy: {stats['strategy']}")
    
    # Show upgrade notice
    print("\n📌 Known Limitations (v1):")
    for limit in chunker.upgrade_notice()["v1_limitations"][:3]:
        print(f"   ⚠️  {limit}")
    
    print("\n" + "=" * 60)
    print("✅ Chunker v1 ready for use")
    print("   Import: from core.chunker import Chunker, chunk_text")
    print("   Full tests: pytest tests/test_chunker.py")
    print("=" * 60)
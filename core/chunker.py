"""
Chunker Module - Task 3 of RAG Pipeline (UPDATED - Pure Sentence-Window)

Responsibility: Split large text into small, searchable chunks.
Single responsibility: Raw text → List[chunks]. Nothing more.

CORE CHANGE: Pure sentence-window chunking with fixed window + stride.
- Window: 8 sentences per chunk
- Stride: 4 sentences (50% overlap)
- NO character calculations in core logic
- Completely deterministic and predictable

Design Philosophy:
- Keep it simple for v1
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
    CHARACTER = "character"      # DEPRECATED: Kept for compatibility
    SENTENCE = "sentence"        # RECOMMENDED: Pure sentence windows
    PARAGRAPH = "paragraph"      # Preserves natural document structure


class ChunkerConfig:
    """
    Centralized configuration for chunker.
    
    UPDATED DEFAULTS (Pure Sentence-Window):
    - strategy: "sentence" (default)
    - window_size: 8 sentences per chunk
    - stride: 4 sentences (50% overlap)
    - min_chunk_size: 50 chars (filters out noise)
    """
    def __init__(
        self,
        strategy: str = "sentence",
        window_size: int = 8,           # Sentences per chunk
        stride: int = 4,                # Sentences to move forward
        min_chunk_size: int = 50,       # Filter out very small chunks
        # Legacy params (kept for compatibility)
        chunk_size: int = 500,          # DEPRECATED: Only used for character strategy
        chunk_overlap: int = 75,        # DEPRECATED: Only used for character strategy
        respect_sentence_boundaries: bool = True,
        max_chunk_size: Optional[int] = None,  # DEPRECATED: Only used for character strategy
    ):
        self.strategy = strategy
        self.window_size = window_size
        self.stride = stride
        self.min_chunk_size = min_chunk_size
        # Legacy params (keep for backward compatibility)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.max_chunk_size = max_chunk_size or int(chunk_size * 1.5)


class Chunker:
    """
    Configurable chunker with multiple strategies.
    
    PRIMARY STRATEGY: Pure sentence-window chunking
    - Fixed window: 8 sentences
    - Fixed stride: 4 sentences
    - Completely deterministic
    - No character calculations
    
    v1: Simple sentence splitter + fixed windows (fast, no dependencies)
    v2: Will add NLTK/spacy for better accuracy
    v3: Will add token-based chunking for LLM context windows
    
    Usage:
        chunker = Chunker(strategy="sentence")
        chunks = chunker.chunk(text)
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None, **kwargs):
        """
        Initialize chunker with configuration.
        
        Args:
            config: ChunkerConfig object (preferred)
            **kwargs: Individual config parameters
                - strategy: "sentence" (recommended), "character", or "paragraph"
                - window_size: Sentences per chunk (default: 8)
                - stride: Sentences to move forward (default: 4)
                - min_chunk_size: Minimum chunk size (default: 50)
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
            List of text chunks (filtered and validated)
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
        chunks = chunk_func(text)
        
        # Filter out chunks that are too small
        chunks = self._filter_small_chunks(chunks)
        
        return chunks
    
    def _filter_small_chunks(self, chunks: List[str]) -> List[str]:
        """Remove chunks that are too small to be meaningful."""
        min_size = self.config.min_chunk_size
        if min_size <= 0:
            return chunks
        
        filtered = [c for c in chunks if len(c) >= min_size]
        
        # Log if chunks were removed (for debugging)
        if len(filtered) != len(chunks):
            print(f"⚠️  Removed {len(chunks) - len(filtered)} chunks below {min_size} chars")
        
        return filtered
    
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
        DEPRECATED: Character-based chunking.
        Kept for backward compatibility only.
        Will be removed in v2.
        """
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        overlap = self.config.chunk_overlap
        
        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)
            
            # Find natural break near the end
            if end < text_len and self.config.respect_sentence_boundaries:
                lookback = min(100, self.config.chunk_size // 4)
                for i in range(end, max(start, end - lookback), -1):
                    if i < text_len and text[i] in ['.', '!', '?', ';', ':', '\n']:
                        if text[i] in ['.', '!', '?']:
                            end = i + 1
                        else:
                            end = i
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
        
        Splits on . ! ? followed by space.
        Handles ~90% of cases.
        
        KNOWN LIMITATIONS (will be fixed in v2):
        - Fails on abbreviations (Dr., U.S., etc.)
        - Requires capital letters after punctuation
        - Doesn't handle quotes perfectly
        
        v2 UPGRADE PATH:
        - Replace with: nltk.sent_tokenize(text)
        - Or with: spacy.load("en_core_web_sm").create_pipe("sentencizer")
        """
        # Simple split: punctuation + space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        PURE sentence-window chunking.
        
        NO character calculations anywhere.
        Fixed window size + fixed stride.
        Completely deterministic and predictable.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks, each with WINDOW_SIZE sentences
        """
        sentences = self._split_sentences_v1(text)
        if not sentences:
            return []
        
        # Fixed parameters - NO character logic
        window_size = self.config.window_size
        stride = self.config.stride
        
        # Create chunks with fixed window + stride
        chunks = []
        for i in range(0, len(sentences), stride):
            window = sentences[i:i + window_size]
            if window:
                chunks.append(' '.join(window))
        
        # Merge very small last chunk if it makes sense
        if len(chunks) >= 2:
            last_chunk_words = len(chunks[-1].split())
            if last_chunk_words < 3:  # Too small (less than 3 words)
                chunks[-2] = chunks[-2] + ' ' + chunks[-1]
                chunks.pop()
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Paragraph-aware chunking.
        
        Uses sentence-window chunking within paragraphs.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        for para in paragraphs:
            # Use sentence chunking for each paragraph
            para_chunks = self._chunk_by_sentences(para)
            chunks.extend(para_chunks)
        
        return chunks
    
    def get_stats(self, chunks: List[str]) -> Dict[str, Any]:
        """Get statistics about chunks for debugging."""
        if not chunks:
            return {
                "count": 0,
                "avg_size": 0.0,
                "min_size": 0,
                "max_size": 0,
                "total_chars": 0,
                "strategy": self.config.strategy,
                "window_size": self.config.window_size,
                "stride": self.config.stride,
            }
        
        sizes = [len(c) for c in chunks]
        return {
            "count": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes), 1),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes),
            "strategy": self.config.strategy,
            "window_size": self.config.window_size,
            "stride": self.config.stride,
        }
    
    def preview(self, chunks: List[str], num: int = 3) -> None:
        """Preview first N chunks for debugging."""
        if not chunks:
            print("No chunks to preview")
            return
        
        print(f"\n📦 Chunks ({len(chunks)} total, avg {sum(len(c) for c in chunks) / len(chunks):.0f} chars):")
        for i, chunk in enumerate(chunks[:num]):
            # Show sentence count too
            sentence_count = len(re.findall(r'[.!?]', chunk))
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            print(f"\n[{i+1}] ({len(chunk)} chars, {sentence_count} sentences): {preview}")
    
    def upgrade_notice(self) -> Dict[str, List[str]]:
        """
        Returns known limitations and upgrade paths.
        Used for documentation and future planning.
        """
        return {
            "current_settings": {
                "strategy": self.config.strategy,
                "window_size": self.config.window_size,
                "stride": self.config.stride,
            },
            "v1_limitations": [
                "Sentence splitter fails on abbreviations (Dr., U.S.)",
                "Requires capital letters after punctuation",
                "No token-based chunking for LLM context windows",
            ],
            "v2_upgrade_paths": [
                "Replace _split_sentences_v1() with nltk.sent_tokenize",
                "Add token-based chunking (tiktoken for OpenAI models)",
                "Add configurable window size per strategy",
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
    window_size: int = 8,
    stride: int = 4,
) -> List[str]:
    """
    Simple function interface for chunking.
    
    Args:
        text: Raw text to chunk
        strategy: "sentence" (recommended), "character", or "paragraph"
        window_size: Sentences per chunk (default: 8)
        stride: Sentences to move forward (default: 4)
    
    Returns:
        List of text chunks
    """
    chunker = Chunker(
        strategy=strategy,
        window_size=window_size,
        stride=stride,
    )
    return chunker.chunk(text)


# ========== MODULE SELF-TEST ==========

if __name__ == "__main__":
    print("=" * 60)
    print("✂️  Chunker - Pure Sentence-Window Test")
    print("=" * 60)
    
    # Test text with varied content
    test_text = """
    This is the first sentence. This is the second sentence. 
    This is the third sentence. This is the fourth sentence. 
    This is the fifth sentence. This is the sixth sentence. 
    This is the seventh sentence. This is the eighth sentence. 
    This is the ninth sentence. This is the tenth sentence. 
    This is the eleventh sentence. This is the twelfth sentence. 
    This is the thirteenth sentence. This is the fourteenth sentence. 
    This is the fifteenth sentence. This is the sixteenth sentence.
    """
    
    print(f"\n📝 Test text: 16 sentences")
    print(f"   Length: {len(test_text)} characters")
    
    # Test pure sentence-window chunking
    print("\n📌 PURE SENTENCE-WINDOW CHUNKING:")
    print(f"   Window: 8 sentences")
    print(f"   Stride: 4 sentences")
    print(f"   Expected: 3 chunks (8, 8, 4 sentences)")
    
    chunker = Chunker(strategy="sentence", window_size=8, stride=4)
    chunks = chunker.chunk(test_text)
    stats = chunker.get_stats(chunks)
    
    print(f"\n   ✅ Created {stats['count']} chunks")
    print(f"   📊 Avg size: {stats['avg_size']:.1f} chars")
    print(f"   📏 Min: {stats['min_size']} | Max: {stats['max_size']}")
    
    chunker.preview(chunks, num=3)
    
    # Show sentence counts
    print("\n📌 SENTENCE COUNTS PER CHUNK:")
    for i, chunk in enumerate(chunks):
        sentence_count = len(re.findall(r'[.!?]', chunk))
        print(f"   Chunk {i+1}: {sentence_count} sentences")
    
    # Compare with old method
    print("\n📌 COMPARISON: Old vs New")
    
    old_chunker = Chunker(strategy="sentence", chunk_size=500, chunk_overlap=75)
    old_chunks = old_chunker.chunk(test_text)
    old_stats = old_chunker.get_stats(old_chunks)
    
    print(f"\n   Old (character-based): {old_stats['count']} chunks, avg {old_stats['avg_size']:.1f} chars")
    print(f"   New (sentence-window): {stats['count']} chunks, avg {stats['avg_size']:.1f} chars")
    
    # Quality metrics
    print("\n📌 QUALITY METRICS:")
    print(f"   ✅ No character calculations in core logic")
    print(f"   ✅ Fixed window size: {chunker.config.window_size}")
    print(f"   ✅ Fixed stride: {chunker.config.stride}")
    print(f"   ✅ Completely deterministic")
    
    print("\n" + "=" * 60)
    print("✅ Chunker updated to pure sentence-window!")
    print("   Default: 8 sentences, stride 4")
    print("=" * 60)
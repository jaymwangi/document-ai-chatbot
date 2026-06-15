"""
Query Guard Module - Pre-Retrieval Intelligence Layer for RAG

This module sits BEFORE the retriever and acts as a gatekeeper + transformer.

Responsibilities:
- Score query relevance against document context
- Decide: USE / REWRITE / SUGGEST
- Transform vague queries into retrieval-optimized questions
- Generate helpful suggestions for off-topic queries
- Prevent irrelevant queries from polluting retrieval

Architecture:
    User Query → QueryGuard → Decision → (Retrieve OR Suggest) → LLM

This dramatically improves RAG quality by ensuring only relevant,
well-formed queries reach the vector store.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =========================
# ACTION TYPES
# =========================

class QueryAction(Enum):
    """Possible actions the guard can take."""
    USE = "use"           # Query is good, pass through as-is
    REWRITE = "rewrite"   # Query needs improvement before retrieval
    SUGGEST = "suggest"   # Query is off-topic, suggest alternatives


# =========================
# QUERY DECISION DATACLASS
# =========================

@dataclass
class QueryDecision:
    """
    Structured decision from the Query Guard.
    
    This is the output that the RAG pipeline uses to decide what to do.
    
    Attributes:
        original_query: The user's original question
        relevance_score: How relevant (0-1) the query is to the document
        action: "use", "rewrite", or "suggest"
        final_query: The query to use for retrieval (if action is "use" or "rewrite")
        suggestions: List of suggested alternative questions (if action is "suggest")
        reason: Why this decision was made (for logging/debugging)
        confidence: Confidence in the decision (0-1)
    """
    original_query: str
    relevance_score: float
    action: QueryAction
    final_query: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "original_query": self.original_query,
            "relevance_score": round(self.relevance_score, 4),
            "action": self.action.value,
            "final_query": self.final_query,
            "suggestions": self.suggestions[:5],  # Limit suggestions
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }
    
    def is_valid_for_retrieval(self) -> bool:
        """Check if this decision should proceed to retrieval."""
        return self.action in [QueryAction.USE, QueryAction.REWRITE]
    
    def get_query_for_retrieval(self) -> Optional[str]:
        """Get the query to use for retrieval (if valid)."""
        if self.is_valid_for_retrieval():
            return self.final_query or self.original_query
        return None


# =========================
# QUERY RELEVANCE SCORER
# =========================

class RelevanceScorer:
    """
    Measures how related a question is to document content.
    
    Uses embedding similarity as the primary method.
    
    Scoring ranges:
        0.7 - 1.0: Highly relevant (USE as-is)
        0.3 - 0.7: Partially relevant (REWRITE)
        0.0 - 0.3: Low relevance (SUGGEST alternatives)
    """
    
    # Threshold configuration (can be tuned)
    HIGH_RELEVANCE_THRESHOLD = 0.7   # USE
    LOW_RELEVANCE_THRESHOLD = 0.3    # SUGGEST (below this)
    # Between 0.3 and 0.7: REWRITE
    
    def __init__(self, embedder=None):
        """
        Initialize relevance scorer.
        
        Args:
            embedder: Embedding model instance (uses get_embedder if None)
        """
        if embedder is None:
            from services.embeddings import get_embedder
            self._embedder = get_embedder()
        else:
            self._embedder = embedder
    
    def score(
        self, 
        query: str, 
        document_context: str,
        use_embedding: bool = True,
    ) -> float:
        """
        Score query relevance against document context.
        
        Args:
            query: User's question
            document_context: Document summary or first few chunks
            use_embedding: If True, use embedding similarity (primary method)
        
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not query or not query.strip():
            return 0.0
        
        if not document_context or not document_context.strip():
            return 0.5  # Unknown, default to medium
        
        if use_embedding:
            return self._score_by_embedding(query, document_context)
        else:
            return self._score_by_keyword_overlap(query, document_context)
    
    def _score_by_embedding(self, query: str, context: str) -> float:
        """
        Score using embedding similarity (primary method).
        
        This captures semantic meaning, not just keyword matches.
        """
        try:
            # Get embeddings
            query_embedding = self._embedder.embed_single(query)
            context_embedding = self._embedder.embed_single(context[:1000])  # Limit context length
            
            if len(query_embedding) == 0 or len(context_embedding) == 0:
                return 0.3  # Fallback to medium relevance
            
            # Cosine similarity (normalized vectors → dot product)
            similarity = float(np.dot(query_embedding, context_embedding))
            
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Embedding scoring failed: {e}")
            return 0.3  # Conservative fallback
    
    def _score_by_keyword_overlap(self, query: str, context: str) -> float:
        """
        Score using keyword overlap (lightweight fallback).
        
        Used when embeddings aren't available.
        """
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & context_words)
        score = overlap / len(query_words)
        
        return min(1.0, score)
    
    def get_relevance_category(self, score: float) -> str:
        """Get human-readable relevance category."""
        if score >= self.HIGH_RELEVANCE_THRESHOLD:
            return "high"
        elif score >= self.LOW_RELEVANCE_THRESHOLD:
            return "medium"
        else:
            return "low"


# =========================
# QUERY REWRITER
# =========================

class QueryRewriter:
    """
    Transforms vague or medium-relevance queries into retrieval-optimized questions.
    
    This is used when the relevance score is in the "REWRITE" range.
    
    Techniques:
    - Add domain-specific context
    - Clarify ambiguous terms
    - Make questions more specific
    - Align with document vocabulary
    """
    
    # Common vague patterns and their improvements
    REWRITE_PATTERNS = {
        r"how does?.*work": "How does {topic} function in the context of document retrieval?",
        r"what is?.*": "What is {topic} according to the document?",
        r"tell me about": "What does the document say about {topic}?",
        r"explain": "How does the document explain {topic}?",
    }
    
    def __init__(self, document_topics: Optional[List[str]] = None):
        """
        Initialize query rewriter.
        
        Args:
            document_topics: Key topics from the document (for context)
        """
        self.document_topics = document_topics or []
    
    def rewrite(self, query: str, context_hint: Optional[str] = None) -> str:
        """
        Rewrite a query to be more retrieval-friendly.
        
        Args:
            query: Original user query
            context_hint: Optional context to guide rewriting
        
        Returns:
            Improved query string
        """
        if not query:
            return query
        
        # Method 1: Pattern-based rewriting
        rewritten = self._pattern_rewrite(query)
        
        # Method 2: Add domain context if available
        if self.document_topics and len(rewritten.split()) < 8:
            rewritten = self._add_domain_context(rewritten)
        
        # Method 3: Make it more specific
        rewritten = self._make_more_specific(rewritten)
        
        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        
        return rewritten
    
    def _pattern_rewrite(self, query: str) -> str:
        """Apply pattern-based rewriting."""
        import re
        
        query_lower = query.lower()
        
        for pattern, template in self.REWRITE_PATTERNS.items():
            match = re.search(pattern, query_lower)
            if match:
                # Extract the main topic (everything after the pattern)
                topic = query
                for p in pattern.split(r'\?'):
                    topic = topic.replace(p, "").strip()
                if not topic:
                    topic = "this concept"
                return template.format(topic=topic)
        
        return query
    
    def _add_domain_context(self, query: str) -> str:
        """Add document domain context to the query."""
        if not self.document_topics:
            return query
        
        # Only add if query is short and missing context
        if len(query.split()) < 6:
            # Check if any topic already in query
            has_topic = any(topic.lower() in query.lower() for topic in self.document_topics[:2])
            if not has_topic and self.document_topics:
                return f"{query} regarding {self.document_topics[0]}"
        
        return query
    
    def _make_more_specific(self, query: str) -> str:
        """Make the query more specific and retrieval-friendly."""
        # Add retrieval-specific phrasing
        specificity_phrases = [
            "according to the document",
            "based on the provided information",
            "what does the document say about",
        ]
        
        # Only add if query is very short
        if len(query.split()) <= 3:
            return f"What does the document say about {query.lower()}?"
        
        return query


# =========================
# SUGGESTION ENGINE
# =========================

class SuggestionEngine:
    """
    Generates relevant questions the user SHOULD ask instead.
    
    Used when the query is off-topic (low relevance).
    
    Suggestions come from:
    - Document topics and keywords
    - Chunk clusters
    - Common RAG question templates
    """
    
    # Default question templates for RAG systems
    DEFAULT_SUGGESTIONS = [
        "What is this document about?",
        "What are the main topics discussed?",
        "Can you summarize the key points?",
        "What information does this document contain?",
    ]
    
    def __init__(self, chunk_texts: Optional[List[str]] = None):
        """
        Initialize suggestion engine.
        
        Args:
            chunk_texts: Document chunks (for extracting topics)
        """
        self.chunk_texts = chunk_texts or []
        self._extracted_topics: List[str] = []
        
        if self.chunk_texts:
            self._extract_topics()
    
    def _extract_topics(self) -> None:
        """Extract key topics from document chunks."""
        if not self.chunk_texts:
            return
        
        # Simple keyword extraction from first few chunks
        all_text = " ".join(self.chunk_texts[:5])
        words = all_text.lower().split()
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(w for w in words if w not in stopwords and len(w) > 3)
        
        # Extract top topics (limit to 10)
        self._extracted_topics = [word for word, _ in word_counts.most_common(10)]
        
        logger.info(f"Extracted topics: {self._extracted_topics[:5]}")
    
    def generate_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """
        Generate relevant suggestion questions.
        
        Args:
            query: The original off-topic query (for context)
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of suggested questions
        """
        suggestions = []
        
        # Method 1: Topic-based suggestions
        if self._extracted_topics:
            for topic in self._extracted_topics[:max_suggestions]:
                suggestions.append(f"What does the document say about {topic}?")
        
        # Method 2: Default suggestions (fallback)
        if len(suggestions) < max_suggestions:
            suggestions.extend(self.DEFAULT_SUGGESTIONS)
        
        # Return unique suggestions, limited to max_suggestions
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions[:max_suggestions]


# =========================
# QUERY GUARD (MAIN CLASS)
# =========================

class QueryGuard:
    """
    Pre-retrieval intelligence layer for RAG.
    
    This is the main entry point for query processing.
    It scores, decides, rewrites, or suggests alternatives.
    
    Usage:
        guard = QueryGuard()
        decision = guard.process(query, document_context)
        
        if decision.action == QueryAction.SUGGEST:
            return decision.suggestions
        else:
            query_to_use = decision.get_query_for_retrieval()
            results = retriever.retrieve(query_to_use)
    """
    
    def __init__(
        self,
        embedder=None,
        chunk_texts: Optional[List[str]] = None,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
        enable_rewrite: bool = True,
        enable_suggestions: bool = True,
    ):
        """
        Initialize Query Guard.
        
        Args:
            embedder: Embedding model instance
            chunk_texts: Document chunks (for topic extraction)
            high_threshold: Score above this → USE action
            low_threshold: Score below this → SUGGEST action
            enable_rewrite: Enable query rewriting
            enable_suggestions: Enable suggestion generation
        """
        self.scorer = RelevanceScorer(embedder)
        self.rewriter = QueryRewriter()
        self.suggestion_engine = SuggestionEngine(chunk_texts)
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.enable_rewrite = enable_rewrite
        self.enable_suggestions = enable_suggestions
        
        # Update thresholds in scorer
        RelevanceScorer.HIGH_RELEVANCE_THRESHOLD = high_threshold
        RelevanceScorer.LOW_RELEVANCE_THRESHOLD = low_threshold
    
    def process(
        self,
        query: str,
        document_context: str,
        document_chunks: Optional[List[str]] = None,
    ) -> QueryDecision:
        """
        Process a user query and return a decision.
        
        This is the main entry point.
        
        Args:
            query: User's question
            document_context: Document summary or concatenated first chunks
            document_chunks: Optional full chunks for suggestion generation
        
        Returns:
            QueryDecision with action, final_query, or suggestions
        """
        if not query or not query.strip():
            return QueryDecision(
                original_query=query,
                relevance_score=0.0,
                action=QueryAction.SUGGEST,
                suggestions=["Please ask a valid question about the document."],
                reason="Empty query provided",
                confidence=1.0,
            )
        
        if not document_context or not document_context.strip():
            return QueryDecision(
                original_query=query,
                relevance_score=0.5,
                action=QueryAction.USE,
                final_query=query,
                reason="No document context available, passing through",
                confidence=0.5,
            )
        
        # Step 1: Score relevance
        score = self.scorer.score(query, document_context)
        category = self.scorer.get_relevance_category(score)
        
        logger.info(f"Query: '{query[:50]}...' | Score: {score:.3f} | Category: {category}")
        
        # Step 2: Decide action based on score
        if score >= self.high_threshold:
            # HIGH RELEVANCE: Use as-is
            return QueryDecision(
                original_query=query,
                relevance_score=score,
                action=QueryAction.USE,
                final_query=query,
                reason=f"High relevance score ({score:.3f} ≥ {self.high_threshold})",
                confidence=score,
            )
        
        elif score >= self.low_threshold:
            # MEDIUM RELEVANCE: Rewrite if enabled
            if self.enable_rewrite:
                rewritten_query = self.rewriter.rewrite(query, document_context)
                return QueryDecision(
                    original_query=query,
                    relevance_score=score,
                    action=QueryAction.REWRITE,
                    final_query=rewritten_query,
                    reason=f"Medium relevance ({score:.3f}), rewritten for better retrieval",
                    confidence=score,
                )
            else:
                # Use as-is but note the low confidence
                return QueryDecision(
                    original_query=query,
                    relevance_score=score,
                    action=QueryAction.USE,
                    final_query=query,
                    reason=f"Medium relevance but rewrite disabled",
                    confidence=score,
                )
        
        else:
            # LOW RELEVANCE: Suggest alternatives
            if self.enable_suggestions:
                # Update suggestion engine with chunks if provided
                if document_chunks:
                    self.suggestion_engine.chunk_texts = document_chunks
                    self.suggestion_engine._extract_topics()
                
                suggestions = self.suggestion_engine.generate_suggestions(query)
                return QueryDecision(
                    original_query=query,
                    relevance_score=score,
                    action=QueryAction.SUGGEST,
                    suggestions=suggestions,
                    reason=f"Low relevance ({score:.3f} < {self.low_threshold}), suggesting alternatives",
                    confidence=1.0 - score,  # Higher confidence for rejection
                )
            else:
                # Fallback: use anyway but warn
                return QueryDecision(
                    original_query=query,
                    relevance_score=score,
                    action=QueryAction.USE,
                    final_query=query,
                    reason=f"Low relevance but suggestions disabled",
                    confidence=score,
                )
    
    def get_document_context(self, chunks: List[str], max_chars: int = 2000) -> str:
        """
        Generate document context summary from chunks.
        
        Args:
            chunks: List of document chunks
            max_chars: Maximum context length
        
        Returns:
            Combined context string
        """
        if not chunks:
            return ""
        
        context = "\n\n".join(chunks)
        if len(context) > max_chars:
            context = context[:max_chars]
            # Find last space to avoid cutting words
            last_space = context.rfind(' ')
            if last_space > 0:
                context = context[:last_space] + "..."
        
        return context


# =========================
# CONVENIENCE FUNCTION
# =========================

def create_query_guard(
    chunk_texts: Optional[List[str]] = None,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
) -> QueryGuard:
    """
    Quick factory for QueryGuard.
    
    Args:
        chunk_texts: Document chunks for topic extraction
        high_threshold: Score above this → USE
        low_threshold: Score below this → SUGGEST
    
    Returns:
        Configured QueryGuard instance
    """
    return QueryGuard(
        chunk_texts=chunk_texts,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )


# =========================
# MODULE SELF-TEST
# =========================

if __name__ == "__main__":
    print("=" * 60)
    print("🛡️ Query Guard Module - Pre-Retrieval Intelligence Test")
    print("=" * 60)
    
    # Mock document context
    mock_document = """
    RAG (Retrieval-Augmented Generation) is a framework that combines 
    information retrieval with large language models. It uses vector 
    embeddings and similarity search to find relevant document chunks.
    Chunking is the process of splitting documents into smaller pieces.
    The retriever finds relevant chunks, and the generator produces answers.
    """
    
    mock_chunks = [
        "RAG combines retrieval with generation.",
        "Vector embeddings enable semantic search.",
        "Chunking splits documents into pieces.",
    ]
    
    # Create query guard
    guard = create_query_guard(mock_chunks)
    
    test_queries = [
        ("What is RAG?", "highly relevant"),
        ("How do embeddings work?", "relevant"),
        ("How does this system work?", "medium relevance"),
        ("Who won the World Cup?", "off-topic"),
        ("", "empty"),
    ]
    
    for query, description in test_queries:
        print(f"\n{'─' * 40}")
        print(f"📝 Query: '{query}' ({description})")
        
        decision = guard.process(query, mock_document, mock_chunks)
        
        print(f"   Score: {decision.relevance_score:.3f}")
        print(f"   Action: {decision.action.value}")
        
        if decision.action == QueryAction.SUGGEST:
            print(f"   Suggestions:")
            for s in decision.suggestions[:3]:
                print(f"      💡 {s}")
        else:
            print(f"   Final query: {decision.final_query}")
        
        print(f"   Reason: {decision.reason}")
    
    print("\n" + "=" * 60)
    print("✅ Query Guard ready!")
    print("   Features: Relevance Scoring | Decision Engine | Rewriter | Suggestions")
    print("=" * 60)
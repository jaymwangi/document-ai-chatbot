"""
Question Generator Service - Document Analysis & Question Generation

This module handles:
- Generating answerable questions from documents (NO LLM VERIFICATION)
- Autocomplete suggestions while typing
- Follow-up questions based on conversation context
- Per-document question tracking
"""

from typing import List, Dict, Any, Optional, Set
import hashlib
import re
from collections import Counter
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================
# QUESTION STATE TRACKING - PER DOCUMENT
# ============================================================

class QuestionTracker:
    """
    Tracks questions per document with usage state.
    No LLM verification - questions are generated from document chunks.
    """
    
    def __init__(self):
        # doc_id -> {"questions": List[str], "used": Set[str], "generated_at": float}
        self._store: Dict[str, Dict[str, Any]] = {}
    
    def get_questions(self, doc_id: str) -> List[str]:
        """Get all questions for a document."""
        return self._store.get(doc_id, {}).get("questions", [])
    
    def get_used_questions(self, doc_id: str) -> Set[str]:
        """Get used questions for a document."""
        return self._store.get(doc_id, {}).get("used", set())
    
    def get_remaining_questions(self, doc_id: str) -> List[str]:
        """Get questions that haven't been used yet."""
        data = self._store.get(doc_id, {})
        all_q = data.get("questions", [])
        used = data.get("used", set())
        return [q for q in all_q if q not in used]
    
    def mark_used(self, doc_id: str, question: str) -> bool:
        """Mark a question as used. Returns True if it was newly marked."""
        if doc_id not in self._store:
            self._store[doc_id] = {"questions": [], "used": set(), "generated_at": 0}
        
        used = self._store[doc_id]["used"]
        if question in used:
            return False
        
        used.add(question)
        return True
    
    def set_questions(self, doc_id: str, questions: List[str]):
        """Set questions for a document (deduplicates)."""
        seen = set()
        unique = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        
        if doc_id not in self._store:
            self._store[doc_id] = {"questions": [], "used": set(), "generated_at": 0}
        
        self._store[doc_id]["questions"] = unique
        self._store[doc_id]["generated_at"] = time.time()
        self._store[doc_id]["needs_expansion"] = False
    
    def add_questions(self, doc_id: str, new_questions: List[str]) -> int:
        """Add new questions to existing pool. Returns number added."""
        if doc_id not in self._store:
            self._store[doc_id] = {"questions": [], "used": set(), "generated_at": 0}
        
        existing = set(self._store[doc_id]["questions"])
        added = 0
        for q in new_questions:
            if q not in existing:
                self._store[doc_id]["questions"].append(q)
                existing.add(q)
                added += 1
        
        if added > 0:
            self._store[doc_id]["generated_at"] = time.time()
            self._store[doc_id]["needs_expansion"] = False
        
        return added
    
    def needs_expansion(self, doc_id: str, threshold: int = 2) -> bool:
        """Check if the question pool needs expansion."""
        if doc_id not in self._store:
            return False
        
        remaining = self.get_remaining_questions(doc_id)
        all_q = self.get_questions(doc_id)
        
        return len(remaining) < threshold and len(all_q) > 0
    
    def get_summary(self, doc_id: str) -> Dict[str, Any]:
        """Get summary of question usage."""
        all_q = self.get_questions(doc_id)
        used = self.get_used_questions(doc_id)
        remaining = [q for q in all_q if q not in used]
        
        return {
            "total": len(all_q),
            "used": len(used),
            "remaining": len(remaining),
            "needs_expansion": self.needs_expansion(doc_id),
            "generated_at": self._store.get(doc_id, {}).get("generated_at", 0),
        }
    
    def clear(self, doc_id: str):
        """Clear all data for a document."""
        if doc_id in self._store:
            del self._store[doc_id]


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_question_tracker = QuestionTracker()


def get_question_tracker() -> QuestionTracker:
    """Get the global question tracker instance."""
    return _question_tracker


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_texts_from_pipeline(pipeline) -> List[str]:
    """Helper to get texts from pipeline."""
    if not pipeline:
        return []
    
    if hasattr(pipeline, 'vector_store') and pipeline.vector_store:
        if hasattr(pipeline.vector_store, 'texts'):
            return pipeline.vector_store.texts
    
    if hasattr(pipeline, '_get_vector_store_texts'):
        return pipeline._get_vector_store_texts()
    
    return []


def _get_document_hash(pipeline) -> str:
    """Generate a hash of document content for caching."""
    try:
        texts = _get_texts_from_pipeline(pipeline)
        if texts:
            combined = "".join(texts[:10])
            return hashlib.md5(combined.encode()).hexdigest()[:8]
    except:
        pass
    return "default"


def get_fallback_questions() -> List[str]:
    """Fallback questions when document analysis fails."""
    return [
        "What is the main topic of this document?",
        "What are the key points?",
        "Can you summarize this document?",
        "What is the purpose of this document?",
    ]


# ============================================================
# ✅ MAIN QUESTION GENERATION - NO LLM VERIFICATION
# ============================================================

def generate_possible_questions(
    pipeline, 
    max_questions: int = 8,
    doc_id: Optional[str] = None
) -> List[str]:
    """
    Generate possible questions from document content.
    NO LLM VERIFICATION - questions are extracted from document chunks.
    
    This runs once during ingestion and is cached.
    """
    if not pipeline or not pipeline.is_ready:
        return get_fallback_questions()
    
    # ✅ Check if we already have questions
    if doc_id:
        tracker = get_question_tracker()
        existing = tracker.get_questions(doc_id)
        if existing:
            return existing
    
    # ✅ Generate questions from document (NO LLM)
    questions = _generate_questions_from_chunks(pipeline, max_questions)
    
    # Store if doc_id provided
    if doc_id and questions:
        tracker = get_question_tracker()
        tracker.set_questions(doc_id, questions)
        logger.info(f"✅ Generated and cached {len(questions)} questions (NO LLM used)")
    
    return questions


def _generate_questions_from_chunks(pipeline, max_questions: int = 8) -> List[str]:
    """
    Generate questions directly from document chunks.
    NO LLM calls - uses pattern matching and text extraction.
    """
    questions = []
    seen = set()
    
    try:
        texts = _get_texts_from_pipeline(pipeline)
        
        if not texts or len(texts) == 0:
            logger.warning("⚠️ No texts found in vector store")
            return get_fallback_questions()
        
        # ✅ Combine first 15 chunks for analysis
        all_text = " ".join(texts[:15])
        
        # ✅ Method 1: Extract capitalized phrases (entities/topics)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
        for phrase in list(dict.fromkeys(capitalized))[:8]:
            if len(phrase) > 3 and len(phrase) < 40 and phrase not in seen:
                # ✅ Create question from entity
                questions.append(f"What is {phrase}?")
                seen.add(phrase)
        
        # ✅ Method 2: Extract sentences with definition patterns
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', all_text) if len(s.strip()) > 25]
        
        for sentence in sentences[:20]:
            sentence_lower = sentence.lower()
            
            # Skip if sentence is too short or generic
            if len(sentence) < 30:
                continue
            
            # ✅ Look for "is" patterns (definitions)
            if " is " in sentence_lower:
                parts = sentence.split(" is ")
                if len(parts) > 1:
                    subject = parts[0].strip()
                    subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject)
                    if len(subject) > 3 and len(subject) < 50 and subject not in seen:
                        if subject.lower() in all_text.lower():
                            questions.append(f"What is {subject}?")
                            seen.add(subject)
            
            # ✅ Look for "are" patterns
            elif " are " in sentence_lower:
                parts = sentence.split(" are ")
                if len(parts) > 1:
                    subject = parts[0].strip()
                    subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject)
                    if len(subject) > 3 and len(subject) < 50 and subject not in seen:
                        if subject.lower() in all_text.lower():
                            questions.append(f"What are {subject}?")
                            seen.add(subject)
            
            # ✅ Look for "how" patterns
            elif " how " in sentence_lower:
                how_index = sentence_lower.find(" how ")
                if how_index > 0:
                    after_how = sentence[how_index + 5:].strip()
                    if after_how and len(after_how) < 60:
                        question = f"How {after_how}?"
                        if question not in seen:
                            questions.append(question)
                            seen.add(question)
            
            # ✅ Look for "why" patterns
            elif " why " in sentence_lower:
                why_index = sentence_lower.find(" why ")
                if why_index > 0:
                    after_why = sentence[why_index + 5:].strip()
                    if after_why and len(after_why) < 60:
                        question = f"Why {after_why}?"
                        if question not in seen:
                            questions.append(question)
                            seen.add(question)
        
        # ✅ Method 3: Extract key sentences as questions
        if len(questions) < max_questions:
            for sentence in sentences[:10]:
                if len(sentence) > 30 and len(sentence) < 100:
                    # Clean up sentence
                    clean = sentence.strip()
                    if clean not in seen and len(clean) > 10:
                        # Make it a question if it doesn't end with ?
                        if not clean.endswith('?'):
                            if clean[0].isupper():
                                questions.append(f"What is the significance of {clean[:50]}?")
                            else:
                                questions.append(f"Can you explain: {clean[:50]}?")
                        else:
                            questions.append(clean)
                        seen.add(clean)
        
        # ✅ Deduplicate and limit
        questions = list(dict.fromkeys(questions))[:max_questions]
        
        if questions:
            logger.info(f"✅ Generated {len(questions)} questions from document (NO LLM)")
            return questions
        
        # ✅ Fallback
        logger.warning("⚠️ No questions generated, using fallback")
        return get_fallback_questions()
        
    except Exception as e:
        logger.warning(f"Question generation failed: {e}")
        return get_fallback_questions()


# ============================================================
# ✅ EXPAND QUESTIONS - NO LLM VERIFICATION
# ============================================================

def expand_question_pool(pipeline, doc_id: str, current_questions: List[str]) -> List[str]:
    """
    Expand the question pool with new questions.
    NO LLM calls - uses pattern matching.
    """
    if not pipeline or not pipeline.is_ready:
        return current_questions
    
    try:
        texts = _get_texts_from_pipeline(pipeline)
        if not texts:
            return current_questions
        
        all_text = " ".join(texts[:15])
        existing = set(current_questions)
        new_questions = []
        
        # ✅ Find new capitalized phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
        for phrase in list(dict.fromkeys(capitalized))[:10]:
            if len(phrase) > 3 and len(phrase) < 40:
                question = f"What is {phrase}?"
                if question not in existing:
                    new_questions.append(question)
                    existing.add(question)
        
        # ✅ Find new sentences
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', all_text) if len(s.strip()) > 30]
        for sentence in sentences[:10]:
            if " is " in sentence.lower():
                parts = sentence.split(" is ")
                if len(parts) > 1:
                    subject = parts[0].strip()
                    subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject)
                    if len(subject) > 3 and len(subject) < 50:
                        question = f"What is {subject}?"
                        if question not in existing:
                            new_questions.append(question)
                            existing.add(question)
        
        return current_questions + new_questions[:4]
        
    except Exception as e:
        logger.warning(f"Question expansion failed: {e}")
        return current_questions


# ============================================================
# AUTOCOMPLETE SUGGESTIONS (Local, No LLM)
# ============================================================

_autocomplete_cache = {}


def get_autocomplete_suggestions(
    pipeline, 
    query: str, 
    max_suggestions: int = 3
) -> List[str]:
    """
    Generate autocomplete suggestions based on partial query.
    NO LLM calls - uses local text matching.
    """
    if not pipeline or not pipeline.is_ready:
        return _get_generic_suggestions(query, max_suggestions)
    
    if len(query) < 2:
        return []
    
    # Check cache
    cache_key = f"autocomplete_{hashlib.md5(query.encode()).hexdigest()}"
    if cache_key in _autocomplete_cache:
        return _autocomplete_cache[cache_key][:max_suggestions]
    
    suggestions = []
    seen = set()
    
    try:
        texts = _get_texts_from_pipeline(pipeline)
        
        if texts:
            query_lower = query.lower()
            
            for text in texts[:15]:
                sentences = re.split(r'[.!?\n]+', text)
                
                for sentence in sentences[:3]:
                    sentence = sentence.strip()
                    if len(sentence) < 15 or len(sentence) > 150:
                        continue
                    
                    sentence_lower = sentence.lower()
                    
                    if query_lower in sentence_lower or sentence_lower.startswith(query_lower):
                        if sentence[0].isupper():
                            if " is " in sentence:
                                subject = sentence.split(" is ")[0].strip()
                                question = f"What is {subject[:50]}?"
                            elif " are " in sentence:
                                subject = sentence.split(" are ")[0].strip()
                                question = f"What are {subject[:50]}?"
                            else:
                                question = sentence[:70]
                        else:
                            question = sentence[:70]
                        
                        if question not in seen:
                            suggestions.append(question)
                            seen.add(question)
        
        if not suggestions:
            suggestions = _get_generic_suggestions(query, max_suggestions)
        
        _autocomplete_cache[cache_key] = suggestions
        return suggestions[:max_suggestions]
        
    except Exception as e:
        logger.warning(f"Autocomplete failed: {e}")
        return _get_generic_suggestions(query, max_suggestions)


def _get_generic_suggestions(query: str, max_suggestions: int = 3) -> List[str]:
    """Generate generic suggestions based on query type."""
    query_lower = query.lower()
    
    if "what" in query_lower:
        return [
            "What is this document about?",
            "What are the main points?",
            "What are the key takeaways?"
        ][:max_suggestions]
    elif "how" in query_lower:
        return [
            "How does this work?",
            "How is this explained?",
            "How can this be applied?"
        ][:max_suggestions]
    elif "why" in query_lower:
        return [
            "Why is this important?",
            "Why does this matter?",
            "Why is this explained?"
        ][:max_suggestions]
    else:
        return [
            "What is this document about?",
            "What are the key points?",
            "Can you summarize this?"
        ][:max_suggestions]


def generate_followup_questions(
    pipeline, 
    last_answer: str, 
    last_query: str, 
    top_k: int = 3
) -> List[str]:
    """Generate context-aware follow-up questions."""
    if not pipeline or not pipeline.is_ready:
        return [
            "Tell me more about this",
            "What else is important?",
            "Can you explain in more detail?",
        ]
    
    # Extract keywords from conversation (NO LLM)
    words = f"{last_query} {last_answer}".lower().split()[:80]
    stop_words = {'what', 'this', 'that', 'these', 'those', 'would', 'could', 'should', 
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'up', 'down', 'is', 'are', 'was', 'were',
                  'it', 'they', 'them', 'their', 'there', 'then', 'than', 'so', 'be',
                  'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did'}
    
    keywords = [w for w in words if len(w) > 4 and w not in stop_words]
    
    unique_keywords = []
    for w in keywords:
        if w not in unique_keywords:
            unique_keywords.append(w)
    unique_keywords = unique_keywords[:3]
    
    if not unique_keywords:
        return [
            "Tell me more about this",
            "What else is important?",
            "Can you explain in more detail?",
        ]
    
    followups = [f"Tell me more about {kw}" for kw in unique_keywords[:2]]
    followups.append("What are the key takeaways?")
    
    return followups[:top_k]


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_question_cache():
    """Clear the question generation cache."""
    global _autocomplete_cache
    _autocomplete_cache = {}


def clear_document_questions(doc_id: str):
    """Clear all question data for a document."""
    tracker = get_question_tracker()
    tracker.clear(doc_id)


# ============================================================
# ✅ REMOVED: verify_questions_against_retrieval()
# ============================================================
# This function has been removed because it was causing:
# 1. 429 Rate Limit errors
# 2. Excessive API usage
# 3. Delayed rendering
# 4. Unresponsive buttons
#
# Questions are now generated from document chunks directly,
# with NO LLM verification.
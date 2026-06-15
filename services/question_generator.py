"""
Question Generator Service - Document Analysis & Question Generation

This module handles:
- Generating answerable questions from documents
- Autocomplete suggestions while typing
- Follow-up questions based on conversation context
"""

from typing import List, Dict, Any
import hashlib
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# Cache for generated questions
_question_cache = {}
_autocomplete_cache = {}


def _get_texts_from_pipeline(pipeline) -> List[str]:
    """
    Helper to get texts from pipeline (works with both old and new orchestrator).
    
    Supports:
        - New orchestrator: pipeline.vector_store.texts
        - Old pipeline: pipeline._get_vector_store_texts()
    """
    if not pipeline:
        return []
    
    # Try new orchestrator API first
    if hasattr(pipeline, 'vector_store') and pipeline.vector_store:
        if hasattr(pipeline.vector_store, 'texts'):
            return pipeline.vector_store.texts
    
    # Try old pipeline API
    if hasattr(pipeline, '_get_vector_store_texts'):
        return pipeline._get_vector_store_texts()
    
    return []


def generate_possible_questions(pipeline, max_questions: int = 5) -> List[str]:
    """
    Generate possible questions that CAN be answered from the document.
    Displays as clickable buttons below chat input.
    """
    if not pipeline or not pipeline.is_ready:
        return get_fallback_questions()
    
    # Check cache
    doc_hash = _get_document_hash(pipeline)
    cache_key = f"possible_questions_{doc_hash}"
    if cache_key in _question_cache:
        return _question_cache[cache_key][:max_questions]
    
    questions = []
    seen = set()
    
    try:
        # Get document chunks using helper
        texts = _get_texts_from_pipeline(pipeline)
        
        # DEBUG: Print to console
        print(f"📄 generate_possible_questions: {len(texts)} texts found")
        
        if not texts or len(texts) == 0:
            print("⚠️ No texts found in vector store")
            return get_fallback_questions()
        
        all_text = " ".join(texts[:10])
        
        # Method 1: Extract capitalized phrases (topics/entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
        for phrase in list(dict.fromkeys(capitalized))[:8]:
            if len(phrase) > 3 and len(phrase) < 40 and phrase not in seen:
                questions.append(f"What is {phrase}?")
                seen.add(phrase)
        
        # Method 2: Extract key sentences and convert to questions
        sentences = [s.strip() for s in all_text.replace('\n', ' ').split('.') if len(s.strip()) > 30]
        
        for sentence in sentences[:15]:
            sentence_lower = sentence.lower()
            
            # Look for definition patterns
            if " is " in sentence_lower:
                parts = sentence.split(" is ")
                if len(parts) > 1:
                    subject = parts[0].strip()
                    subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject)
                    if len(subject) > 3 and len(subject) < 50 and subject not in seen:
                        questions.append(f"What is {subject}?")
                        seen.add(subject)
            
            elif " are " in sentence_lower:
                parts = sentence.split(" are ")
                if len(parts) > 1:
                    subject = parts[0].strip()
                    subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject)
                    if len(subject) > 3 and len(subject) < 50 and subject not in seen:
                        questions.append(f"What are {subject}?")
                        seen.add(subject)
        
        # If we have real questions from the document, use them
        if questions:
            questions = list(dict.fromkeys(questions))[:max_questions]
            _question_cache[cache_key] = questions
            print(f"✅ Generated {len(questions)} questions from document")
            return questions
        
        # Fallback to common questions
        print("⚠️ No questions generated from document, using fallback")
        return get_fallback_questions()
        
    except Exception as e:
        logger.warning(f"Question generation failed: {e}")
        print(f"❌ Error in generate_possible_questions: {e}")
        return get_fallback_questions()

def get_autocomplete_suggestions(pipeline, query: str, max_suggestions: int = 3) -> List[str]:
    """
    Generate autocomplete suggestions based on partial query.
    ALWAYS returns something - never fails.
    """
    # Quick return for invalid input
    if not pipeline:
        print("❌ No pipeline")
        return _get_generic_suggestions(query, max_suggestions)
    
    if not pipeline.is_ready:
        print("⚠️ Pipeline not ready")
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
        print(f"🔍 Getting suggestions for: '{query}'")
        
        # First try: Get document texts directly using helper
        texts = _get_texts_from_pipeline(pipeline)
        
        if texts:
            print(f"📄 Found {len(texts)} texts in vector store")
            query_lower = query.lower()
            
            for text in texts[:15]:
                # Split into sentences
                sentences = re.split(r'[.!?\n]+', text)
                
                for sentence in sentences[:3]:
                    sentence = sentence.strip()
                    if len(sentence) < 15 or len(sentence) > 150:
                        continue
                    
                    sentence_lower = sentence.lower()
                    
                    # Check if sentence contains the query
                    if query_lower in sentence_lower or sentence_lower.startswith(query_lower):
                        # Create a question from the sentence
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
                            print(f"   ✅ Found suggestion: {question[:50]}...")
        
        # Second try: Use retrieval if direct search didn't work
        if not suggestions:
            print("⚠️ No direct matches, trying retrieval...")
            try:
                # CORRECTED: Use retrieve(query) not retrieve_vector
                results = pipeline.retriever.retrieve(query, top_k=10)
                
                print(f"📊 Retrieved {len(results)} chunks")
                
                for r in results[:5]:
                    # Extract text from result object
                    if hasattr(r, 'text'):
                        text = r.text
                    elif isinstance(r, dict):
                        text = r.get("text", str(r))
                    else:
                        text = str(r)
                    
                    # Clean up text
                    text = text.replace('\n', ' ').strip()
                    
                    # Strategy 1: Use first sentence
                    sentences = re.split(r'[.!?\n]+', text)
                    first_sentence = sentences[0].strip() if sentences else ""
                    
                    if len(first_sentence) > 20 and len(first_sentence) < 100:
                        if first_sentence[0].isupper() and " is " in first_sentence:
                            subject = first_sentence.split(" is ")[0].strip()
                            question = f"What is {subject[:50]}?"
                        elif first_sentence[0].isupper() and " are " in first_sentence:
                            subject = first_sentence.split(" are ")[0].strip()
                            question = f"What are {subject[:50]}?"
                        else:
                            question = first_sentence[:70]
                        
                        if question not in seen:
                            suggestions.append(question)
                            seen.add(question)
                            print(f"   ✅ Found from first sentence: {question[:50]}...")
                            continue
                    
                    # Strategy 2: Use chunk preview as fallback
                    if len(text) > 30:
                        fallback = text[:70]
                        if fallback not in seen:
                            suggestions.append(fallback)
                            seen.add(fallback)
                            print(f"   ✅ Using chunk preview: {fallback[:50]}...")
                            
            except Exception as e:
                print(f"❌ Retrieval failed: {e}")
        
        # If still no suggestions, use generic ones
        if not suggestions:
            print("⚠️ No suggestions found, using generic fallback")
            suggestions = _get_generic_suggestions(query, max_suggestions)
        
        _autocomplete_cache[cache_key] = suggestions
        print(f"📝 Returning {len(suggestions)} suggestions")
        return suggestions[:max_suggestions]
        
    except Exception as e:
        logger.warning(f"Autocomplete failed: {e}")
        print(f"❌ Error in get_autocomplete_suggestions: {e}")
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
    elif "summar" in query_lower:
        return [
            "Can you summarize this document?",
            "What is the summary?",
            "Summarize the main points"
        ][:max_suggestions]
    else:
        return [
            "What is this document about?",
            "What are the key points?",
            "Can you summarize this?"
        ][:max_suggestions]


def generate_followup_questions(pipeline, last_answer: str, last_query: str, top_k: int = 3) -> List[str]:
    """
    Generate context-aware follow-up questions based on last interaction.
    """
    if not pipeline or not pipeline.is_ready:
        return [
            "Tell me more about this",
            "What else is important?",
            "Can you explain in more detail?",
        ]
    
    # Extract keywords from conversation
    words = f"{last_query} {last_answer}".lower().split()[:80]
    stop_words = {'what', 'this', 'that', 'these', 'those', 'would', 'could', 'should', 
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'up', 'down', 'is', 'are', 'was', 'were',
                  'it', 'they', 'them', 'their', 'there', 'then', 'than', 'so', 'be',
                  'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did'}
    
    keywords = [w for w in words if len(w) > 4 and w not in stop_words]
    
    # Remove duplicates while preserving order
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


def clear_question_cache():
    """Clear the question generation cache."""
    global _question_cache, _autocomplete_cache
    _question_cache = {}
    _autocomplete_cache = {}
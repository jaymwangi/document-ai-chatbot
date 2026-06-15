"""
LLM Generator Module - Task 7 of RAG Pipeline (PRODUCTION STABLE + HARDENED)

Groq model status (2025):
- ✅ RECOMMENDED: llama-3.1-8b-instant (fast, stable, great for RAG)
- ✅ HIGH QUALITY: llama-3.3-70b-versatile (better for complex tasks)
- ❌ DEPRECATED: llama3-70b-8192
- ❌ DEPRECATED: mixtral-8x7b-32768

TASK 1: Improved system prompt with strict behavioral rules
TASK 7: Prompt hardening with context boundaries and refusal enforcement
"""
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import logging

logger = logging.getLogger(__name__)


# =========================
# TASK 1: ENHANCED SYSTEM PROMPT (Global Behavior Rules)
# =========================

# These are the "laws" that govern the LLM's behavior globally
# Task 1 makes the model a strict document-grounded assistant
SYSTEM_PROMPT_TASK_1 = """You are a precise and reliable AI assistant for answering questions based ONLY on the provided context from documents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 STRICT BEHAVIOR RULES (TASK 1 - MUST FOLLOW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ GROUNDING RULE:
   → ONLY use the provided context to answer questions
   → Do NOT use your training data or external knowledge

2️⃣ REFUSAL RULE:
   → If the answer is NOT in the context, say:
     "I don't have enough information in the document to answer this."
   → Do NOT guess, infer, or make assumptions

3️⃣ ANTI-HALLUCINATION RULE:
   → Never invent facts, dates, names, or relationships
   → If uncertain, say you don't know

4️⃣ CONCISENESS RULE:
   → Be direct and brief
   → Avoid unnecessary explanations or commentary

5️⃣ AMBIGUITY RULE:
   → If the question is unclear, ask for clarification
   → Do not guess what the user meant

6️⃣ TRACEABILITY RULE:
   → When possible, indicate which source document provided the information
   → Format: "According to [source], ..."

7️⃣ SELF-AWARENESS RULE:
   → Do NOT mention that you are "using context" or "based on the provided documents"
   → Just answer as if this is your knowledge

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Remember: You are a document QA system, NOT a general chatbot.
"""

# Task 7: Hardened instruction for individual prompts
USER_PROMPT_TEMPLATE = """Answer the question using ONLY the provided context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 CONTEXT (from retrieved documents):
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION: {query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 INSTRUCTIONS (TASK 7 - PROMPT HARDENING):
- If the answer is NOT in the context above, say "I don't have enough information"
- Do NOT use any external knowledge
- Be concise and direct
- Do NOT mention that you are "using context"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANSWER:"""


# =========================
# CONFIGURATION (PRODUCTION STABLE)
# =========================

@dataclass
class LLMConfig:
    """
    Configuration for LLM generation.
    
    Attributes:
        provider: "openai", "groq", or "mock"
        model: Model name (provider-specific)
        temperature: Creativity (0 = deterministic, 1 = creative)
        max_tokens: Maximum response length
        system_prompt: Optional system instruction override (Task 1)
    """
    provider: str = "groq"
    # PRODUCTION STABLE: Using recommended model
    model: str = "llama-3.1-8b-instant"  # Fast, stable, great for RAG
    temperature: float = 0.2
    max_tokens: int = 500
    
    # TASK 1: Enhanced system prompt for strict RAG behavior
    system_prompt: str = SYSTEM_PROMPT_TASK_1
    
    # TASK 7: Enable prompt hardening features
    enable_context_boundaries: bool = True
    enable_source_citation: bool = True
    enable_refusal_behavior: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.temperature <= 1:
            raise ValueError(f"temperature must be between 0 and 1, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        
        # Production-stable Groq models only
        if self.provider == "groq":
            stable_models = [
                "llama-3.1-8b-instant",     # FAST + STABLE (BEST FOR RAG)
                "llama-3.3-70b-versatile",   # HIGHER QUALITY
            ]
            if self.model not in stable_models:
                logger.warning(f"Model {self.model} may be deprecated. Using llama-3.1-8b-instant")
                self.model = "llama-3.1-8b-instant"
                
        elif self.provider == "openai" and self.model == "gpt-4o-mini":
            pass  # Cheap and capable


# =========================
# TASK 7: PROMPT BUILDER (HARDENED)
# =========================

class PromptBuilder:
    """
    Converts retrieval results into LLM-ready prompts.
    
    TASK 7 improvements:
    - Clear context boundaries (visual separators)
    - Explicit instruction blocks
    - Refusal behavior enforcement
    - Source attribution support
    """
    
    @staticmethod
    def build(query: str, context: str, config: Optional[LLMConfig] = None) -> str:
        """
        Build a hardened prompt with clear boundaries (TASK 7).
        
        Args:
            query: User's question
            context: Formatted context from retriever
            config: LLMConfig for feature flags
        
        Returns:
            Hardened prompt string with clear instructions
        """
        use_boundaries = config.enable_context_boundaries if config else True
        
        if not context or context == "No relevant documents found.":
            return f"""❓ QUESTION: {query}

🔍 RESULT: I don't have enough information to answer this question based on the provided documents.

💡 Please ask about content that is available in the source material."""
        
        if use_boundaries:
            # TASK 7: Hardened prompt with clear boundaries
            return USER_PROMPT_TEMPLATE.format(context=context, query=query)
        else:
            # Simpler version (backward compatible)
            return f"""Answer the question using ONLY the provided context.

If the answer is NOT in the context below, say "I don't have enough information".

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    @staticmethod
    def build_with_sources(query: str, chunks: List[Dict[str, Any]], config: Optional[LLMConfig] = None) -> str:
        """
        Build prompt with explicit source attribution (TASK 7).
        
        Args:
            query: User's question
            chunks: List of retrieval results with metadata
            config: LLMConfig for feature flags
        
        Returns:
            Structured prompt with source citations
        """
        if not chunks:
            return PromptBuilder.build(query, "", config)
        
        use_citation = config.enable_source_citation if config else True
        
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", f"Document {i}")
            
            if use_citation:
                # TASK 7: Enhanced source attribution
                context_blocks.append(f"[SOURCE: {source}]\n{chunk['text']}")
            else:
                context_blocks.append(chunk['text'])
        
        context = "\n\n---\n\n".join(context_blocks)
        
        return f"""Using ONLY the provided sources, answer the question below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 SOURCES (from document retrieval):
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION: {query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 RULES:
- Only use information from the sources above
- If the sources don't contain the answer, say "I don't have enough information"
- Cite the source when possible (e.g., "According to [source]...")
- Do not add external knowledge
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANSWER:"""
    
    @staticmethod
    def build_chat_messages(
        query: str, 
        context: str, 
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[LLMConfig] = None,
    ) -> List[Dict[str, str]]:
        """
        Build message list for chat-style LLM APIs (TASK 7).
        
        Args:
            query: User's question
            context: Formatted context string
            system_prompt: Optional override for system prompt
            chat_history: Previous conversation turns
            config: LLMConfig for feature flags
        
        Returns:
            List of message dicts for API
        """
        # TASK 1: Use enhanced system prompt
        final_system = system_prompt or SYSTEM_PROMPT_TASK_1
        
        # TASK 7: Hardened user message with boundaries
        use_boundaries = config.enable_context_boundaries if config else True
        
        if use_boundaries:
            user_content = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 CONTEXT:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION: {query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 INSTRUCTIONS:
- ONLY use the context above
- If answer not in context, say "I don't have enough information"
- Be concise and direct
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        else:
            user_content = f"""CONTEXT:
{context}

QUESTION:
{query}"""
        
        messages = [{"role": "system", "content": final_system}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": user_content})
        
        return messages


# =========================
# LLM CLIENTS (PRODUCTION STABLE)
# =========================

class BaseLLMClient(ABC):
    """Abstract base for all LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from a prompt string."""
        pass
    
    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from chat messages."""
        pass


class GroqClient(BaseLLMClient):
    """
    Groq API client with production-stable models.
    
    Stable models (always available):
    - llama-3.1-8b-instant: Fast, reliable, great for RAG
    - llama-3.3-70b-versatile: Higher quality, slightly slower
    
    Deprecated models (DO NOT USE):
    - llama3-70b-8192 ❌
    - mixtral-8x7b-32768 ❌
    """
    
    # Production-stable models only
    STABLE_MODELS = [
        "llama-3.1-8b-instant",     # Primary: fast + stable
        "llama-3.3-70b-versatile",   # Fallback: higher quality
    ]
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Groq not installed. Run: pip install groq")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        """Generate from prompt string."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.generate_chat(messages)
    
    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate from chat messages with automatic fallback to stable models."""
        models_to_try = [self.config.model]
        for model in self.STABLE_MODELS:
            if model not in models_to_try:
                models_to_try.append(model)
        
        last_error = None
        
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                if model != self.config.model:
                    logger.info(f"Using {model} (fallback from {self.config.model})")
                    self.config.model = model
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                if "decommissioned" in error_msg or "deprecated" in error_msg:
                    logger.warning(f"Model {model} is deprecated, trying next...")
                    continue
                else:
                    return f"[Error calling Groq API: {error_msg}]"
        
        return f"[Error: No working Groq models. Last error: {last_error}]"


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        """Generate from prompt string."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.generate_chat(messages)
    
    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate from chat messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error calling OpenAI API: {e}]"


class MockClient(BaseLLMClient):
    """Mock client for testing without API calls."""
    
    def __init__(self, config: LLMConfig, mock_response: Optional[str] = None):
        self.config = config
        self.mock_response = mock_response
    
    def generate(self, prompt: str) -> str:
        """Return mock response."""
        if self.mock_response:
            return self.mock_response
        
        lines = prompt.split("\n")
        question = "unknown"
        for i, line in enumerate(lines):
            if "QUESTION:" in line and i + 1 < len(lines):
                question = lines[i + 1].strip()
                break
        
        return f"[MOCK] Answer to: {question}\n\nBased on the provided context, this is a simulated response for testing purposes."
    
    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """Return mock response from chat messages."""
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"[MOCK] Response to: {user_message[:100]}..."


# =========================
# GENERATOR (MAIN CLASS)
# =========================

class LLMGenerator:
    """Main RAG generation engine."""
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or LLMConfig()
        self.prompt_builder = PromptBuilder()
        
        if client:
            self.client = client
        else:
            self.client = self._create_client()
    
    def _create_client(self) -> BaseLLMClient:
        """Create LLM client based on provider."""
        if self.config.provider == "groq":
            return GroqClient(self.config)
        elif self.config.provider == "openai":
            return OpenAIClient(self.config)
        elif self.config.provider == "mock":
            return MockClient(self.config)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer from query and context string."""
        if not query or not query.strip():
            return "No question provided."
        
        # Check for refusal cases
        if not context or context == "No relevant documents found.":
            return "I don't have enough information to answer that question based on the available documents."
        
        # TASK 7: Use hardened prompt builder
        prompt = self.prompt_builder.build(query, context, self.config)
        
        if system_prompt:
            original_system = self.config.system_prompt
            self.config.system_prompt = system_prompt
            result = self.client.generate(prompt)
            self.config.system_prompt = original_system
            return result
        
        return self.client.generate(prompt)
    
    def generate_from_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        include_sources: bool = False,
    ) -> str:
        """Generate answer directly from retrieval results."""
        if not chunks:
            return "No relevant information found to answer this question."
        
        # TASK 7: Use hardened prompt with source support
        if include_sources:
            prompt = self.prompt_builder.build_with_sources(query, chunks, self.config)
        else:
            context_parts = [chunk["text"] for chunk in chunks]
            context = "\n\n---\n\n".join(context_parts)
            prompt = self.prompt_builder.build(query, context, self.config)
        
        return self.client.generate(prompt)
    
    def generate_chat(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate with chat history support."""
        messages = self.prompt_builder.build_chat_messages(
            query=query,
            context=context,
            system_prompt=self.config.system_prompt,
            chat_history=chat_history,
            config=self.config,
        )
        
        return self.client.generate_chat(messages)
    
    def get_info(self) -> Dict[str, Any]:
        """Get generator configuration."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "features": {
                "context_boundaries": self.config.enable_context_boundaries,
                "source_citation": self.config.enable_source_citation,
                "refusal_behavior": self.config.enable_refusal_behavior,
            }
        }


# =========================
# FACTORY FUNCTIONS
# =========================

def create_generator(
    provider: str = "groq",
    model: Optional[str] = None,
    temperature: float = 0.2,
    mock_response: Optional[str] = None,
    enable_hardened_prompts: bool = True,
) -> LLMGenerator:
    """
    Quick generator creation helper.
    
    Args:
        provider: "groq", "openai", or "mock"
        model: Model name (uses provider default if None)
        temperature: Generation temperature
        mock_response: Custom mock response (mock provider only)
        enable_hardened_prompts: Enable TASK 7 prompt hardening
    
    Returns:
        Configured LLMGenerator
    """
    if provider == "mock":
        config = LLMConfig(
            provider="mock", 
            temperature=temperature,
            enable_context_boundaries=enable_hardened_prompts,
        )
        client = MockClient(config, mock_response=mock_response)
        return LLMGenerator(config=config, client=client)
    
    config = LLMConfig(
        provider=provider,
        temperature=temperature,
        enable_context_boundaries=enable_hardened_prompts,
    )
    
    if model:
        config.model = model
    elif provider == "groq":
        config.model = "llama-3.1-8b-instant"
    
    return LLMGenerator(config=config)


# =========================
# MODULE SELF-TEST
# =========================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 LLM Generator - Tasks 1 & 7 Test")
    print("=" * 60)
    
    # Test with mock
    print("\n📝 Test 1: Mock generator with hardened prompts")
    generator = create_generator("mock", enable_hardened_prompts=True)
    
    test_query = "What is RAG?"
    test_context = "RAG stands for Retrieval-Augmented Generation."
    
    answer = generator.generate(test_query, test_context)
    print(f"\nQuery: {test_query}")
    print(f"Answer: {answer}")
    
    # Test 2: Refusal behavior (TASK 1)
    print("\n📝 Test 2: Refusal behavior (TASK 1)")
    test_query_outside = "What is the capital of France?"
    test_context_outside = "This document is about RAG systems only."
    
    answer_refusal = generator.generate(test_query_outside, test_context_outside)
    print(f"\nQuery: {test_query_outside}")
    print(f"Answer: {answer_refusal}")
    
    # Test 3: Source attribution (TASK 7)
    print("\n📝 Test 3: Source attribution (TASK 7)")
    test_chunks = [
        {"text": "RAG combines retrieval and generation.", "metadata": {"source": "paper1.pdf"}},
        {"text": "Vector databases enable semantic search.", "metadata": {"source": "paper2.pdf"}},
    ]
    answer_with_sources = generator.generate_from_chunks(test_query, test_chunks, include_sources=True)
    print(f"\nWith sources:\n{answer_with_sources[:200]}...")
    
    # Test 4: Show configuration
    print("\n📝 Test 4: Generator configuration")
    info = generator.get_info()
    print(f"   Provider: {info['provider']}")
    print(f"   Model: {info['model']}")
    print(f"   Features: {info['features']}")
    
    print("\n" + "=" * 60)
    print("✅ Generator module updated with:")
    print("   - TASK 1: Enhanced system prompt (7 behavioral rules)")
    print("   - TASK 7: Prompt hardening (boundaries, citations, refusal)")
    print("   - Production-stable models")
    print("=" * 60)
"""
LLM Generator Module - Task 7 of RAG Pipeline (PRODUCTION STABLE)

Groq model status (2025):
- ✅ RECOMMENDED: llama-3.1-8b-instant (fast, stable, great for RAG)
- ✅ HIGH QUALITY: llama-3.3-70b-versatile (better for complex tasks)
- ❌ DEPRECATED: llama3-70b-8192
- ❌ DEPRECATED: mixtral-8x7b-32768
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os


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
        system_prompt: Optional system instruction override
    """
    provider: str = "groq"
    # PRODUCTION STABLE: Using recommended model
    model: str = "llama-3.1-8b-instant"  # Fast, stable, great for RAG
    temperature: float = 0.2
    max_tokens: int = 500
    system_prompt: str = "You are a helpful AI assistant. Answer accurately and concisely."
    
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
                print(f"⚠️  Model {self.model} is deprecated. Using llama-3.1-8b-instant")
                self.model = "llama-3.1-8b-instant"
                
        elif self.provider == "openai" and self.model == "gpt-4o-mini":
            pass  # Cheap and capable


# =========================
# PROMPT BUILDER
# =========================

class PromptBuilder:
    """Converts retrieval results into LLM-ready prompts."""
    
    @staticmethod
    def build(query: str, context: str) -> str:
        """Build a simple prompt from query and context string."""
        if not context or context == "No relevant documents found.":
            return f"""
Question: {query}

I don't have enough information to answer this question based on the provided documents.

Please ask about content that is available in the source material.
"""
        
        return f"""Based on the following context, answer the question accurately and concisely.

If the context doesn't contain the answer, say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    @staticmethod
    def build_with_sources(query: str, chunks: List[Dict[str, Any]]) -> str:
        """Build prompt with explicit source attribution."""
        if not chunks:
            return PromptBuilder.build(query, "")
        
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", f"Document {i}")
            context_blocks.append(f"[Source: {source}]\n{chunk['text']}")
        
        context = "\n\n---\n\n".join(context_blocks)
        
        return f"""Using the provided sources, answer the question below.

SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer concisely but completely
2. Cite sources when possible
3. If information is missing, say so

ANSWER:"""
    
    @staticmethod
    def build_chat_messages(
        query: str, 
        context: str, 
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build message list for chat-style LLM APIs."""
        default_system = "You are a helpful AI assistant. Use the provided context to answer accurately. If the context lacks information, say so clearly."
        
        system_content = system_prompt or default_system
        
        user_content = f"""CONTEXT:
{context}

QUESTION:
{query}"""
        
        messages = [{"role": "system", "content": system_content}]
        
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
        # Build list of models to try (current + stable fallbacks)
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
                # If we used a different model, update config
                if model != self.config.model:
                    print(f"ℹ️  Using {model} (fallback from {self.config.model})")
                    self.config.model = model
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Handle model deprecation gracefully
                if "decommissioned" in error_msg or "deprecated" in error_msg:
                    print(f"⚠️  Model {model} is deprecated, trying next...")
                    continue
                else:
                    # Non-model error (rate limit, auth, etc.) - don't retry
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
        
        if not context or context == "No relevant documents found.":
            return "I don't have enough information to answer that question based on the available documents."
        
        prompt = self.prompt_builder.build(query, context)
        
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
        
        if include_sources:
            prompt = self.prompt_builder.build_with_sources(query, chunks)
        else:
            context_parts = [chunk["text"] for chunk in chunks]
            context = "\n\n---\n\n".join(context_parts)
            prompt = self.prompt_builder.build(query, context)
        
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
        )
        
        return self.client.generate_chat(messages)
    
    def get_info(self) -> Dict[str, Any]:
        """Get generator configuration."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }


# =========================
# FACTORY FUNCTIONS
# =========================

def create_generator(
    provider: str = "groq",
    model: Optional[str] = None,
    temperature: float = 0.2,
    mock_response: Optional[str] = None,
) -> LLMGenerator:
    """Quick generator creation helper."""
    if provider == "mock":
        config = LLMConfig(provider="mock", temperature=temperature)
        client = MockClient(config, mock_response=mock_response)
        return LLMGenerator(config=config, client=client)
    
    config = LLMConfig(
        provider=provider,
        temperature=temperature,
    )
    
    if model:
        config.model = model
    elif provider == "groq":
        # Use production-stable model
        config.model = "llama-3.1-8b-instant"
    
    return LLMGenerator(config=config)


# =========================
# MODULE SELF-TEST
# =========================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 LLM Generator - Production Stable Test")
    print("=" * 60)
    
    # Test with mock
    print("\n📝 Test 1: Mock generator")
    generator = create_generator("mock")
    
    test_query = "What is RAG?"
    test_context = "RAG stands for Retrieval-Augmented Generation."
    
    answer = generator.generate(test_query, test_context)
    print(f"\nQuery: {test_query}")
    print(f"Answer: {answer}")
    
    # Show stable models
    print("\n📝 Production-Stable Groq Models:")
    print("   ✅ llama-3.1-8b-instant (RECOMMENDED - fast, stable)")
    print("   ✅ llama-3.3-70b-versatile (higher quality)")
    print("\n❌ Deprecated Models (DO NOT USE):")
    print("   ❌ llama3-70b-8192")
    print("   ❌ mixtral-8x7b-32768")
    
    print("\n" + "=" * 60)
    print("✅ Generator module updated with production-stable models!")
    print("   Default model: llama-3.1-8b-instant")
    print("=" * 60)
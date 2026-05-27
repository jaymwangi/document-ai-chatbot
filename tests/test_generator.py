"""Unit tests for LLM Generator module - Task 7"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from services.generator import (
    LLMConfig,
    PromptBuilder,
    MockClient,
    LLMGenerator,
    create_generator,
)


class TestLLMConfig:
    """Tests for LLMConfig"""
    
    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == "groq"
        assert config.temperature == 0.2
        assert config.max_tokens == 500
    
    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            LLMConfig(temperature=1.5)
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.5)
    
    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)


class TestPromptBuilder:
    """Tests for PromptBuilder"""
    
    def test_build_with_context(self):
        query = "What is RAG?"
        context = "RAG stands for Retrieval-Augmented Generation."
        
        prompt = PromptBuilder.build(query, context)
        
        assert "QUESTION: What is RAG?" in prompt
        assert "CONTEXT:" in prompt
        assert "RAG stands for Retrieval-Augmented Generation" in prompt
    
    def test_build_without_context(self):
        query = "What is RAG?"
        context = ""
        
        prompt = PromptBuilder.build(query, context)
        
        assert "don't have enough information" in prompt.lower()
    
    def test_build_with_sources(self):
        query = "What is RAG?"
        chunks = [
            {"text": "RAG is Retrieval-Augmented Generation.", "metadata": {"source": "paper1.pdf"}},
            {"text": "It improves LLM accuracy.", "metadata": {"source": "paper2.pdf"}},
        ]
        
        prompt = PromptBuilder.build_with_sources(query, chunks)
        
        assert "[Source: paper1.pdf]" in prompt
        assert "RAG is Retrieval-Augmented Generation" in prompt
    
    def test_build_chat_messages(self):
        query = "What is AI?"
        context = "AI is artificial intelligence."
        
        messages = PromptBuilder.build_chat_messages(query, context)
        
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "CONTEXT:" in messages[1]["content"]
    
    def test_build_chat_messages_with_history(self):
        query = "What is RAG?"
        context = "RAG is Retrieval-Augmented Generation."
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        
        messages = PromptBuilder.build_chat_messages(query, context, chat_history=history)
        
        assert len(messages) == 4  # system + history (2) + user


class TestMockClient:
    """Tests for MockClient"""
    
    def test_generate(self):
        client = MockClient(LLMConfig())
        response = client.generate("What is RAG?")
        
        assert "[MOCK]" in response
        assert "RAG" in response or "question" in response.lower()
    
    def test_generate_with_custom_response(self):
        custom = "This is my custom mock response"
        client = MockClient(LLMConfig(), mock_response=custom)
        response = client.generate("Any question")
        
        assert response == custom
    
    def test_generate_chat(self):
        client = MockClient(LLMConfig())
        messages = [{"role": "user", "content": "Test message"}]
        response = client.generate_chat(messages)
        
        assert "[MOCK]" in response


class TestLLMGenerator:
    """Tests for LLMGenerator"""
    
    def test_generate_with_mock(self):
        generator = create_generator("mock")
        
        answer = generator.generate(
            query="What is RAG?",
            context="RAG combines retrieval and generation."
        )
        
        assert answer is not None
        assert isinstance(answer, str)
    
    def test_generate_empty_query(self):
        generator = create_generator("mock")
        answer = generator.generate(query="", context="Some context")
        
        assert "No question provided" in answer
    
    def test_generate_empty_context(self):
        generator = create_generator("mock")
        answer = generator.generate(query="What is RAG?", context="")
        
        assert "don't have enough information" in answer.lower() or "No relevant" in answer
    
    def test_generate_from_chunks(self):
        generator = create_generator("mock")
        chunks = [
            {"text": "RAG is Retrieval-Augmented Generation."},
            {"text": "Vector search finds relevant documents."},
        ]
        
        answer = generator.generate_from_chunks("What is RAG?", chunks)
        
        assert answer is not None
        assert isinstance(answer, str)
    
    def test_generate_from_chunks_empty(self):
        generator = create_generator("mock")
        answer = generator.generate_from_chunks("What is RAG?", [])
        
        assert "No relevant information" in answer
    
    def test_generate_from_chunks_with_sources(self):
        generator = create_generator("mock")
        chunks = [
            {"text": "RAG text", "metadata": {"source": "doc.pdf"}},
        ]
        
        answer = generator.generate_from_chunks("What is RAG?", chunks, include_sources=True)
        
        assert answer is not None
    
    def test_generate_chat(self):
        generator = create_generator("mock")
        answer = generator.generate_chat(
            query="What is RAG?",
            context="RAG stands for Retrieval-Augmented Generation.",
            chat_history=[{"role": "user", "content": "Hello"}],
        )
        
        assert answer is not None
    
    def test_get_info(self):
        generator = create_generator("mock")
        info = generator.get_info()
        
        assert "provider" in info
        assert "model" in info
        assert "temperature" in info
    
    def test_custom_system_prompt(self):
        generator = create_generator("mock")
        custom_prompt = "You are a technical expert."
        
        answer = generator.generate(
            query="What is RAG?",
            context="RAG is retrieval-augmented generation.",
            system_prompt=custom_prompt,
        )
        
        assert answer is not None


def run_tests():
    """Manual test runner"""
    
    print("Running Generator tests...")
    
    # Config tests
    test_config = TestLLMConfig()
    test_config.test_default_config()
    print("✅ test_default_config")
    test_config.test_invalid_temperature()
    print("✅ test_invalid_temperature")
    test_config.test_invalid_max_tokens()
    print("✅ test_invalid_max_tokens")
    
    # PromptBuilder tests
    test_prompt = TestPromptBuilder()
    test_prompt.test_build_with_context()
    print("✅ test_build_with_context")
    test_prompt.test_build_without_context()
    print("✅ test_build_without_context")
    test_prompt.test_build_with_sources()
    print("✅ test_build_with_sources")
    test_prompt.test_build_chat_messages()
    print("✅ test_build_chat_messages")
    test_prompt.test_build_chat_messages_with_history()
    print("✅ test_build_chat_messages_with_history")
    
    # MockClient tests
    test_mock = TestMockClient()
    test_mock.test_generate()
    print("✅ test_generate")
    test_mock.test_generate_with_custom_response()
    print("✅ test_generate_with_custom_response")
    test_mock.test_generate_chat()
    print("✅ test_generate_chat")
    
    # Generator tests
    test_gen = TestLLMGenerator()
    test_gen.test_generate_with_mock()
    print("✅ test_generate_with_mock")
    test_gen.test_generate_empty_query()
    print("✅ test_generate_empty_query")
    test_gen.test_generate_empty_context()
    print("✅ test_generate_empty_context")
    test_gen.test_generate_from_chunks()
    print("✅ test_generate_from_chunks")
    test_gen.test_generate_from_chunks_empty()
    print("✅ test_generate_from_chunks_empty")
    test_gen.test_generate_from_chunks_with_sources()
    print("✅ test_generate_from_chunks_with_sources")
    test_gen.test_generate_chat()
    print("✅ test_generate_chat")
    test_gen.test_get_info()
    print("✅ test_get_info")
    test_gen.test_custom_system_prompt()
    print("✅ test_custom_system_prompt")
    
    print("\n🎉 All Task 7 tests passed!")
    print("\n📌 Your complete RAG pipeline is now:")
    print("   Task 2: PDF Loader 📄")
    print("   Task 3: Chunker ✂️")
    print("   Task 4: Embeddings 🧠")
    print("   Task 5: Vector Store 🗂️")
    print("   Task 6: Retriever 🔍")
    print("   Task 7: Generator 🤖 ← COMPLETE!")
    print("\n🚀 Next: Task 8 - Streamlit UI or FastAPI")


if __name__ == "__main__":
    run_tests()
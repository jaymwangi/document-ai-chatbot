"""Test RAG with stable Groq models"""

from rag_pipeline import RAGPipeline

print("Initializing RAG with stable model...")
pipeline = RAGPipeline(
    llm_provider="groq",
    llm_model="llama-3.1-8b-instant",  # STABLE
    top_k=5,
    score_threshold=0.2,  # Lower threshold for better retrieval
    auto_load=True,
)

# Test questions
questions = [
    "What is this document about?",
    "What are the main topics discussed?",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print("A: ", end="", flush=True)
    answer = pipeline.ask(question)
    print(answer[:400] if len(answer) > 400 else answer)
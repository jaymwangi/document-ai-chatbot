"""
RAG Pipeline - Task 8: Complete System Integration

This is the final assembly of all RAG components into a working AI assistant.

Pipeline flow:
    User Question → Retrieve Context → Generate Answer → Return Response

This module orchestrates:
    - Embeddings (Task 4)
    - Vector Store (Task 5)
    - Retriever (Task 6)
    - Generator (Task 7)

Usage:
    pipeline = RAGPipeline()
    answer = pipeline.ask("What is RAG?")
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all RAG components
from core.pdf_loader import load_pdf, get_first_pdf
from core.chunker import Chunker, ChunkingStrategy
from services.embeddings import embed_chunks, get_embedder
from core.vector_store import VectorStore, create_vector_store_from_chunks
from services.retriever import Retriever, RetrieverConfig
from services.generator import LLMGenerator, create_generator


class RAGPipeline:
    """
    Complete RAG pipeline orchestrating all components.
    
    This is the main interface for the entire RAG system.
    
    Usage:
        # Initialize
        pipeline = RAGPipeline()
        
        # Load documents
        pipeline.ingest_documents("data/")
        
        # Ask questions
        answer = pipeline.ask("What is the document about?")
        
        # Get answer with sources
        result = pipeline.ask_with_sources("Explain the main concept")
    """
    
    def __init__(
        self,
        embedding_model: str = "mini-lm",
        llm_provider: str = "groq",
        llm_model: Optional[str] = None,  # ADDED: Model parameter
        top_k: int = 5,
        score_threshold: float = 0.2,  # Lowered default for better retrieval
        temperature: float = 0.2,
        auto_load: bool = True,
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            embedding_model: "mini-lm", "mpnet", or "openai-small"
            llm_provider: "groq", "openai", or "mock"
            llm_model: Model name (e.g., "llama-3.1-8b-instant", "gpt-4o-mini")
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            temperature: LLM creativity (0-1)
            auto_load: If True, automatically load documents from data/
        """
        print("=" * 60)
        print("🚀 Initializing RAG Pipeline...")
        print("=" * 60)
        
        # Store configuration
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # Set default model if not provided
        if llm_model is None:
            if llm_provider == "groq":
                llm_model = "llama-3.1-8b-instant"  # Production stable
            elif llm_provider == "openai":
                llm_model = "gpt-4o-mini"  # Cost-effective
            else:
                llm_model = "mock-model"
        
        # Initialize components
        print("\n📦 Component Initialization:")
        
        # 1. Embedder (Task 4)
        print("   - Loading embedder...")
        self.embedder = get_embedder(model_name=embedding_model)
        
        # 2. Vector Store (Task 5)
        print("   - Initializing vector store...")
        self.vector_store = VectorStore()
        
        # 3. Retriever (Task 6)
        print("   - Configuring retriever...")
        retriever_config = RetrieverConfig(
            top_k=top_k,
            score_threshold=score_threshold,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            config=retriever_config,
            embedder=self.embedder,
        )
        
        # 4. Generator (Task 7) - WITH MODEL PARAMETER
        print(f"   - Initializing LLM ({llm_provider}) with model: {llm_model}...")
        self.generator = create_generator(
            provider=llm_provider,
            model=llm_model,  # Pass the model parameter
            temperature=temperature,
        )
        
        # 5. Chunker (Task 3)
        self.chunker = Chunker(
            strategy="sentence",
            chunk_size=600,
            chunk_overlap=50,
        )
        
        # Track state
        self.is_ready = False
        self.document_count = 0
        self.chunk_count = 0
        
        # Auto-load documents if requested
        if auto_load:
            self.ingest_documents()
        
        print("\n" + "=" * 60)
        if self.is_ready:
            print("✅ Pipeline ready! Ask questions with pipeline.ask()")
        else:
            print("⚠️  Pipeline initialized. Load documents with pipeline.ingest_documents()")
        print("=" * 60)
    
    def ingest_documents(self, data_dir: str = "data") -> Dict[str, Any]:
        """
        Load and index all PDF documents from a directory.
        
        This is where documents become searchable.
        
        Args:
            data_dir: Directory containing PDF files
            
        Returns:
            Dictionary with ingestion statistics
        """
        print("\n" + "=" * 60)
        print("📄 Ingesting Documents")
        print("=" * 60)
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Directory not found: {data_dir}")
            return {"success": False, "error": "Directory not found"}
        
        pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {data_dir}")
            return {"success": False, "error": "No PDF files found"}
        
        print(f"📁 Found {len(pdf_files)} PDF files")
        
        all_chunks = []
        all_metadata = []
        
        for pdf_path in pdf_files:
            print(f"\n   Processing: {pdf_path.name}")
            
            # Load PDF text (Task 2)
            try:
                text = load_pdf(str(pdf_path), light_clean=True)
                print(f"      Extracted {len(text)} characters")
            except Exception as e:
                print(f"      ❌ Error loading: {e}")
                continue
            
            # Chunk text (Task 3)
            chunks = self.chunker.chunk(text)
            print(f"      Created {len(chunks)} chunks")
            
            # Store with metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": pdf_path.name,
                    "page": i // 10,  # Approximate page
                    "chunk_index": i,
                })
        
        if not all_chunks:
            print("\n❌ No chunks extracted from any documents")
            return {"success": False, "error": "No chunks extracted"}
        
        # Generate embeddings and store (Tasks 4 & 5)
        print(f"\n🧠 Generating embeddings for {len(all_chunks)} chunks...")
        
        # Embed in batches to show progress
        batch_size = 16
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            batch_metadata = all_metadata[i:i+batch_size]
            
            # Embed batch
            embedded = embed_chunks(batch, batch_size=batch_size)
            
            # Add to vector store
            for j, item in enumerate(embedded):
                item["metadata"] = batch_metadata[j]
            
            self.vector_store.add_batch(embedded)
            
            # Progress indicator
            progress = min(i + batch_size, len(all_chunks))
            print(f"   Progress: {progress}/{len(all_chunks)}", end="\r")
        
        print(f"\n\n✅ Ingestion complete!")
        print(f"   Documents: {len(pdf_files)}")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Vectors stored: {self.vector_store.size()}")
        
        self.is_ready = True
        self.document_count = len(pdf_files)
        self.chunk_count = len(all_chunks)
        
        return {
            "success": True,
            "documents": len(pdf_files),
            "chunks": len(all_chunks),
            "vectors": self.vector_store.size(),
        }
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer.
        
        This is the main interface for the RAG system.
        
        Args:
            question: User's question
            
        Returns:
            Generated answer string
        """
        if not self.is_ready:
            return "Pipeline not ready. Please load documents first with ingest_documents()."
        
        if not question or not question.strip():
            return "Please provide a valid question."
        
        # Handle generic questions by using first few chunks
        generic_questions = [
            "what is this document about", 
            "what are the key concepts", 
            "summarize this",
            "tell me about this document"
        ]
        
        if question.lower().strip() in generic_questions:
            if self.vector_store.texts:
                summary_chunks = self.vector_store.texts[:3]
                context = "\n\n".join(summary_chunks)
                return self.generator.generate(
                    f"Based on these document excerpts, what is this document about? Answer in 2-3 sentences.\n\nExcerpts:\n{context}",
                    context
                )
        
        # Step 1: Retrieve relevant chunks (Task 6)
        results = self.retriever.retrieve(question, top_k=self.top_k)
        
        if not results:
            # Fallback: return first few chunks as context
            if self.vector_store.texts:
                fallback_chunks = self.vector_store.texts[:2]
                context = "\n\n".join(fallback_chunks)
                return self.generator.generate(
                    f"Answer this question based ONLY on the provided context. If you cannot answer, say so.\n\nQuestion: {question}\n\nContext:\n{context}",
                    context
                )
            return "I couldn't find any relevant information to answer your question."
        
        # Step 2: Generate answer from context (Task 7)
        answer = self.generator.generate_from_chunks(question, [
            {"text": r.text, "metadata": r.metadata} for r in results
        ])
        
        return answer
    
    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get answer with source attribution.
        
        Useful for debugging or when you need to show sources.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and retrieval info
        """
        if not self.is_ready:
            return {
                "answer": "Pipeline not ready. Please load documents first.",
                "sources": [],
                "retrieval_info": {"error": "Not ready"},
            }
        
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "retrieval_info": {"error": "Empty question"},
            }
        
        # Retrieve chunks
        results = self.retriever.retrieve(question, top_k=self.top_k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information.",
                "sources": [],
                "retrieval_info": {"chunks_found": 0},
            }
        
        # Generate answer
        chunks_for_generation = [
            {"text": r.text, "metadata": r.metadata} for r in results
        ]
        answer = self.generator.generate_from_chunks(question, chunks_for_generation)
        
        # Prepare sources
        sources = []
        for r in results:
            sources.append({
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "score": round(r.score, 4),
                "source": r.metadata.get("source", "Unknown"),
                "rank": r.rank,
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_info": {
                "chunks_found": len(results),
                "top_score": round(results[0].score, 4) if results else 0,
                "question": question,
            },
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and statistics."""
        return {
            "ready": self.is_ready,
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "vectors": self.vector_store.size(),
            "config": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
            },
            "embedder_info": self.embedder.get_info(),
            "generator_info": self.generator.get_info(),
        }
    
    def clear(self):
        """Clear all documents from the pipeline."""
        self.vector_store.clear()
        self.is_ready = False
        self.document_count = 0
        self.chunk_count = 0
        print("🗑️ Pipeline cleared. All documents removed.")


# =========================
# QUICK TEST
# =========================

def quick_test():
    """Quick test of the complete pipeline."""
    print("\n" + "=" * 60)
    print("🧪 Quick Pipeline Test")
    print("=" * 60)
    
    # Create pipeline with mock LLM (no API key needed)
    pipeline = RAGPipeline(
        llm_provider="mock",
        auto_load=False,
    )
    
    # For demo, we need a PDF. If none exists, show instructions
    pdf_files = list(Path("data").glob("*.pdf"))
    
    if not pdf_files:
        print("\n⚠️  No PDF files found in data/ directory")
        print("   Please add a PDF to data/ and run again")
        print("\n   Or use the Streamlit UI to upload documents")
        return
    
    # Ingest documents
    result = pipeline.ingest_documents()
    
    if result["success"]:
        # Ask a question
        print("\n" + "=" * 60)
        print("💬 Testing Q&A")
        print("=" * 60)
        
        test_question = "What is this document about?"
        print(f"\n❓ Question: {test_question}")
        
        answer = pipeline.ask(test_question)
        print(f"\n🤖 Answer: {answer}")
        
        # Test with sources
        print("\n" + "-" * 40)
        print("With sources:")
        result_with_sources = pipeline.ask_with_sources(test_question)
        print(f"\nAnswer: {result_with_sources['answer'][:200]}...")
        print(f"\nSources found: {len(result_with_sources['sources'])}")
        
        # Show status
        print("\n" + "-" * 40)
        print("Pipeline Status:")
        status = pipeline.get_status()
        for key, value in status.items():
            if not isinstance(value, dict):
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()
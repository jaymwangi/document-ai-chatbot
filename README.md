# RAG Document Chatbot

A document question-answering application built with **Retrieval-Augmented Generation (RAG)**. Upload a PDF, ask questions in natural language, and receive answers generated using retrieved document passages.

<p align="center">
  <a href="https://jay-rag-chatbot.streamlit.app/"><strong>Live Demo</strong></a>
  &nbsp;•&nbsp;
  <a href="https://github.com/jaymwangi/document-ai-chatbot/blob/main/assets/demo/rag_doc_chatbot_demo.mp4">Watch the Demo</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Retrieval" src="https://img.shields.io/badge/Retrieval-FAISS%20%2B%20BM25-orange">
  <img alt="LLM Providers" src="https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI-6C5CE7">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

---

## Overview

The RAG Document Chatbot combines dense semantic retrieval with sparse keyword retrieval to find relevant passages from uploaded PDFs. Retrieved rankings are combined, and selected document context is passed to a language model to generate a response.

The project explores practical RAG engineering concerns, including document processing, hybrid retrieval, reranking, caching, request coordination, source attribution, and pipeline debugging.

## Key Features

- **PDF processing:** Extract text from uploaded PDF documents.
- **Text chunking:** Split extracted text into overlapping passages to retain surrounding context.
- **Semantic search:** Generate Sentence Transformer embeddings and retrieve candidates using FAISS.
- **Keyword search:** Use BM25 to retrieve passages based on lexical matches.
- **Hybrid retrieval:** Combine dense and sparse rankings using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** Refine the ordering of retrieved candidates.
- **LLM integration:** Generate responses using Groq or OpenAI.
- **Source attribution:** Inspect retrieved passages and available retrieval scores.
- **Caching and persistence:** Reuse cached embeddings and stored vector-related artifacts where supported by the implementation.
- **Request coordination:** Queue-based request handling and single-flight execution.
- **Resilience and observability:** Rate-limit handling, pipeline timing, and retrieval debugging.
- **Question assistance:** Generate document-aware questions and suggest follow-ups.
- **Interactive interface:** Chat with uploaded documents through Streamlit.

## Architecture

The following diagram illustrates the high-level pipeline.

```text
                ┌────────────────────────┐
                │      Streamlit UI      │
                │  Upload PDF / Ask      │
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │   Request Coordination │
                │  Queue / Single-Flight │
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │    RAG Orchestrator    │
                └────────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼──────────┐       ┌──────────▼─────────┐
    │  Dense Retrieval   │       │  Sparse Retrieval  │
    │  Embeddings/FAISS  │       │       BM25         │
    └─────────┬──────────┘       └──────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                ┌────────────▼───────────┐
                │ Reciprocal Rank Fusion │
                │    and Reranking       │
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │  Select Context/Sources│
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │    LLM Generation      │
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │ Answer and Sources     │
                └────────────────────────┘
```

This is a conceptual overview. The actual execution path is defined by `app.py`, `pipeline/orchestrator.py`, and the supporting modules.

## How It Works

1. **Upload:** The user uploads a PDF through the Streamlit interface.
2. **Extract:** The PDF loader extracts document text.
3. **Chunk:** The text is divided into overlapping passages.
4. **Embed and index:** Sentence Transformers generate embeddings for semantic retrieval. FAISS indexes vectors, while BM25 supports lexical retrieval.
5. **Retrieve:** The user's question is processed by the retrieval pipeline. Dense and sparse retrievers return candidate passages.
6. **Fuse and rerank:** Reciprocal Rank Fusion combines the rankings, and the reranking stage refines candidate ordering.
7. **Generate:** Selected passages are supplied as context to the configured language model.
8. **Display:** The interface presents the response and available source information.

**Important limitation:** RAG does not guarantee factual correctness. PDF extraction quality, retrieval relevance, context selection, and language-model behavior can all affect the final answer.

## Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Language | Python |
| PDF processing | pypdf |
| Embeddings | Sentence Transformers |
| Dense retrieval | FAISS |
| Sparse retrieval | BM25 |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder |
| Answer generation | Groq / OpenAI APIs |
| Testing | pytest |

## Project Structure

```text
document-ai-chatbot/
├── app.py
├── app_state.py
├── requirements.txt
├── requirements.lock.txt
├── runtime.txt
├── core/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── vector_store.py
│   └── faiss_index.py
├── pipeline/
│   ├── orchestrator.py
│   └── cache.py
├── services/
│   ├── embeddings.py
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── query_guard.py
│   └── question_generator.py
├── data/
├── assets/
│   └── demo/
└── tests/
```

The `data/` directory holds project data and persistent artifacts where configured. Runtime-generated data may vary between local and deployed environments.

## Getting Started

### Prerequisites

- Python 3.11, subject to the project's runtime configuration
- Git
- A Groq API key or OpenAI API key, depending on the provider selected

### 1. Clone the repository

```bash
git clone https://github.com/jaymwangi/document-ai-chatbot.git
cd document-ai-chatbot
```

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API credentials

Create a `.env` file in the project root and add the credentials for your chosen provider.

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

Keep API keys private. Never commit real credentials or include them in source code.

### 5. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Testing

Run the existing test suite from the repository root:

```bash
python -m pytest tests/ -v
```

Test results and retrieval quality should be reported from actual test runs and controlled evaluations. No benchmark scores are claimed here.

## Engineering Considerations

The implementation brings together several practical software and information-retrieval patterns:

- Separation of interface, orchestration, retrieval, and generation responsibilities.
- Combination of semantic and lexical retrieval signals.
- Reuse of embedding models and cached computations.
- Persistence of vector-related artifacts where supported.
- Request coordination and API rate-limit handling.
- Visibility into retrieved sources, retrieval details, and pipeline timing.

These are engineering design choices, not a claim that the application is a fully production-hardened distributed service. Deployment persistence, concurrency behavior, retrieval accuracy, and answer faithfulness require separate testing.

## Roadmap

Potential next steps include:

- Create a labelled evaluation dataset containing questions and relevant source passages.
- Measure retrieval using Recall@k, Mean Reciprocal Rank (MRR), and normalized Discounted Cumulative Gain (nDCG).
- Compare dense retrieval, BM25, hybrid fusion, and reranking through repeatable experiments.
- Evaluate answer faithfulness and source attribution.
- Expand automated tests for end-to-end retrieval and generation.
- Test deployment persistence, concurrency, and multi-user isolation.
- Assess whether streaming, authentication, containerization, or a separate API service is justified by future requirements.

## Skills Demonstrated

**Machine Learning and NLP**
- Retrieval-Augmented Generation
- Sentence Transformer embeddings
- Vector search and document representation

**Information Retrieval**
- BM25 ranking
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Context retrieval and source grounding

**Software Engineering**
- Modular Python architecture
- Caching and persistence
- Request coordination
- Error handling and testing
- Pipeline observability

**Application Development**
- Streamlit
- External LLM API integration
- Environment configuration
- Application deployment

## Demo

- **Live application:** [jay-rag-chatbot.streamlit.app](https://jay-rag-chatbot.streamlit.app/)
- **Video walkthrough:** [Watch the demo](https://github.com/jaymwangi/document-ai-chatbot/blob/main/assets/demo/rag_doc_chatbot_demo.mp4)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Feedback and suggestions are welcome. Feel free to open an issue or contribute improvements.

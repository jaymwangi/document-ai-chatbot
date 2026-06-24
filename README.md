# 📄 RAG Document Chatbot

### Hybrid Retrieval (FAISS + BM25) • Queue-Based Processing • Modular RAG Architecture

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange.svg)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI-purple.svg)
![RAG](https://img.shields.io/badge/AI-RAG%20Pipeline-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

# 🚀 Live Demo

## 🌐 Deployed Application

👉 **Live App:**
https://jay-rag-chatbot.streamlit.app/

---

## 🎥 Video Demonstration

https://github.com/jaymwangi/document-ai-chatbot/blob/main/assets/demo/rag_doc_chatbot_demo.mp4

---

# 📌 Overview

The **RAG Document Chatbot** is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask natural language questions grounded in document content.

The system combines:

* Dense retrieval using FAISS
* Sparse retrieval using BM25
* Reciprocal Rank Fusion (RRF)
* LLM-powered answer generation
* Queue-based request management

This hybrid retrieval architecture improves both semantic understanding and keyword recall compared to traditional vector-only search systems.

---

# 🏛️ Architecture Highlights

This project was designed using production-inspired engineering principles:

* Hybrid retrieval (FAISS + BM25)
* Reciprocal Rank Fusion (RRF)
* Modular orchestration layer
* Queue-based request processing
* Single-flight execution model
* Persistent vector storage
* Embedding cache persistence
* Automatic index versioning and rebuilding
* Rate-limit handling and retry logic
* Retrieval observability and debugging
* Separation of UI and business logic

---

# 🔄 Project Evolution

Major improvements from the original implementation:

* Migrated from NumPy cosine similarity to FAISS indexing
* Added Hybrid Retrieval (FAISS + BM25)
* Implemented Reciprocal Rank Fusion (RRF)
* Introduced modular pipeline architecture
* Added singleton embedding model loading
* Implemented persistent embedding caching
* Added automatic index rebuild/version tracking
* Introduced retrieval debugging tools
* Added request timing and performance monitoring
* Implemented queue-based request management
* Added rate-limit recovery handling

---

# 🧠 Key Features

* 📄 PDF upload and parsing
* ✂️ Intelligent overlapping chunking
* 🧠 Sentence Transformer embeddings
* ⚡ FAISS vector indexing
* 🔍 Hybrid retrieval (FAISS + BM25)
* 🎯 Reciprocal Rank Fusion (RRF)
* 💾 Persistent embedding cache
* 🧠 Singleton embedding model
* 📋 FIFO request queue
* 🔒 Single-flight request execution
* ⏳ Automatic rate-limit handling
* 📚 Document-aware question generation
* ❓ Follow-up question suggestions
* ✨ Autocomplete support
* 🧾 Source attribution with retrieval scores
* 🧪 Retrieval debugging panel
* ⏱️ Pipeline performance tracing
* 🤖 Groq / OpenAI integration
* 💬 Streamlit chat interface

---

# 🏗️ System Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
Request Queue
 │
 ▼
Single-Flight Controller
 │
 ▼
RAG Orchestrator
 │
 ├──────── Query Embedding
 │
 ├──────── FAISS Retrieval
 │
 ├──────── BM25 Retrieval
 │
 └──────── RRF Fusion
          │
          ▼
      Top-K Context
          │
          ▼
      LLM Generator
          │
          ▼
      Final Answer
```

---

# ⚙️ How the Pipeline Works

## 1️⃣ Document Upload

PDF files are uploaded through the Streamlit interface.

## 2️⃣ Text Extraction

Documents are parsed into raw text.

## 3️⃣ Chunking

Text is split into overlapping chunks to preserve context.

## 4️⃣ Embedding Generation

Sentence Transformers convert chunks into vector embeddings.

## 5️⃣ Index Construction

* FAISS builds the dense vector index
* BM25 builds the sparse lexical index

## 6️⃣ Query Processing

The user query is embedded and prepared for retrieval.

## 7️⃣ Hybrid Retrieval

* FAISS retrieves semantic matches
* BM25 retrieves lexical matches
* RRF combines rankings into a final result set

## 8️⃣ Answer Generation

Retrieved context is passed to the LLM to generate a grounded answer.

---

# ⚡ Performance Optimizations

* Singleton embedding model prevents repeated loading
* Embedding cache avoids recomputation
* FAISS provides efficient semantic retrieval
* BM25 rebuilds only when documents change
* Persistent storage reduces startup overhead
* Queue-based execution prevents request collisions
* Rate-limit recovery prevents request loss

---

# 🔧 Engineering Challenges Solved

## Retrieval Quality

Implemented hybrid retrieval combining:

* Dense semantic search (FAISS)
* Sparse lexical search (BM25)
* Reciprocal Rank Fusion (RRF)

to improve ranking robustness and retrieval recall.

## Embedding Performance

Implemented:

* Singleton model loading
* Persistent embedding caching

to eliminate unnecessary recomputation.

## Request Management

Implemented:

* Queue-based processing
* Single-flight execution
* Rate-limit recovery

to ensure predictable application behavior under user load.

---

# 🧰 Tech Stack

| Component        | Technology                   |
| ---------------- | ---------------------------- |
| Frontend         | Streamlit                    |
| Backend          | Python                       |
| Embeddings       | Sentence Transformers        |
| Vector Search    | FAISS                        |
| Sparse Retrieval | BM25                         |
| Fusion           | Reciprocal Rank Fusion (RRF) |
| LLMs             | Groq / OpenAI                |
| PDF Parsing      | PyPDF                        |
| Architecture     | Modular RAG Pipeline         |
| Deployment       | Streamlit Cloud              |

---

# 📁 Project Structure

```text
rag-document-chatbot/
│
├── app.py
├── app_state.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── core/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── vector_store.py
│   └── faiss_index.py
│
├── pipeline/
│   ├── orchestrator.py
│   └── cache.py
│
├── services/
│   ├── embeddings.py
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── generator.py
│   ├── question_generator.py
│
├── data/
│   ├── stores/
│   └── cache_manifest.json
│
├── assets/
│   └── demo/
│
└── tests/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/jaymwangi/document-ai-chatbot.git
cd document-ai-chatbot
```

## Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 📈 Results

The current system provides:

* Hybrid retrieval using FAISS + BM25
* Source-grounded responses
* Persistent embedding caching
* Automatic index management
* Follow-up question generation
* Retrieval score inspection
* Queue-based request processing

Compared to vector-only retrieval, hybrid search improves robustness for both semantic and keyword-heavy queries.

---

# 🔮 Future Improvements

* 🧠 Cross-encoder reranking
* 📊 Retrieval evaluation benchmarks
* 🚀 GPU-accelerated FAISS
* 🔍 Query rewriting and expansion
* ⚡ Streaming token responses
* 🧵 Background worker architecture
* 🐳 Docker deployment
* 🔐 Authentication and user accounts
* 📡 FastAPI backend separation
* 👥 Multi-user session support

---

# 🎯 Skills Demonstrated

### Machine Learning & NLP

* Retrieval-Augmented Generation (RAG)
* Hybrid Retrieval Systems
* Sentence Transformer Embeddings
* Vector Search with FAISS

### Information Retrieval

* BM25 Ranking
* Reciprocal Rank Fusion (RRF)
* Context Grounding
* Retrieval System Design

### Software Engineering

* Modular Architecture Design
* Queue-Based Processing
* State Management
* Caching Strategies
* Error Handling & Retry Logic
* Performance Optimization

### Deployment

* Streamlit Cloud Deployment
* Environment Configuration
* Application Monitoring

---

# 📜 License

MIT License

---

# ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Build upon it
* 📢 Share feedback

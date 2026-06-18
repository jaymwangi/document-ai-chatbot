# 📄 RAG Document Chatbot (v2 - Hybrid RAG + FAISS Upgrade + Modular Pipeline)

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
[https://jay-rag-chatbot.streamlit.app/](https://jay-rag-chatbot.streamlit.app/)

---

## 🎥 Video Demonstration

[https://github.com/jaymwangi/document-ai-chatbot/blob/main/assets/demo/rag_doc_chatbot_demo.mp4](https://github.com/jaymwangi/document-ai-chatbot/blob/main/assets/demo/rag_doc_chatbot_demo.mp4)

---

# 📌 Project Evolution (IMPORTANT UPDATE)

> ⚠️ This project has evolved from a basic FAISS RAG chatbot into a **production-style Hybrid Retrieval System**

### 🔄 Major Upgrades

* Migrated from **NumPy cosine similarity → FAISS vector indexing**
* Added **Hybrid Retrieval (FAISS + BM25)**
* Implemented **Reciprocal Rank Fusion (RRF)** for improved ranking
* Introduced **modular RAG architecture (pipeline/orchestrator design)**
* Added **singleton embedding model (loaded once for performance)**
* Implemented **embedding cache with persistence**
* Added **automatic index versioning & rebuild logic**
* Introduced **retrieval debugging panel (scores + sources)**
* Added **query/response performance timing logs**
* Improved **UI feedback with real-time pipeline visibility**
* Enhanced **scalability for 1000+ document chunks**

---

# 📌 Overview

The **RAG Document Chatbot** is a full-stack AI system that allows users to upload PDF documents and ask natural language questions grounded strictly in their content.

It uses a **Retrieval-Augmented Generation (RAG)** pipeline combining:

* Dense retrieval (FAISS)
* Sparse retrieval (BM25)
* Ranking fusion (RRF)
* LLM-based response generation

This hybrid design significantly improves **accuracy, recall, and robustness** compared to pure vector search systems.

---

# 🧠 Key Features

* 📄 PDF upload and parsing
* ✂️ Intelligent overlapping chunking
* 🧠 Sentence-transformer embeddings
* ⚡ FAISS vector indexing (fast semantic search)
* 🔍 Hybrid retrieval (FAISS + BM25)
* 🎯 Reciprocal Rank Fusion (RRF)
* 💾 Persistent embedding cache
* 🧠 Singleton embedding model (loaded once)
* 📚 Document-aware question generation
* ❓ Follow-up question suggestions
* ✨ Autocomplete system
* 🧾 Source attribution with retrieval scores
* 🧪 Retrieval debug panel (inspect chunks + ranking)
* ⏱️ Full pipeline performance tracing
* 🤖 Groq / OpenAI LLM integration
* 💬 Streamlit chat interface

---

# 🏗️ System Architecture

```text
User Query
    ↓
Streamlit Frontend
    ↓
RAG Orchestrator
    ↓
Query Embedding
    ↓
Hybrid Retrieval Layer
   ┌──────────────┬──────────────┐
   │              │              │
 FAISS         BM25        Metadata Filter
   │              │
   └────── RRF (Fusion) ───────┘
              ↓
      Top-K Retrieved Chunks
              ↓
        LLM Generator
              ↓
        Final Answer
```

---

# ⚙️ How the Pipeline Works

## 1️⃣ Document Upload

PDF files are uploaded via Streamlit UI.

## 2️⃣ Text Extraction

Documents are parsed into raw text.

## 3️⃣ Chunking

Overlapping chunks preserve semantic continuity.

## 4️⃣ Embedding Generation

Sentence Transformers convert chunks into vectors.

## 5️⃣ Indexing

* FAISS builds dense vector index
* BM25 builds sparse lexical index

## 6️⃣ Query Processing

User query is embedded and tokenized.

## 7️⃣ Hybrid Retrieval

* FAISS retrieves semantic matches
* BM25 retrieves keyword matches
* RRF merges rankings into final results

## 8️⃣ LLM Generation

Retrieved context is passed to Groq/OpenAI for grounded answers.

---

# ⚙️ Performance Optimizations

* Singleton embedding model prevents repeated loading
* Embedding cache avoids recomputation
* FAISS search typically < 50ms
* BM25 rebuilds only when documents change
* Hybrid retrieval improves recall vs vector-only systems
* Small ingestion pipeline: ~2–5 seconds per document batch

---

# 🧰 Tech Stack

| Component     | Technology                   |
| ------------- | ---------------------------- |
| Frontend      | Streamlit                    |
| Backend       | Python                       |
| Embeddings    | Sentence Transformers        |
| Vector DB     | FAISS                        |
| Sparse Search | BM25                         |
| Fusion        | Reciprocal Rank Fusion (RRF) |
| LLMs          | Groq / OpenAI                |
| PDF Parsing   | PyPDF                        |
| Architecture  | Modular RAG Pipeline         |
| Deployment    | Streamlit Cloud              |

---

# 📁 Project Structure

```text
rag-document-chatbot/
│
├── app.py
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
│   ├── reranker.py
│   ├── query_guard.py
│   └── question_generator.py
│
├── data/
│   ├── stores/
│   └── cache_manifest.json
│
├── assets/
│   ├── demo/
│   └── screenshots/
│
└── tests/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/jaymwangi/document-ai-chatbot.git
cd document-ai-chatbot
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate
```

## 3️⃣ Install Dependencies

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

```
http://localhost:8501
```

---

# ☁️ Deployment

1. Push to GitHub
2. Connect Streamlit Cloud
3. Add secrets
4. Deploy

---

# 📊 Performance Notes

* FAISS enables sub-50ms semantic search
* BM25 improves keyword recall
* RRF fusion improves ranking robustness
* Modular pipeline enables production scaling
* Retrieval debugging improves interpretability

---

# 🔮 Future Improvements

* 🧠 Cross-encoder reranking
* 📈 Retrieval evaluation benchmark suite
* 🚀 GPU-accelerated FAISS
* 🔍 Query rewriting & expansion
* 🧵 Conversational memory layer
* 🐳 Docker production deployment
* 🔐 Authentication system
* 📡 FastAPI backend migration

---

# 🎯 Skills Demonstrated

* Hybrid Retrieval Systems (Dense + Sparse + Fusion)
* FAISS vector database engineering
* Information Retrieval (IR) design
* RAG pipeline architecture
* Embedding caching strategies
* Production-grade Python modular design
* Performance profiling & optimization
* LLM orchestration (Groq / OpenAI)
* Streamlit deployment engineering

---

# 📜 License

MIT License

---

# ⭐ Support

If you found this useful:

* ⭐ Star the repo
* 🍴 Fork it
* 🚀 Build on it
* 📢 Share feedback

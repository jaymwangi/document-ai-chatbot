# 📄 RAG Document Chatbot

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LLM](https://img.shields.io/badge/LLM-Groq%2FOpenAI-purple.svg)
![Vector Search](https://img.shields.io/badge/VectorSearch-Numpy%20Cosine-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🚀 Overview

The **RAG Document Chatbot** is an end-to-end AI system that allows users to upload PDF documents and ask natural language questions about their content.

It implements a **Retrieval-Augmented Generation (RAG)** pipeline that ensures all responses are grounded in the provided documents, combining semantic search with large language models.

Unlike simple chatbots, this system **retrieves relevant knowledge before generating answers**, making responses more accurate, explainable, and context-aware.

---

## 🧠 Key Features

* 📄 Upload and process PDF documents
* ✂️ Intelligent text chunking with overlap
* 🧠 Sentence-transformer embeddings
* 🔍 Semantic search using cosine similarity
* 🗂️ Custom in-memory vector store
* 🤖 LLM-powered answer generation (Groq / OpenAI / Mock)
* 📚 Source-aware responses
* 💬 Interactive Streamlit chat interface
* 🧩 Fully modular and extensible architecture

---

## 🏗️ System Architecture

```text
User Query
    ↓
Streamlit UI (app.py)
    ↓
RAG Pipeline (rag_pipeline.py)
    ↓
Retriever (Task 6)
    ↓
Vector Store (NumPy Cosine Similarity)
    ↓
Top-K Relevant Chunks
    ↓
LLM Generator (Groq / OpenAI / Mock)
    ↓
Final Answer
    ↓
User Response
```

---

## ⚙️ How It Works

### 1. Document Upload

Users upload PDF files through the Streamlit interface.

### 2. Text Extraction

PDFs are parsed into raw text using a lightweight loader.

### 3. Chunking

Text is split into overlapping semantic chunks for better retrieval accuracy.

### 4. Embedding Generation

Each chunk is converted into dense vector representations using Sentence Transformers.

### 5. Vector Storage

Embeddings are stored in a custom in-memory vector store.

### 6. Query Processing

User questions are embedded into the same vector space.

### 7. Retrieval

Top-K most relevant chunks are retrieved using cosine similarity search.

### 8. Answer Generation

A Large Language Model generates a final answer using retrieved context.

---

## 🧰 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Embeddings:** Sentence Transformers (MiniLM / MPNet)
* **Vector Store:** Custom NumPy-based cosine similarity engine
* **LLM Providers:** Groq / OpenAI / Mock mode (testing)
* **PDF Processing:** PyPDF

---

## 📁 Project Structure

```text
rag-document-chatbot/
│
├── app.py                  # Streamlit UI (Task 8)
├── rag_pipeline.py         # Full system orchestration
├── config.py
│
├── core/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunker.py          # Text splitting logic
│   ├── vector_store.py     # In-memory vector database
│
├── services/
│   ├── embeddings.py       # Embedding models
│   ├── retriever.py        # Semantic search (Task 6)
│   ├── generator.py        # LLM interface (Task 7)
│
├── tests/                  # Unit tests for all modules
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/rag-document-chatbot.git
cd rag-document-chatbot
```

---

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

* Groq is recommended for fast and free inference
* OpenAI is optional alternative

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## ☁️ Deployment

You can deploy this project using:

* Streamlit Cloud (recommended)
* Render
* AWS EC2
* HuggingFace Spaces

For Streamlit Cloud:

1. Push project to GitHub
2. Connect repository
3. Add `GROQ_API_KEY` in secrets
4. Deploy

---

## 📸 Screenshots

### 📄 Document Upload Interface

(Add screenshot here)

```
assets/screenshots/upload.png
```

### 💬 Question & Answer Interface

(Add screenshot here)

```
assets/screenshots/chat.png
```

### 📚 Retrieved Context

(Add screenshot here)

```
assets/screenshots/context.png
```

---

## 📊 Performance Notes

* Uses in-memory vector storage (fast for small-medium datasets)
* Suitable for up to ~10,000 chunks
* For larger scale, FAISS or ChromaDB upgrade recommended

---

## 🔮 Future Improvements

* ⚡ FAISS-based optimized vector search
* 🧠 Reranking with cross-encoders
* 💾 Persistent vector database (ChromaDB)
* 💬 Chat memory per document
* 🐳 Docker containerization
* 🔐 User authentication system
* 📡 FastAPI backend for production scaling

---

## 🎯 Skills Demonstrated

* Retrieval-Augmented Generation (RAG) systems
* Embedding-based semantic search
* Vector similarity search (cosine similarity)
* LLM integration (Groq / OpenAI)
* Modular Python system design
* Full-stack AI application development
* Streamlit UI development

---

## 📌 Why This Project Matters

This project demonstrates how modern AI applications are built beyond simple prompting.

It shows how to design a complete system that:

* understands documents
* retrieves relevant knowledge
* and generates grounded, accurate answers

This architecture is widely used in:

* AI copilots
* enterprise search systems
* legal/document analysis tools
* knowledge assistants

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Improve it further



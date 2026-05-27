# 📄 RAG Document Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)
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

The **RAG Document Chatbot** is a full-stack AI application that enables users to upload PDF documents and ask natural language questions grounded in the uploaded content.

The system implements a **Retrieval-Augmented Generation (RAG)** pipeline that combines:

- semantic retrieval
- vector similarity search
- contextual document grounding
- LLM-powered response generation

Unlike traditional chatbots that rely purely on prompting, this system retrieves relevant document context before generating responses, significantly improving factual accuracy and explainability.

---

# 🧠 Key Features

- 📄 PDF upload and parsing
- ✂️ Intelligent overlapping text chunking
- 🧠 Sentence-transformer embeddings
- 🔍 Semantic retrieval using cosine similarity
- 🗂️ Custom vector store architecture
- 🤖 Groq/OpenAI LLM integration
- 📚 Source-grounded responses
- 💬 Interactive Streamlit chat interface
- ⚡ Real-time document ingestion
- 🧩 Modular production-style architecture

---

# 📸 Application Screenshots

---

## 🏠 Homepage / Empty State

Demonstrates:
- clean UI
- sidebar controls
- upload workflow
- configurable retrieval settings

![Homepage](assets/screenshots/Doc_rag_chatbot_Homepage.PNG)

---

## 📄 Document Upload & Processing

Demonstrates:
- PDF ingestion
- real-time processing
- document management workflow

![Document Upload](assets/screenshots/Doc_rag_chatbot_Document.PNG)

---

## 💬 Question & Answer Interaction

Demonstrates:
- grounded response generation
- contextual answering
- semantic retrieval in action

![QnA](assets/screenshots/Doc_rag_chatbotQnA.PNG)

---

## 📚 Retrieval & Context Display

Demonstrates:
- retrieved chunks
- source-aware answering
- explainable RAG pipeline behavior

![Retrieval](assets/screenshots/Doc_rag_chatbot_Retrival.PNG)

---

# 🏗️ System Architecture

```text
User Query
    ↓
Streamlit Frontend (app.py)
    ↓
RAG Pipeline Orchestrator
    ↓
Semantic Retriever
    ↓
Vector Similarity Search
    ↓
Top-K Relevant Chunks
    ↓
LLM Generator (Groq/OpenAI)
    ↓
Grounded Final Response
```

---

# ⚙️ How the RAG Pipeline Works

## 1️⃣ Document Upload

Users upload PDF documents through the Streamlit interface.

---

## 2️⃣ PDF Text Extraction

The system extracts raw text from PDFs using PyPDF.

---

## 3️⃣ Intelligent Chunking

Documents are split into overlapping semantic chunks to preserve contextual continuity during retrieval.

---

## 4️⃣ Embedding Generation

Chunks are transformed into dense vector embeddings using Sentence Transformers.

---

## 5️⃣ Vector Storage

Embeddings are stored inside a custom vector similarity engine.

---

## 6️⃣ Query Embedding

User questions are embedded into the same semantic vector space.

---

## 7️⃣ Semantic Retrieval

Top-K relevant chunks are retrieved using cosine similarity search.

---

## 8️⃣ LLM Answer Generation

Retrieved context is injected into the prompt sent to the LLM to generate grounded answers.

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Embeddings | Sentence Transformers |
| Vector Search | NumPy Cosine Similarity |
| LLM Providers | Groq / OpenAI |
| PDF Parsing | PyPDF |
| Deployment | Streamlit Cloud |

---

# 📁 Project Structure

```text
rag-document-chatbot/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── core/
│   ├── pdf_loader.py
│   ├── chunker.py
│   └── vector_store.py
│
├── services/
│   ├── embeddings.py
│   ├── retriever.py
│   └── generator.py
│
├── assets/
│   ├── demo/
│   │   └── rag_doc_chatbot_demo.mp4
│   │
│   └── screenshots/
│       ├── Doc_rag_chatbot_Homepage.PNG
│       ├── Doc_rag_chatbot_Document.PNG
│       ├── Doc_rag_chatbotQnA.PNG
│       └── Doc_rag_chatbot_Retrival.PNG
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

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Groq is recommended for fast inference and lower latency.

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# ☁️ Deployment

This project is deployed on Streamlit Cloud.

## Deployment Steps

1. Push project to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets/environment variables
4. Deploy application

---

# 📊 Performance Notes

- Uses lightweight in-memory vector retrieval
- Fast semantic search for small-to-medium datasets
- Optimized for educational and portfolio-scale RAG systems
- Modular architecture enables future FAISS/Chroma upgrades

---

# 🔮 Future Improvements

- ⚡ FAISS vector indexing
- 🧠 Cross-encoder reranking
- 💾 Persistent ChromaDB storage
- 🧵 Conversational memory
- 🐳 Docker containerization
- 🔐 Authentication system
- 📡 FastAPI backend
- ☁️ Production cloud deployment architecture

---

# 🎯 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic search systems
- Embedding pipelines
- Vector similarity search
- LLM orchestration
- AI system architecture
- Full-stack AI application development
- Streamlit deployment workflows
- Environment/debugging management
- Production dependency resolution

---

# 📌 Why This Project Matters

Modern AI systems increasingly require retrieval-based architectures rather than standalone prompting.

This project demonstrates:

- grounded response generation
- semantic understanding
- explainable AI workflows
- document-aware conversational systems

These concepts power:
- enterprise copilots
- legal AI systems
- internal knowledge assistants
- research copilots
- AI search platforms

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🚀 Build upon it
- 📢 Share feedback
# 📚 RAG PDF Chatbot

A Retrieval Augmented Generation (RAG) application that allows users to ask questions about PDF documents using semantic search and vector embeddings.

## 🚀 Features

* PDF document ingestion
* Automatic text chunking
* Semantic search using vector embeddings
* FAISS vector database
* Interactive Streamlit interface
* Source document retrieval
* Context aware question answering

## 🛠️ Tech Stack

* Python
* LangChain
* FAISS
* Hugging Face Embeddings
* Streamlit
* PyPDF

## 📂 Project Structure

simple_rag_project/

├── streamlit_app.py

├── data/

├── vectorstore/

├── utils/

│   ├── loader.py

│   ├── splitter.py

│   └── embeddings.py

└── README.md

## 🔄 Workflow

PDF Document

↓

Text Extraction

↓

Chunking

↓

Embedding Generation

↓

FAISS Vector Store

↓

Semantic Retrieval

↓

Answer Generation

## ▶️ Installation

```bash
git clone <repository-url>
cd simple_rag_project

pip install -r requirements.txt
```

## ▶️ Run Application

```bash
streamlit run streamlit_app.py
```

## 💡 Sample Questions

* What is solar energy?
* Explain hydropower.
* What are the advantages of wind energy?
* What is renewable energy?

## 🎯 Future Improvements

* LLM-based answer generation
* Conversational memory
* Multi PDF support
* Hybrid search
* Multimodal RAG with image understanding







# Basic RAG App

A simple local Retrieval-Augmented Generation (RAG) application built using FastAPI, Sentence Transformers, and Ollama.

This project allows you to:
- Load PDFs
- Chunk and embed document text
- Perform semantic search
- Ask questions against your documents using a local LLM (Mistral via Ollama)

---

# Features

- PDF ingestion
- Semantic vector search
- Local embeddings using Sentence Transformers
- Local LLM inference using Ollama
- FastAPI REST APIs
- Simple RAG pipeline implementation
- Fully local execution (no cloud APIs required)

---

# Tech Stack

- Python
- FastAPI
- Sentence Transformers
- Ollama
- Mistral
- NumPy
- PyPDF

---

# Prerequisites

## 1. Python

Install Python 3.10+.

Verify:

```bash
python3 --version
```

---

## 2. Ollama

Install Ollama:

https://ollama.com

Verify installation:

```bash
ollama --version
```

---

## 3. Pull Mistral Model

Run:

```bash
ollama run mistral
```

This downloads the model locally.

---

# Installation

Clone repository:

```bash
git clone <YOUR_REPO_URL>
cd Basic-RAG-App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Project Structure

```text
Basic-RAG-App/
│
├── main.py
├── requirements.txt
├── README.md
├── data.json
└── pdfs/
```

---

# Running The Application

## Step 1 — Start Ollama

In Terminal 1:

```bash
ollama run mistral
```

Keep it running.

---

## Step 2 — Start FastAPI Server

In Terminal 2:

```bash
uvicorn main:app --reload
```

---

## Step 3 — Open Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

---

# Using The Application

## 1. Add PDFs

Place PDF files inside:

```text
pdfs/
```

---

## 2. Load PDFs

Call:

```text
POST /load-pdf
```

This:
- extracts text
- chunks content
- generates embeddings
- indexes documents

---

## 3. Search Documents

Endpoint:

```text
GET /search
```

Example query:

```text
AWS services
```

---

## 4. Ask Questions

Endpoint:

```text
GET /ask
```

Example:

```text
What technologies has the candidate worked with?
```

The application:
1. Retrieves relevant document chunks
2. Sends context to Mistral
3. Generates an answer

---

# Example Architecture

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Search
 ↓
Context Retrieval
 ↓
Mistral (Ollama)
 ↓
Generated Answer
```

---

# Notes

- This is a basic RAG implementation intended for learning and experimentation.
- Retrieval quality depends heavily on chunking strategy and PDF text quality.
- LLM responses may still hallucinate or miscalculate factual information.
- Better chunking, hybrid search, and structured extraction can improve results significantly.

---

# Future Improvements

- Better chunking strategies
- Overlapping chunks
- FAISS / Elasticsearch vector search
- Streaming responses
- Chat UI
- Metadata filtering
- Hybrid search
- Citation support
- Structured extraction pipelines

---

# License

MIT
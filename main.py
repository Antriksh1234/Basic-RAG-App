from fastapi import FastAPI
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import numpy as np
import json
import os

app = FastAPI()

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_FILE = "data.json"

documents_db = []
embeddings_db = []

# Request model
class DocumentRequest(BaseModel):
    text: str

# Similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

def ask_mistral(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return data["response"]

# Split text into chunks
def chunk_text(text):

    raw_chunks = text.split("\n\n")

    chunks = []

    for chunk in raw_chunks:

        chunk = chunk.strip()

        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks

# Save data
def save_data():

    with open(DATA_FILE, "w") as file:
        json.dump(documents_db, file)

# Load data
# def load_data():

#     global documents_db
#     global embeddings_db

#     if not os.path.exists(DATA_FILE):
#         return

#     with open(DATA_FILE, "r") as file:
#         documents_db = json.load(file)

#     embeddings_db = list(model.encode(documents_db))

# # Load existing data
# load_data()

@app.get("/")
def home():
    return {"message": "PDF Semantic Search Running"}

# Add manual text
@app.post("/add")
def add_document(request: DocumentRequest):

    text = request.text

    documents_db.append(text)

    embedding = model.encode(text)

    embeddings_db.append(embedding)

    save_data()

    return {
        "message": "Document added successfully"
    }

# Load PDF
@app.post("/load-pdf")
def load_pdf():

    pdf_folder = "pdfs"

    added_chunks = 0

    for filename in os.listdir(pdf_folder):

        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(pdf_folder, filename)

        reader = PdfReader(path)

        full_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                full_text += text

        # Split into chunks
        chunks = chunk_text(full_text)

        for chunk in chunks:

            documents_db.append(chunk)

            embedding = model.encode(chunk)

            embeddings_db.append(embedding)

            added_chunks += 1

    save_data()

    return {
        "message": "PDFs loaded successfully",
        "chunks_added": added_chunks
    }

# Semantic search
@app.get("/search")
def search(query: str):

    if not documents_db:
        return {"message": "Database is empty"}

    query_embedding = model.encode(query)

    results = []

    for text, embedding in zip(documents_db, embeddings_db):

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append({
            "text": text,
            "score": float(score)
        })

    # Sort by score
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]

@app.get("/ask")
def ask(query: str):

    if not documents_db:
        return {"message": "Database is empty"}

    query_embedding = model.encode(query)

    results = []

    # Find relevant chunks
    for text, embedding in zip(documents_db, embeddings_db):

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        if score > 0.05:

            results.append({
                "text": text,
                "score": float(score)
            })

    # Sort best chunks
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Take top 3 chunks
    top_chunks = results[:3]

    context = "\n\n".join(
        [chunk["text"] for chunk in top_chunks]
    )

    # Build prompt
    prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
    """

    # Ask mistral
    answer = ask_mistral(prompt)

    return {
        "question": query,
        "answer": answer,
        "context_used": top_chunks
    }
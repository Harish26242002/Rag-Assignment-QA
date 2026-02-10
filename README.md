
# RAG Document Q&A System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that answers
questions from a collection of PDF documents. It retrieves relevant document chunks
from a vector database and generates grounded answers using an LLM.

## Features
- PDF ingestion and chunking
- HuggingFace embeddings (MiniLM)
- Vector database: Chroma (local)
- Optional Pinecone support (cloud)
- Streamlit UI
- Source citations
- Metrics (retrieval time, generation time)

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Ingest documents:
   python ingest.py

3. Run app:
   streamlit run app.py

## Tech Stack
- LangChain
- HuggingFace Transformers
- ChromaDB
- Streamlit

## Notes
- Uses ChromaDB for local vector storage
- Pinecone support available for scalable deployment
- Answers are grounded with source references

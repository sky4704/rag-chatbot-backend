# RAG Chatbot with Gemini and Langchain

This is a simple RAG (Retrieval-Augmented Generation) application built with Python, Langchain, Google Gemini API, and ChromaDB.

## Features
- PDF ingestion and chunking.
- Vector storage with ChromaDB.
- Semantic search using Gemini embeddings.
- Question answering using Gemini 1.5 Flash.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   - Copy `.env.example` to `.env`.
   - Add your Google Gemini API key to the `.env` file.

3. **Ingest Data:**
   - Place your PDF files in the `data/` folder.
   - Run the ingestion script:
     ```bash
     python ingest.py
     ```

4. **Run the Chatbot:**
   ```bash
   python app.py
   ```

## Project Structure
- `app.py`: Main chat interface.
- `ingest.py`: Script to process PDFs.
- `utils/llm_config.py`: LLM and Embedding configuration.
- `data/`: Place your PDFs here.
- `db/`: Persistent vector store directory.

# RAG Chatbot with Gemini and Langchain

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on a provided PDF document.

## Tech Stack
- **Language:** Python 3.12+
- **Framework:** Langchain
- **LLM & Embeddings:** Google Gemini API
- **Vector Database:** ChromaDB

## Project Structure
- `app.py`: Main entry point for the chatbot.
- `ingest.py`: Script to process PDF and populate the vector database.
- `utils/`: Helper functions for PDF processing and Langchain setup.
- `data/`: Directory for input PDF documents.
- `db/`: Persistent storage for ChromaDB.

## Development Workflow
1. **Environment Setup:** Use a virtual environment and install dependencies from `requirements.txt`.
2. **Configuration:** Store the Gemini API key and Groq API key in a `.env` file (`GOOGLE_API_KEY`, `GROQ_API_KEY`).
    - **Admin Recovery:** Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` in your environment to automatically create or reset your admin account on startup.
    - **Frontend Connection:** In production (e.g., Render), set `PUBLIC_API_URL` on the frontend service to point to your backend URL.
3. **Ingestion:** Run `ingest.py` to process a PDF.
4. **Chat:** Run `app.py` or start the API with `main.py` to start the chatbot session. The system uses Gemini by default and falls back to Groq (Llama 3) automatically if Gemini's quota is exceeded or rate limits are hit.

## Management & Professionalism
- **Automatic Fallback:** If Gemini's free tier quota (429 error) is reached, the system instantly switches to Groq to ensure uninterrupted service.
- **Scanned PDFs (OCR):** The system automatically detects scanned PDFs and uses Tesseract OCR to extract text for analysis.
- **Admin Management:** Admins are stored in the database. Use the environment variables `ADMIN_USERNAME` and `ADMIN_PASSWORD` for easy setup and password recovery on platforms like Render.
- **Security:** Industry-standard JWT and Bcrypt hashing are used for all admin operations.
- **Storage:** PDFs are stored in Supabase Storage, and vector embeddings are stored in Pinecone (forced to 768 dimensions for compatibility).

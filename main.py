import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from utils.db import get_db, init_db, Book, User
from utils.llm_config import get_embeddings, get_llm, get_vector_store
from utils.storage import upload_file, delete_file
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from fastapi.staticfiles import StaticFiles
from utils.auth import create_access_token, verify_password, get_current_admin, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta

load_dotenv()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on start
    init_db()
    yield

app = FastAPI(title="Multi-Book RAG API", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
COVERS_DIR = os.path.join(STATIC_DIR, "covers")

for path in [DATA_DIR, DB_DIR, COVERS_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/books", response_model=List[dict])
def list_books(db: Session = Depends(get_db)):
    books = db.query(Book).order_by(Book.created_at.desc()).all()
    
    def get_full_cover_url(url):
        if not url:
            return None
        # If it's already a full URL (from Supabase/Cloud), return it.
        if url.startswith("http"):
            return url
        # Otherwise, assume it's a legacy filename and prepend static path.
        return f"{os.getenv('BASE_URL', 'http://localhost:8000')}/static/covers/{url}"

    return [{
        "id": b.id, 
        "title": b.title, 
        "collection_id": b.collection_id, 
        "cover_url": get_full_cover_url(b.cover_url)
    } for b in books]


@app.post("/upload")
async def upload_book(
    title: str = File(...),
    file: UploadFile = File(...), 
    cover: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save PDF file
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(DATA_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    pdf_url = upload_file(file_path, filename)

    # Save Cover file if provided
    cover_url = None
    if cover:
        cover_ext = cover.filename.split(".")[-1]
        cover_filename = f"{file_id}.{cover_ext}"
        cover_path = os.path.join(COVERS_DIR, cover_filename)
        with open(cover_path, "wb") as f:
            f.write(await cover.read())
        cover_url = upload_file(cover_path, f"covers/{cover_filename}")

    # Ingest PDF into a specific collection
    collection_id = f"col_{file_id.replace('-', '_')}"
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import pytesseract
        from pdf2image import convert_from_path
        from langchain_core.documents import Document
        
        print(f"Loading '{file.filename}' for processing...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Check if the PDF is scanned (no text)
        full_text_check = "".join([doc.page_content for doc in documents])
        if len(full_text_check.strip()) < 100:
            print("Scanned PDF detected. Running local OCR...")
            images = convert_from_path(file_path)
            ocr_documents = []
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                ocr_documents.append(Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": i}
                ))
            documents = ocr_documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = get_vector_store(collection_id)
        
        # Batch ingestion with robust retry logic for free tier rate limits
        batch_size = 10  # Smaller batches for free tier
        from google.api_core import exceptions as google_exceptions
        
        print(f"Ingesting {len(chunks)} fragments into neural archives...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            success = False
            retries = 5
            wait_time = 5  # Start with 5 seconds
            
            while not success and retries > 0:
                try:
                    vectorstore.add_documents(batch)
                    success = True
                    print(f"  - Fragment {i//batch_size + 1}/{len(chunks)//batch_size + 1} secured.")
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        print(f"  ! Neural gateway busy, cooling down for {wait_time}s... (Retries left: {retries})")
                        time.sleep(wait_time)
                        wait_time *= 2  # Exponential backoff
                        retries -= 1
                    else:
                        raise e
            
            if not success:
                raise Exception("The neural gateway is heavily congested. Please try again in a few minutes.")
                
            if i + batch_size < len(chunks):
                time.sleep(3) # Base delay between successful fragments

        # Save to metadata DB
        new_book = Book(
            title=title,
            filename=filename,
            collection_id=collection_id,
            cover_url=cover_url
        )
        db.add(new_book)
        db.commit()
        db.refresh(new_book)
        
        return {"id": new_book.id, "title": new_book.title, "collection_id": collection_id}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        if cover_filename and os.path.exists(os.path.join(COVERS_DIR, cover_filename)):
            os.remove(os.path.join(COVERS_DIR, cover_filename))
        
        # Professional Error Parsing
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            friendly_msg = "The system's Access Key is invalid or expired. Please contact the administrator to update the neural gateway credentials."
        elif "dimension" in error_msg.lower() or "400" in error_msg:
            friendly_msg = f"Neural Archive Mismatch: The vector dimension or configuration is incorrect. Details: {error_msg[:100]}"
        elif "429" in error_msg:
            friendly_msg = "The archives are currently receiving too many inquiries. Please wait a few moments before trying again."
        elif "tesseract" in error_msg.lower() or "poppler" in error_msg.lower():
            friendly_msg = "The Archive's scanning engine (Tesseract/Poppler) is not correctly configured on the host system."
        else:
            friendly_msg = f"Archive Ingestion Failed: {error_msg[:150]}"

        raise HTTPException(status_code=500, detail=friendly_msg)

@app.patch("/books/{book_id}")
async def update_book(
    book_id: int,
    title: Optional[str] = Form(None),
    cover: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin)
):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    if title:
        book.title = title
    
    if cover:
        # Delete old cover
        if book.cover_url:
            old_path = os.path.join(COVERS_DIR, book.cover_url)
            if os.path.exists(old_path):
                os.remove(old_path)
        
        # Save new cover
        cover_ext = cover.filename.split(".")[-1]
        file_id = book.collection_id.replace("col_", "").replace("_", "-")
        cover_filename = f"{file_id}.{cover_ext}"
        cover_path = os.path.join(COVERS_DIR, cover_filename)
        with open(cover_path, "wb") as f:
            f.write(await cover.read())
        book.cover_url = cover_filename
        
    db.commit()
    db.refresh(book)
    
    # Helper to construct full URL
    def get_full_cover_url(url):
        if not url:
            return None
        if url.startswith("http"):
            return url
        return f"{os.getenv('BASE_URL', 'http://localhost:8000')}/static/covers/{url}"

    return {
        "id": book.id, 
        "title": book.title, 
        "cover_url": get_full_cover_url(book.cover_url)
    }

@app.delete("/books/{book_id}")
async def delete_book(book_id: int, db: Session = Depends(get_db), admin: str = Depends(get_current_admin)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    try:
        # Delete from Vector Store
        vectorstore = get_vector_store(book.collection_id)
        if hasattr(vectorstore, "delete_collection"):
            vectorstore.delete_collection()
        else:
            # For Pinecone or stores without delete_collection
            vectorstore.delete(delete_all=True)
    except Exception as e:
        print(f"Error deleting collection: {e}")
    
    # Delete files
    delete_file(book.filename, DATA_DIR)
    if book.cover_url:
        delete_file(book.cover_url, COVERS_DIR)
            
    # Delete from DB
    db.delete(book)
    db.commit()
    return {"message": "Book deleted successfully"}

@app.post("/chat/{collection_id}")
async def chat(collection_id: str, payload: dict):
    user_input = payload.get("message")
    if not user_input:
        raise HTTPException(status_code=400, detail="Message is required")

    vectorstore = get_vector_store(collection_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = get_llm()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({"input": user_input})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

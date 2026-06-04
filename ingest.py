import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.llm_config import get_embeddings, get_vector_store
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(file_path: str, collection_name: str = "default_collection"):
    """
    Loads a PDF, splits it into chunks, and stores them in the vector store.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Creating vector store for collection '{collection_name}'...")
    vectorstore = get_vector_store(collection_name)
    
    import time
    from langchain_google_genai._common import GoogleGenerativeAIError

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Ingesting batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                vectorstore.add_documents(batch)
                success = True
            except GoogleGenerativeAIError as e:
                if "429" in str(e):
                    print("Rate limit hit, waiting 30 seconds...")
                    time.sleep(30)
                    retries -= 1
                else:
                    raise e
        
        if not success:
            print(f"Failed to ingest batch starting at index {i} after retries.")
            break

        if i + batch_size < len(chunks):
            time.sleep(5)  # Wait 5 seconds between batches
            
    print("Ingestion complete!")

if __name__ == "__main__":
    # Example usage: process any pdf in the data folder
    DATA_DIR = "data"
    DB_DIR = "db"
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Please add a PDF file and run again.")
    else:
        for pdf in pdf_files:
            ingest_pdf(os.path.join(DATA_DIR, pdf), "main_collection")

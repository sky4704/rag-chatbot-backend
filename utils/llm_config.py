import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

def get_embeddings():
    """Initializes and returns Google Gemini embeddings."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    """Initializes and returns the Gemini Pro chat model with Groq fallback."""
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Fallback to Groq if Gemini hits rate limits and API key is provided
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq = ChatGroq(model="llama3-70b-8192", temperature=0.3)
        return gemini.with_fallbacks([groq])
    
    return gemini

def get_vector_store(collection_name: str):
    """
    Returns a vector store instance. 
    Uses Pinecone if PINECONE_API_KEY is present, otherwise defaults to local ChromaDB.
    """
    embeddings = get_embeddings()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if pinecone_api_key:
        from langchain_pinecone import PineconeVectorStore
        # Note: In Pinecone, we use the collection_name as a namespace or index name
        # For simplicity in free tier (1 index limit), we'll use one index and namespaces
        index_name = os.getenv("PINECONE_INDEX", "rag-index")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=collection_name,
            pinecone_api_key=pinecone_api_key
        )
    else:
        from langchain_chroma import Chroma
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_dir = os.path.join(base_dir, "db")
        return Chroma(
            collection_name=collection_name,
            persist_directory=db_dir,
            embedding_function=embeddings
        )

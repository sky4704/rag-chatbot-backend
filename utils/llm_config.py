import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import google.generativeai as genai
from typing import List, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda

load_dotenv()

class CustomGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return [
            genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=768
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768
        )["embedding"]

def get_embeddings():
    """Initializes and returns Google Gemini embeddings with forced 768 dimensions."""
    return CustomGoogleEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    """Initializes and returns the Gemini Pro chat model with Groq fallback and signatures."""
    # 1. Primary Model: Gemini
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        max_retries=1
    )
    
    # 2. Fallback Model: Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Signatures
    def add_gemini_signature(res):
        if hasattr(res, 'content'):
            res.content += "\n\n✨ *Answered by Neural Gateway (Gemini)*"
        return res

    def add_groq_signature(res):
        if hasattr(res, 'content'):
            res.content += "\n\n🚀 *Answered by Deep Memory (Groq/Llama)*"
        return res

    # Create tagged chains
    gemini_tagged = gemini | RunnableLambda(add_gemini_signature)
    
    if groq_api_key:
        print("LOG: Groq fallback with signature enabled.")
        groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        groq_tagged = groq | RunnableLambda(add_groq_signature)
        
        return gemini_tagged.with_fallbacks(
            fallbacks=[groq_tagged],
            exceptions_to_handle=(Exception,)
        )
    
    return gemini_tagged

def get_vector_store(collection_name: str):
    """
    Returns a vector store instance. 
    Uses Pinecone if PINECONE_API_KEY is present, otherwise defaults to local ChromaDB.
    """
    embeddings = get_embeddings()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if pinecone_api_key:
        from langchain_pinecone import PineconeVectorStore
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

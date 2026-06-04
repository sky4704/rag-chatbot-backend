import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from utils.llm_config import get_embeddings, get_llm

load_dotenv()

def setup_rag_chain(db_dir: str):
    """
    Sets up the RAG chain using ChromaDB and Gemini.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def chat_loop():
    DB_DIR = "db"
    if not os.path.exists(DB_DIR):
        print("Vector database not found. Please run ingest.py first.")
        return

    print("Initializing RAG system...")
    rag_chain = setup_rag_chain(DB_DIR)
    
    print("\nRAG Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        print("Thinking...")
        try:
            response = rag_chain.invoke({"input": user_input})
            print(f"AI: {response['answer']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()

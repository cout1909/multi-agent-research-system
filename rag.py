"""
rag.py
RAG (Retrieval Augmented Generation) layer for our research system.
Allows users to upload their own PDF documents, which get stored in
ChromaDB (a free vector database). Agents can then search BOTH
the live web AND the user's uploaded documents.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Setup ChromaDB with a free HuggingFace embedding model
# No API key needed - runs completely locally on your PC
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # small, fast, free embedding model
)

vectorstore = Chroma(
    collection_name="research_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # saves to a folder on your PC
)


def add_pdf_to_vectorstore(pdf_path: str):
    """
    Loads a PDF file, splits it into chunks,
    and stores those chunks in ChromaDB.
    Call this when user uploads a PDF.
    """
    # Step 1: Load the PDF and extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split into smaller chunks
    # (LLMs can't read a whole PDF at once - we break it into pieces)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk = 500 characters
        chunk_overlap=50     # chunks overlap by 50 chars so context isn't lost
    )
    chunks = splitter.split_documents(documents)

    # Step 3: Store chunks in ChromaDB
    # (ChromaDB converts text to numbers/vectors so it can find similar content)
    vectorstore.add_documents(chunks)

    return f"✅ Added {len(chunks)} chunks from PDF to knowledge base."


@tool
def search_documents(query: str) -> str:
    """
    Searches the user's uploaded PDF documents for information
    related to the query. Use this alongside search_tool to find
    information from BOTH the web AND uploaded documents.
    """
    # Search ChromaDB for the most relevant chunks
    results = vectorstore.similarity_search(query, k=4)  # top 4 most relevant chunks

    if not results:
        return "No relevant content found in uploaded documents."

    # Format results nicely
    output = []
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "Unknown PDF")
        output.append(f"[Document {i+1}] Source: {source}\n{doc.page_content}")

    return "\n\n---\n\n".join(output)
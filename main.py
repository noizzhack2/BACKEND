import os
import json
import sys
import warnings
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field 
from parse_form_helper import parse_form_from_text, AdaptiveForm, FormField
import numpy as np
# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api_routes import register_routes
from api_route_chat import register_chat_routes

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()

# Silence langchain/old-pydantic compatibility warning on Python 3.14+
if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater."
    )

# Ensure API key is read from environment
API_KEY = os.getenv("GENERATIVE_AI_KEY")
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"
DIRECTORY_PATH = "data"
PERSIST_DIR = "./chroma_db_py"


# --- 2. Pydantic Schema Definitions (Same as before) ---

# 2.3. Define the request body for the API endpoint


# --- 3. Indexing & RAG Initialization (Run Once on Startup) ---


# Global variables to store the RAG components
RAG_CHAIN = None
AVAILABLE_FORMS = set()
# Map: form_name -> {path, content, embedding}
FORM_INDEX: Dict[str, Dict[str, Any]] = {}
# Embeddings instance (initialized at startup if API_KEY is present)
EMBEDDINGS = None

API_KEY = os.getenv("GENERATIVE_AI_KEY") or ""

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global RAG_CHAIN, AVAILABLE_FORMS, FORM_INDEX, EMBEDDINGS
    AVAILABLE_FORMS = get_available_forms()
    RAG_CHAIN = build_rag_chain()

    # Precompute embeddings for each form (if API key available)
    FORM_INDEX.clear()
    if API_KEY:
        try:
            EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
            for form_name in AVAILABLE_FORMS:
                for ext in [".txt", ".pdf"]:
                    candidate = os.path.join(DIRECTORY_PATH, form_name + ext)
                    if os.path.isfile(candidate):
                        try:
                            if ext == ".txt":
                                with open(candidate, "r", encoding="utf-8") as f:
                                    content = f.read()
                            else:
                                pdf_loader = PyPDFLoader(candidate)
                                docs = pdf_loader.load()
                                content = "\n\n".join(doc.page_content for doc in docs)
                            snippet = content[:20000]
                            # Use embed_documents to get consistent vector length
                            emb_list = EMBEDDINGS.embed_documents([snippet])
                            emb = emb_list[0] if emb_list else []
                            FORM_INDEX[form_name] = {"path": candidate, "content": content, "embedding": emb}
                        except Exception as file_error:
                            print(f"❌ Error loading file {candidate}: {file_error}")
                        break
            print(f"✅ Precomputed embeddings for {len(FORM_INDEX)} forms.")
        except Exception as e:
            print(f"⚠️ Warning: failed to precompute embeddings: {e}")
            EMBEDDINGS = None
    else:
        print("⚠️ GENERATIVE_AI_KEY not set — semantic matching disabled. API will use exact filename matching only.")

    print(f"🚀 API Ready to accept requests. Available forms: {AVAILABLE_FORMS}")
    yield
    # Shutdown logic (if needed)
    print("🛑 API shutting down...")

# Initialize FastAPI with lifespan
API = FastAPI(title="Adaptive Form Generator API", lifespan=lifespan)

# CORS setup to allow local frontend and deployments
allowed_origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Optionally allow additional origins via env var (comma separated)
extra_origins = os.getenv("CORS_EXTRA_ORIGINS", "").strip()
if extra_origins:
    allowed_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

API.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_docs(docs: List[Document]) -> str:
    """Combines the content of the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def cosine_similarity(vec_a, vec_b) -> float:
    try:
        a = np.array(vec_a, dtype=float)
        b = np.array(vec_b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0

def get_available_forms():
    """Scan DIRECTORY_PATH for available form names (filenames without extension)."""
    forms = set()
    if os.path.isdir(DIRECTORY_PATH):
        for fname in os.listdir(DIRECTORY_PATH):
            if fname.endswith('.txt') or fname.endswith('.pdf'):
                forms.add(os.path.splitext(fname)[0].lower())
    return forms

def build_rag_chain():
    """Initializes and builds the RAG chain for form generation."""
    
    # Check if API key is available
    if not API_KEY:
        print("⚠️ Warning: GENERATIVE_AI_KEY not set. RAG features disabled.")
        return None
    
    # Load and Index Data (The same stable loading logic)
    print("🛠️ Starting Indexing Phase...")
    
    try:
        # Loaders are split by file type to avoid configuration errors.
        pdf_loader = DirectoryLoader(DIRECTORY_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
        txt_loader = DirectoryLoader(DIRECTORY_PATH, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
        
        pdf_docs: List[Document] = pdf_loader.load()
        txt_docs: List[Document] = txt_loader.load()
        documents: List[Document] = pdf_docs + txt_docs
        
        # Define LLM
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=API_KEY)
        llm_form_generator = llm.with_structured_output(AdaptiveForm)

        # Define the Prompt
        form_prompt_template = """
        You are an expert form generator. Your task is to analyze the user's request and the provided context 
        (if any) and generate a complete JSON schema for an adaptive web form. 
        The form must collect ALL necessary information based on the user's intent.

        Context: {context}

        User Request: {question}

        INSTRUCTIONS: Based on the request and context, generate the Pydantic schema for the form.
        """
        form_prompt = ChatPromptTemplate.from_template(form_prompt_template)

        # Build the LCEL Chain with conditional context handling
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            print(f"✅ Indexing complete. {len(chunks)} chunks indexed.")
            
            chain = (
                {
                    "context": retriever | format_docs, 
                    "question": RunnablePassthrough()
                }
                | form_prompt
                | llm_form_generator
            )
        else:
            print(f"⚠️ Warning: No documents found in '{DIRECTORY_PATH}'. Retriever will use empty context.")
            # Chain without context when no documents are available
            chain = (
                {
                    "context": lambda x: "",  # Empty context
                    "question": RunnablePassthrough()
                }
                | form_prompt
                | llm_form_generator
            )
        
        return chain
    except Exception as e:
        print(f"⚠️ Error initializing RAG chain: {e}")
        return None


# --- 4. REST API Endpoint ---
# Routes are registered from api_routes.py for clarity.
register_routes(API, FORM_INDEX, lambda: EMBEDDINGS, parse_form_from_text, cosine_similarity)
register_chat_routes(API)


# --- 5. Run the API (For standalone execution) ---
if __name__ == "__main__":
    # Ensure ChromaDB server is running if not using persistent local mode (docker run -d -p 8000:8000 --name chroma-server chromadb/chroma)
    print("Starting Uvicorn server...")
    uvicorn.run("main:API", host="0.0.0.0", port=8000, reload=False) # Use reload=True for development

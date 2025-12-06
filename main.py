import os
import json
import sys
import warnings
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field 

# FastAPI Imports
from fastapi import FastAPI, HTTPException
import uvicorn

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os

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
DIRECTORY_PATH = "./data" 
PERSIST_DIR = "./chroma_db_py"


# --- 2. Pydantic Schema Definitions (Same as before) ---

# 2.1. Define a single form field
class FormField(BaseModel):
    name: str = Field(description="A unique programmatic ID for the field (e.g., 'user_email').")
    label: str = Field(description="The user-facing label for the field (e.g., 'Your Full Name').")
    type: Literal["text", "number", "email", "textarea", "checkbox", "date"] = Field(
        description="The HTML input type."
    )
    initilal_value: Optional[Any] = Field(description="The default value for the field, if any.")
    required: bool = Field(description="Whether the field is mandatory.")
    
# 2.2. Define the complete adaptive form structure
class AdaptiveForm(BaseModel):
    title: str = Field(description="A clear and concise title for the form.")
    description: str = Field(description="A brief explanation of the form's purpose.")
    fields: List[FormField] = Field(description="A list of all required input fields.")

# 2.3. Define the request body for the API endpoint
class FormRequest(BaseModel):
    user_request: str = Field(description="The user's natural language request for a form (e.g., 'I need a trip expense report form').")


# --- 3. Indexing & RAG Initialization (Run Once on Startup) ---


# Global variables to store the RAG components
RAG_CHAIN = None
AVAILABLE_FORMS = set()
API_KEY = "AIzaSyBXgfxPmcpcAktzecqiKO_5FS5_gU9VnhA"

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global RAG_CHAIN, AVAILABLE_FORMS
    AVAILABLE_FORMS = get_available_forms()
    RAG_CHAIN = build_rag_chain()
    print(f"🚀 API Ready to accept requests. Available forms: {AVAILABLE_FORMS}")
    yield
    # Shutdown logic (if needed)
    print("🛑 API shutting down...")

# Initialize FastAPI with lifespan
API = FastAPI(title="Adaptive Form Generator API", lifespan=lifespan)

def format_docs(docs: List[Document]) -> str:
    """Combines the content of the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

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



@API.post("/generate_form", response_model=AdaptiveForm)
async def generate_form_endpoint(request: FormRequest):
    """
    Accepts a user request and returns a structured JSON schema for an adaptive form.
    """
    # Extract requested form name (simple heuristic: lowercase, strip spaces)
    requested_form = request.user_request.strip().lower().replace(' form', '').replace('form', '').replace('for ', '').replace('a ', '').replace('the ', '').replace('an ', '').replace('request ', '').replace('create ', '').replace('make ', '').replace('need ', '').replace('want ', '').replace('to ', '').replace('get ', '').replace('generate ', '').replace('build ', '').replace('provide ', '').replace('show ', '').replace('give ', '').replace('open ', '').replace('start ', '').replace('submit ', '').replace('fill ', '').replace('registration ', 'registration').replace('expense ', 'expense').replace('report ', 'report').replace('trip ', 'trip').replace('conference ', 'conference').replace('application ', 'application').replace('feedback ', 'feedback').replace('survey ', 'survey').replace('contact ', 'contact').replace('signup ', 'signup').replace('sign up ', 'signup').replace('sign-up ', 'signup').replace('join ', 'join').replace('apply ', 'apply').replace('enroll ', 'enroll').replace('register ', 'register').replace('booking ', 'booking').replace('reservation ', 'reservation').replace('order ', 'order').replace('purchase ', 'purchase').replace('request ', 'request').replace('form', '').strip()

    # Find the best matching form file
    matched_form = None
    for form_name in AVAILABLE_FORMS:
        if form_name in requested_form or requested_form in form_name:
            matched_form = form_name
            break
    if not matched_form:
        raise HTTPException(
            status_code=414,
            detail=f"Form not found. Available forms: {sorted(AVAILABLE_FORMS)}"
        )

    # Find the file path for the matched form
    form_file_path = None
    for ext in [".txt", ".pdf"]:
        candidate = os.path.join(DIRECTORY_PATH, matched_form + ext)
        if os.path.isfile(candidate):
            form_file_path = candidate
            break
    if not form_file_path:
        raise HTTPException(status_code=500, detail="Matched form file not found on disk.")

    # Load the matched form file as a document
    if form_file_path.endswith(".txt"):
        with open(form_file_path, encoding="utf-8") as f:
            form_content = f.read()
        docs = [Document(page_content=form_content)]
    elif form_file_path.endswith(".pdf"):
        # Use PyPDFLoader to load PDF
        pdf_loader = PyPDFLoader(form_file_path)
        docs = pdf_loader.load()
    else:
        raise HTTPException(status_code=500, detail="Unsupported form file type.")

    # Build a temporary RAG chain for this form only
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=None)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=API_KEY)
        llm_form_generator = llm.with_structured_output(AdaptiveForm)
        form_prompt_template = """
        You are an expert form generator. Your task is to analyze the user's request and the provided context 
        (if any) and generate a complete JSON schema for an adaptive web form. 
        The form must collect ALL necessary information based on the user's intent.

        Context: {context}

        User Request: {question}

        INSTRUCTIONS: Based on the request and context, generate the Pydantic schema for the form.
        """
        form_prompt = ChatPromptTemplate.from_template(form_prompt_template)
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | form_prompt
            | llm_form_generator
        )
        form_pydantic_object: AdaptiveForm = await chain.ainvoke(request.user_request)
        return form_pydantic_object
    except Exception as e:
        print(f"Error during chain execution: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate form: An internal error occurred."
        )


# --- 5. Run the API (For standalone execution) ---
if __name__ == "__main__":
    # Ensure ChromaDB server is running if not using persistent local mode (docker run -d -p 8000:8000 --name chroma-server chromadb/chroma)
    print("Starting Uvicorn server...")
    uvicorn.run("main:API", host="0.0.0.0", port=8000, reload=False) # Use reload=True for development

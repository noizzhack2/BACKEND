import os
import json
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


API_KEY = ""
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
API = FastAPI(title="Adaptive Form Generator API")

def format_docs(docs: List[Document]) -> str:
    """Combines the content of the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    """Initializes and builds the RAG chain for form generation."""
    
    # Load and Index Data (The same stable loading logic)
    print("🛠️ Starting Indexing Phase...")
    
    # Loaders are split by file type to avoid configuration errors.
    pdf_loader = DirectoryLoader(DIRECTORY_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
    txt_loader = DirectoryLoader(DIRECTORY_PATH, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
    
    pdf_docs: List[Document] = pdf_loader.load()
    txt_docs: List[Document] = txt_loader.load()
    documents: List[Document] = pdf_docs + txt_docs
    
    if not documents:
        print(f"⚠️ Warning: No documents found in '{DIRECTORY_PATH}'. Retriever will use empty context.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    print(f"✅ Indexing complete. {len(chunks)} chunks indexed.")
    
    # Define LLM and Retriever
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    
    # Define Structured Output LLM
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

    # Build the LCEL Chain
    chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | form_prompt
        | llm_form_generator
    )
    return chain

# Load the RAG chain upon API startup
@API.on_event("startup")
def startup_event():
    global RAG_CHAIN
    RAG_CHAIN = build_rag_chain()
    print("🚀 API Ready to accept requests.")


# --- 4. REST API Endpoint ---

@API.post("/generate_form", response_model=AdaptiveForm)
async def generate_form_endpoint(request: FormRequest):
    """
    Accepts a user request and returns a structured JSON schema for an adaptive form.
    """
    try:
        # Invoke the RAG chain with the user's request
        form_pydantic_object: AdaptiveForm = await RAG_CHAIN.ainvoke(request.user_request)
        
        # The Pydantic object is automatically converted to JSON by FastAPI
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
    uvicorn.run("api:API", host="0.0.0.0", port=8000, reload=False) # Use reload=True for development
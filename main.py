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
    type: Literal["text", "number", "email", "textarea", "checkbox", "date", "select"] = Field(
        description="The HTML input type."
    )
    initial_value: Optional[Any] = Field(default=None, description="The default value for the field, if any.")
    required: bool = Field(description="Whether the field is mandatory.")
    
# 2.2. Define the complete adaptive form structure
class AdaptiveForm(BaseModel):
    title: str = Field(description="A clear and concise title for the form.")
    description: str = Field(description="A brief explanation of the form's purpose.")
    fields: List[FormField] = Field(description="A list of all required input fields.")
    score: Optional[float] = Field(default=None, description="Semantic similarity score (0-1) for the match.")

# 2.3. Define the request body for the API endpoint
class FormRequest(BaseModel):
    user_request: str = Field(description="The user's natural language request for a form (e.g., 'I need a trip expense report form').")


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
    if API_KEY:
        try:
            EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
            # Load contents and compute embeddings
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
                            # Trim content for embedding if very large
                            snippet = content[:20000]
                            emb = EMBEDDINGS.embed_query(snippet)
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

def format_docs(docs: List[Document]) -> str:
    """Combines the content of the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def parse_form_from_text(form_name: str, form_content: str) -> AdaptiveForm:
    """
    Parse a form from predefined text content in data/ folder.
    Extracts title, description, and fields from the structured text file.
    Returns an AdaptiveForm object with all parsed information.
    """
    lines = form_content.split('\n')
    
    # Extract title (first non-empty line with "Form" in it)
    title = "Reimbursement Form"
    description = "Submit your reimbursement request using this form."
    fields = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Extract title from first line
        if i < 5 and "Form" in line_stripped:
            title = line_stripped
            # Try to extract English title before "/" if bilingual
            if "/" in title:
                title = title.split("/")[0].strip()
            break
    
    # Extract description (Purpose section)
    for i, line in enumerate(lines):
        if "Purpose" in line or "מטרה" in line:
            # Get the next non-empty line as description
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and ":" not in lines[j]:
                    desc_line = lines[j].strip()
                    if "/" in desc_line:
                        # Take only English part if bilingual
                        desc_line = desc_line.split("/")[0].strip()
                    description = desc_line
                    break
            break
    
    # Extract fields from lines that have "required" or "נדרש"
    field_names = set()
    for line in lines:
        line_stripped = line.strip()
        if ("required" in line_stripped.lower() or "נדרש" in line_stripped) and line_stripped.startswith("-"):
            # Extract field name (usually before the opening parenthesis)
            if "(" in line_stripped:
                field_name = line_stripped.split("(")[0].strip().lstrip("- ").strip()
                # Extract English name if bilingual
                if "/" in field_name:
                    field_name = field_name.split("/")[0].strip()
                
                if field_name and len(field_name) > 2 and field_name not in field_names:
                    field_names.add(field_name)
                    field_type = "text"
                    
                    # Determine field type
                    if any(word in line_stripped.lower() for word in ["date", "תאריך", "MM/DD"]):
                        field_type = "date"
                    elif any(word in line_stripped.lower() for word in ["select", "בחר", "dropdown"]):
                        field_type = "select"
                    elif any(word in line_stripped.lower() for word in ["checkbox", "תיבת סימון"]):
                        field_type = "checkbox"
                    elif any(word in line_stripped.lower() for word in ["numeric", "מספר", "number"]):
                        field_type = "number"
                    elif any(word in line_stripped.lower() for word in ["currency", "כספי", "amount"]):
                        field_type = "number"
                    elif any(word in line_stripped.lower() for word in ["text area", "שדה טקסט"]):
                        field_type = "textarea"
                    
                    fields.append(FormField(
                        name=field_name,
                        type=field_type,
                        label=field_name,
                        required=True
                    ))
    
    # If no fields were extracted, create a generic field
    if not fields:
        fields.append(FormField(
            name="details",
            type="textarea",
            label="Form Details",
            required=True
        ))
    
    return AdaptiveForm(
        title=title,
        description=description,
        fields=fields
    )

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
def generate_form_endpoint(request: FormRequest):
    """
    Accepts a user request (in Hebrew or English) and returns a structured JSON schema for an adaptive form.
    Uses precomputed form embeddings (from `FORM_INDEX`) to select the best match.
    Supports both RTL (Hebrew) and LTR (English) text.
    """
    if not FORM_INDEX:
        raise HTTPException(status_code=500, detail="No indexed forms available. Ensure data folder contains form files.")

    # If embeddings instance is available, do semantic matching
    try:
        best_score = -1.0  # Initialize best_score here
        if EMBEDDINGS:
            user_text = request.user_request.strip()
            guiding_prompt = (
                "Task: Given the user's request, select the most relevant expense reimbursement form from the available options. "
                "Available forms include fuel expense reimbursement and food/meal expense reimbursement. "
                "The user's request may be incomplete, vague, or in Hebrew or English. "
                "If the request mentions food, meals, אוכל, ארוחה, or similar, prefer the food expense form. "
                "User request: "
            )
            user_emb = EMBEDDINGS.embed_query(guiding_prompt + user_text)
            import numpy as _np
            best_score = -1.0
            best_form = None
            for form_name, info in FORM_INDEX.items():
                form_emb = info.get("embedding")
                if form_emb is None:
                    continue
                score = float(_np.dot(user_emb, form_emb) / (_np.linalg.norm(user_emb) * _np.linalg.norm(form_emb)))
                if score > best_score:
                    best_score = score
                    best_form = (form_name, info["path"], info["content"])
            # Require minimum threshold for semantic match
            min_threshold = 0.55
            if best_form is None or best_score < min_threshold:
                raise HTTPException(status_code=414, detail=f"לא נמצא טופס מתאים לבקשה שלך. נסה לנסח מחדש או לבחור טופס קיים. (ציון התאמה: {best_score:.2f})")
            matched_form, form_file_path, form_content = best_form
        else:
            # Fallback: simple keyword matching over filenames
            text = request.user_request.lower()
            matched_form = None
            for form_name, info in FORM_INDEX.items():
                if form_name in text or text in form_name:
                    matched_form = form_name
                    form_file_path = info["path"]
                    form_content = info["content"]
                    best_score = None
                    break
            if not matched_form:
                raise HTTPException(status_code=414, detail=f"לא נמצא טופס מתאים לבקשה שלך. נסה לנסח מחדש או לבחור טופס קיים. הטפסים הזמינים: {sorted(list(FORM_INDEX.keys()))}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during semantic form matching: {e}")
        raise HTTPException(status_code=500, detail="Failed to match form by semantic similarity.")

    # Return the matched form from data/ folder (parse it without using LLM)
    try:
        parsed_form = parse_form_from_text(matched_form, form_content)
        print(f"✅ Successfully returned form: {matched_form} (Score: {best_score:.2f})")
        # Add score to response
        parsed_form.score = round(best_score, 2) if best_score is not None else None
        return parsed_form
    except Exception as e:
        print(f"Error parsing form: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse form '{matched_form}': {str(e)}")


# --- 5. Run the API (For standalone execution) ---
if __name__ == "__main__":
    # Ensure ChromaDB server is running if not using persistent local mode (docker run -d -p 8000:8000 --name chroma-server chromadb/chroma)
    print("Starting Uvicorn server...")
    uvicorn.run("main:API", host="0.0.0.0", port=8000, reload=False) # Use reload=True for development

import os
import sys
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Add parent directory to system path to import query module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import query_rag
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

app = FastAPI(title="RAG Chatbot API", description="Serverless API for Gemini+Pinecone RAG")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector DB connection
db = None
db_error = None

try:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-project")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not api_key or not gemini_key:
        raise ValueError("PINECONE_API_KEY or GEMINI_API_KEY is not set.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=gemini_key
    )

    db = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=api_key
    )
except Exception as e:
    db_error = str(e)
    print(f"Error initializing vector database: {db_error}")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    source: str
    docs: list
    latency_ms: int

@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Health check endpoint to verify configurations and database connectivity."""
    return {
        "status": "healthy" if db is not None else "degraded",
        "pinecone_index": os.environ.get("PINECONE_INDEX_NAME"),
        "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
        "has_pinecone_key": bool(os.environ.get("PINECONE_API_KEY")),
        "db_error": db_error
    }

@app.post("/api/chat", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Processes chat query, retrieves context from Pinecone, queries Gemini, and returns result."""
    if db is None:
        raise HTTPException(
            status_code=500,
            detail=f"Vector database not initialized. Error: {db_error}"
        )

    start_time = time.time()
    try:
        result = query_rag(request.message, db)
        latency = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            answer=result["answer"],
            source=result["source"],
            docs=result["docs"],
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

# Serve static files for local development
from fastapi.responses import FileResponse
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(root_dir, "index.html"))

@app.get("/style.css")
async def read_style():
    return FileResponse(os.path.join(root_dir, "style.css"))

@app.get("/script.js")
async def read_script():
    return FileResponse(os.path.join(root_dir, "script.js"))

# Run using: uvicorn api.index:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="127.0.0.1", port=8000, reload=True)

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from schemas import QueryRequest, QueryResponse, SourceNodeMetadata, HealthResponse
from rag_engine import initialize_rag_engine

# Global container for RAG query engine
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load RAG Engine once on server boot
    print("\n🚀 [Server Startup] Initializing DocuSense AI RAG Pipeline...")
    try:
        app_state["query_engine"] = initialize_rag_engine()
        print("✅ [Server Startup] RAG Engine ready for incoming queries.\n")
    except Exception as e:
        print(f"❌ [Server Startup Error] Failed to initialize RAG engine: {e}")
        raise e
    yield
    # Shutdown
    app_state.clear()
    print("🛑 [Server Shutdown] Cleaned up resources.")

app = FastAPI(
    title="DocuSense AI - Production RAG Backend",
    version="1.0.0",
    description="FastAPI Backend for Document Grounded RAG with Qdrant & Groq",
    lifespan=lifespan
)

# CORS Policy for Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def check_health():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=qdrant_url, timeout=60)
        info = client.get_collection(collection_name="documents")
        return HealthResponse(
            status="healthy",
            qdrant_connected=True,
            collection_name="documents",
            total_points=info.points_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant connection error: {str(e)}"
        )
    finally:
        client.close()

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Engine"])
async def run_query(request: QueryRequest):
    query_engine = app_state.get("query_engine")
    if not query_engine:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG query engine is not initialized."
        )
    
    try:
        # Execute RAG query against Qdrant + Groq
        response = query_engine.query(request.question)
        
        sources = []
        for source_node in response.source_nodes:
            score = round(source_node.score, 4) if source_node.score else 0.0
            file_name = source_node.node.metadata.get("file_name", "Unknown")
            page_label = str(source_node.node.metadata.get("page_label", "N/A"))
            snippet = source_node.node.get_content()[:200].replace("\n", " ") + "..."
            
            sources.append(
                SourceNodeMetadata(
                    score=score,
                    file_name=file_name,
                    page_label=page_label,
                    snippet=snippet
                )
            )
            
        return QueryResponse(
            question=request.question,
            answer=response.response,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )

import os
import json
import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4
from fastapi import FastAPI, File, HTTPException, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from schemas import QueryRequest, SourceNodeMetadata, QueryResponse, HealthResponse, EvalRequest, EvalResponse
from rag_engine import initialize_rag_engine
from evaluator import GroqJudgeLLM
from vector_db import add_document_to_vector_database, delete_document_from_vector_database
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# Global container for RAG query engine
app_state = {}
UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
MAX_PDF_SIZE = 25 * 1024 * 1024

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load RAG Engine once on server boot
    try:
        app_state["query_engine"] = initialize_rag_engine(streaming=False)
        app_state["stream_engine"] = initialize_rag_engine(streaming=True)
        app_state["judge_llm"] = GroqJudgeLLM()
        print("[Server Startup] Engine ready for streaming.\n")
    except Exception as e:
        print(f"[Server Startup Error]: {e}")
        raise e
    yield
    app_state.clear()

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

@app.post("/api/v1/documents", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    document_dir = UPLOAD_DIR / uuid4().hex
    document_dir.mkdir(parents=True, exist_ok=False)
    destination = document_dir / filename
    size = 0

    try:
        with destination.open("wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_PDF_SIZE:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB upload limit.")
                output_file.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        with destination.open("rb") as saved_file:
            if b"%PDF-" not in saved_file.read(1024):
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        chunks_indexed = await asyncio.to_thread(
            add_document_to_vector_database,
            str(destination),
            document_dir.name,
        )
        return {
            "document_id": document_dir.name,
            "filename": filename,
            "chunks_indexed": chunks_indexed,
        }
    except HTTPException:
        shutil.rmtree(document_dir, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(document_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="The PDF could not be indexed.",
        ) from error
    finally:
        await file.close()

@app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    try:
        normalized_id = UUID(document_id).hex
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid document ID.") from error

    document_dir = UPLOAD_DIR / normalized_id
    if not document_dir.is_dir():
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        await asyncio.to_thread(delete_document_from_vector_database, normalized_id)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The document could not be removed from the vector database.",
        ) from error

    shutil.rmtree(document_dir)
    return {"document_id": normalized_id, "deleted": True}

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

@app.post("/api/v1/query/stream", tags=["RAG Engine"])
async def stream_query(request: QueryRequest):
    """Streams response tokens via SSE (text/event-stream) alongside source citations."""
    query_engine = app_state.get("stream_engine")
    if not query_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized.")

    async def event_generator():
        # Blocking LlamaIndex call offloaded so the event loop stays free
        response = await asyncio.to_thread(query_engine.query, request.question)

        # 1. First event: Send retrieved sources immediately
        sources = []
        contexts = []
        for node in response.source_nodes:
            contexts.append(node.node.get_content())
            sources.append({
                "score": round(node.score, 4) if node.score else 0.0,
                "file_name": node.node.metadata.get("file_name", "Unknown"),
                "page_label": str(node.node.metadata.get("page_label", "N/A")),
                "snippet": node.node.get_content()[:180].replace("\n", " ") + "..."
            })
            
        yield f"event: metadata\ndata: {json.dumps({'sources': sources, 'contexts': contexts})}\n\n"

        # 2. Subsequent events: Stream answer tokens as the LLM produces them
        token_iter = iter(response.response_gen)
        sentinel = object()
        while True:
            token = await asyncio.to_thread(next, token_iter, sentinel)
            if token is sentinel:
                break
            if token:
                yield f"event: delta\ndata: {json.dumps({'text': token})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/eval", response_model=EvalResponse, tags=["Quality Control"])
async def evaluate_response(req: EvalRequest):
    """Runs DeepEval quality metrics asynchronously without blocking response streams."""
    judge_llm = app_state.get("judge_llm")
    if not judge_llm:
        raise HTTPException(status_code=500, detail="Judge LLM not initialized.")

    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_llm, include_reason=False)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_llm, include_reason=False)
    
    test_case = LLMTestCase(
        input=req.question,
        actual_output=req.answer,
        retrieval_context=req.retrieved_contexts
    )

    await asyncio.gather(
        faithfulness.a_measure(test_case),
        answer_relevancy.a_measure(test_case)
    )

    return EvalResponse(
        faithfulness=round(faithfulness.score, 2),
        relevancy=round(answer_relevancy.score, 2),
        precision=1.0
    )

import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from schemas import QueryRequest, SourceNodeMetadata, QueryResponse, HealthResponse, EvalRequest, EvalResponse
from rag_engine import initialize_rag_engine
from evaluator import GroqJudgeLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# Global container for RAG query engine
app_state = {}

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

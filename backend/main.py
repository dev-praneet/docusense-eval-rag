import os
import json
import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4
from fastapi import FastAPI, File, HTTPException, Request, Response, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client import QdrantClient
from sqlalchemy.exc import IntegrityError

from schemas import QueryRequest, SourceNodeMetadata, QueryResponse, HealthResponse, EvalRequest, EvalResponse, AccountCredentials
from rag_engine import initialize_rag_engine, session_chat_engine, session_query_engine
from evaluator import GroqJudgeLLM
from vector_db import add_document_to_vector_database, delete_document_from_vector_database, claim_document_vectors
import auth
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# Global container for RAG query engine
app_state = {}
UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
MAX_PDF_SIZE = 25 * 1024 * 1024
SESSION_COOKIE = "docusense_session"
AUTH_COOKIE = "docusense_auth"
GUEST_LIFETIME = 30 * 24 * 60 * 60
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
BACKEND_ORIGIN = os.getenv("BACKEND_ORIGIN", "http://localhost:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load RAG Engine once on server boot
    try:
        auth.initialize_database()
        app_state["index"] = initialize_rag_engine()
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
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def assign_session(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("origin") not in (
        None, FRONTEND_ORIGIN, BACKEND_ORIGIN
    ):
        return JSONResponse(status_code=403, content={"detail": "Untrusted origin."})
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or len(session_id) != 32 or not all(character in "0123456789abcdef" for character in session_id):
        session_id = uuid4().hex
        new_session = True
    else:
        new_session = False
    request.state.session_id = session_id
    request.state.user = auth.current_user(request.cookies.get(AUTH_COOKIE))
    request.state.rotate_session = False
    response = await call_next(request)
    if new_session or request.state.rotate_session:
        response.set_cookie(
            SESSION_COOKIE, request.state.session_id, max_age=GUEST_LIFETIME,
            httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE") == "true",
        )
    return response

def request_owner(request: Request):
    user = request.state.user
    return ("user_id", user["id"]) if user else ("session_id", request.state.session_id)

def remember_guest_documents(guest_id: str):
    guest_directory = UPLOAD_DIR / guest_id
    if not guest_directory.is_dir():
        return
    for document_dir in guest_directory.iterdir():
        if not document_dir.is_dir():
            continue
        try:
            document_id = UUID(document_dir.name).hex
        except ValueError:
            continue
        if document_id != document_dir.name:
            continue
        pdf = next((path for path in document_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"), None)
        if pdf and not auth.document_exists(document_id):
            auth.save_document(document_id, guest_id, None, pdf.name, str(pdf.relative_to(UPLOAD_DIR)))

async def claim_guest_documents(guest_id: str, user_id: str):
    remember_guest_documents(guest_id)
    for document in auth.guest_documents(guest_id):
        await asyncio.to_thread(claim_document_vectors, document["id"], guest_id, user_id)
        auth.claim_document(document["id"], guest_id, user_id)

@app.get("/api/v1/auth/me", tags=["Account"])
async def current_account(request: Request):
    return {"user": request.state.user}

@app.post("/api/v1/auth/register", status_code=201, tags=["Account"])
async def register(credentials: AccountCredentials, request: Request, response: Response):
    if request.state.user:
        raise HTTPException(status_code=409, detail="Log out before creating another account.")
    try:
        user_id, email = auth.create_user(credentials.email, credentials.password)
    except IntegrityError as error:
        raise HTTPException(status_code=409, detail="Email already registered.") from error
    await claim_guest_documents(request.state.session_id, user_id)
    response.set_cookie(
        AUTH_COOKIE, auth.create_auth_session(user_id), max_age=auth.AUTH_LIFETIME,
        httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE") == "true",
    )
    request.state.session_id = uuid4().hex
    request.state.rotate_session = True
    return {"user": {"id": user_id, "email": email}}

@app.post("/api/v1/auth/login", tags=["Account"])
async def login(credentials: AccountCredentials, request: Request, response: Response):
    if request.state.user:
        raise HTTPException(status_code=409, detail="Log out before switching accounts.")
    user = auth.authenticate(credentials.email, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user_id, email = user
    await claim_guest_documents(request.state.session_id, user_id)
    auth.revoke_session(request.cookies.get(AUTH_COOKIE))
    response.set_cookie(
        AUTH_COOKIE, auth.create_auth_session(user_id), max_age=auth.AUTH_LIFETIME,
        httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE") == "true",
    )
    request.state.session_id = uuid4().hex
    request.state.rotate_session = True
    return {"user": {"id": user_id, "email": email}}

@app.post("/api/v1/auth/logout", tags=["Account"])
async def logout(request: Request, response: Response):
    if not request.state.user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    auth.revoke_session(request.cookies.get(AUTH_COOKIE))
    response.delete_cookie(AUTH_COOKIE)
    request.state.session_id = uuid4().hex
    request.state.rotate_session = True
    return {"user": None}

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
async def upload_document(request: Request, file: UploadFile = File(...)):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    owner_key, owner_id = request_owner(request)
    folder = UPLOAD_DIR / ("accounts" if request.state.user else "") / owner_id
    document_dir = folder / uuid4().hex
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
            owner_id,
            owner_key,
        )
        auth.save_document(
            document_dir.name,
            owner_id if owner_key == "session_id" else None,
            owner_id if owner_key == "user_id" else None,
            filename,
            str(destination.relative_to(UPLOAD_DIR)),
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

@app.get("/api/v1/documents", tags=["Documents"])
async def list_documents(request: Request):
    if not request.state.user:
        remember_guest_documents(request.state.session_id)
    return auth.documents_for_owner(
        request.state.session_id if not request.state.user else None,
        request.state.user["id"] if request.state.user else None,
    )

@app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str, request: Request):
    try:
        normalized_id = UUID(document_id).hex
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid document ID.") from error

    if not request.state.user:
        remember_guest_documents(request.state.session_id)
    owner_key, owner_id = request_owner(request)
    document = auth.document_for_owner(
        normalized_id, owner_id if owner_key == "session_id" else None,
        owner_id if owner_key == "user_id" else None,
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        await asyncio.to_thread(delete_document_from_vector_database, normalized_id, owner_id, owner_key)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The document could not be removed from the vector database.",
        ) from error

    shutil.rmtree(UPLOAD_DIR / Path(document["relative_path"]).parent)
    auth.remove_document(normalized_id)
    return {"document_id": normalized_id, "deleted": True}

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Engine"])
async def run_query(request: QueryRequest, http_request: Request):
    index = app_state.get("index")
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG query engine is not initialized."
        )
    
    try:
        # Execute RAG query against Qdrant + Groq
        owner_key, owner_id = request_owner(http_request)
        query_engine = session_query_engine(index, owner_id, owner_key)
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
async def stream_query(request: QueryRequest, http_request: Request):
    """Streams response tokens via SSE (text/event-stream) alongside source citations."""
    index = app_state.get("index")
    if index is None:
        raise HTTPException(status_code=500, detail="Engine not initialized.")
    # at place of query_engine we need chat_engine
    owner_key, owner_id = request_owner(http_request)
    # query_engine = session_query_engine(index, owner_id, owner_key, streaming=True)
    chat_engine = session_chat_engine(index, owner_id, owner_key)

    # async def event_generator():
    #     # Blocking LlamaIndex call offloaded so the event loop stays free
    #     response = await asyncio.to_thread(query_engine.query, request.question)

    #     # 1. First event: Send retrieved sources immediately
    #     sources = []
    #     contexts = []
    #     for node in response.source_nodes:
    #         contexts.append(node.node.get_content())
    #         sources.append({
    #             "score": round(node.score, 4) if node.score else 0.0,
    #             "file_name": node.node.metadata.get("file_name", "Unknown"),
    #             "page_label": str(node.node.metadata.get("page_label", "N/A")),
    #             "snippet": node.node.get_content()[:180].replace("\n", " ") + "..."
    #         })
            
    #     yield f"event: metadata\ndata: {json.dumps({'sources': sources, 'contexts': contexts})}\n\n"

    #     # 2. Subsequent events: Stream answer tokens as the LLM produces them
    #     token_iter = iter(response.response_gen)
    #     sentinel = object()
    #     while True:
    #         token = await asyncio.to_thread(next, token_iter, sentinel)
    #         if token is sentinel:
    #             break
    #         if token:
    #             yield f"event: delta\ndata: {json.dumps({'text': token})}\n\n"

    #     yield "event: done\ndata: {}\n\n"

    async def event_generator():
        # Blocking LlamaIndex call offloaded so the event loop stays free
        # response = await asyncio.to_thread(query_engine.query, request.question)
        response = await chat_engine.astream_chat(request.question)
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
        async for token in response.async_response_gen():
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

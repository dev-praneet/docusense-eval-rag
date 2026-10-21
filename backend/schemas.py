from pydantic import BaseModel, Field
from typing import List, Optional

class AccountCredentials(BaseModel):
    email: str = Field(..., min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(..., min_length=12, max_length=128)

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the primary algorithm discussed?")
    similarity_top_k: Optional[int] = Field(default=3, ge=1, le=10)

class SourceNodeMetadata(BaseModel):
    score: Optional[float]
    file_name: str
    page_label: str
    snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceNodeMetadata]

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_name: str
    total_points: int

class EvalRequest(BaseModel):
    question: str
    answer: str
    retrieved_contexts: List[str]

class EvalResponse(BaseModel):
    faithfulness: float
    relevancy: float
    precision: float

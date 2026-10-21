import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ingest import load_and_chunk_documents

DATA_DIR = "./data"
QDRANT_STORAGE_PATH = "./qdrant_data"
COLLECTION_NAME = "documents"

def index_nodes(nodes):
    if not nodes:
        return 0

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url, timeout=60)
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        print(f"\nEmbedded and indexed {len(nodes)} chunks into Qdrant ('{COLLECTION_NAME}').")
        return len(nodes)
    finally:
        client.close()

def add_document_to_vector_database(file_path: str, document_id: str, owner_id: str, owner_key: str = "session_id"):
    nodes = load_and_chunk_documents(file_path, document_id=document_id, **{owner_key: owner_id})
    return index_nodes(nodes)

def delete_document_from_vector_database(document_id: str, owner_id: str, owner_key: str = "session_id"):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url, timeout=60)
    try:
        return client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=document_id),
                        ),
                        FieldCondition(
                            key=owner_key,
                            match=MatchValue(value=owner_id),
                        ),
                    ]
                )
            ),
            wait=True,
        )
    finally:
        client.close()

def claim_document_vectors(document_id: str, guest_id: str, user_id: str):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url, timeout=60)
    try:
        guest_filter = Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=document_id)),
            FieldCondition(key="session_id", match=MatchValue(value=guest_id)),
        ])
        account_filter = Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=document_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ])
        client.set_payload(COLLECTION_NAME, {"user_id": user_id}, points=guest_filter, wait=True)
        client.delete_payload(COLLECTION_NAME, ["session_id"], points=account_filter, wait=True)
    finally:
        client.close()

def build_vector_database():
    nodes = load_and_chunk_documents(DATA_DIR)
    index_nodes(nodes)

if __name__ == "__main__":
    load_dotenv()

    try:
        build_vector_database()
    except Exception as e:
        print(f"\nError during vector database indexing: {e}")

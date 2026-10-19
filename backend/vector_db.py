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

def add_document_to_vector_database(file_path: str, document_id: str):
    nodes = load_and_chunk_documents(file_path, document_id=document_id)
    return index_nodes(nodes)

def delete_document_from_vector_database(document_id: str):
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
                        )
                    ]
                )
            ),
            wait=True,
        )
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

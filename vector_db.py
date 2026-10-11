import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ingest import load_and_chunk_documents

load_dotenv()
DATA_DIR = "./data"
QDRANT_STORAGE_PATH = "./qdrant_data"
COLLECTION_NAME = "documents"

def build_vector_database():
    nodes = load_and_chunk_documents(DATA_DIR)
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    # client = QdrantClient(path=QDRANT_STORAGE_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print(f"\nDay 2 Complete: Embedded and indexed {len(nodes)} chunks locally into Qdrant ('{COLLECTION_NAME}').")
    return index

if __name__ == "__main__":
    try:
        build_vector_database()
    except Exception as e:
        print(f"\nError during vector database indexing: {e}")

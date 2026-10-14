import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
load_dotenv()
COLLECTION_NAME = "documents"

def initialize_rag_engine(streaming: bool = False):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"\n[1/3] Connecting to Qdrant Docker server at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, timeout=60)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )
    print("[2/3] Loading local embedding model (BAAI/bge-small-en-v1.5)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "gsk_your_free_groq_key_here":
        raise ValueError("GROQ_API_KEY is missing or unconfigured in your .env file!")

    llm = Groq(
        model="openai/gpt-oss-20b",
        # model="llama-3.3-70b-versatile",
        api_key=groq_api_key
    )
    Settings.embed_model = embed_model
    Settings.llm = llm
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    # Retrieve top 3 most relevant context chunks per query
    query_engine = index.as_query_engine(similarity_top_k=3, streaming=streaming)
    return query_engine

def ask_question(query_text: str):
    query_engine = initialize_rag_engine()
    print(f"\nUser Question: '{query_text}'")
    print("Searching Qdrant vector space & generating response with Groq...\n")
    response = query_engine.query(query_text)
    print("=================== DocuSense AI Response =")
    print(response.response)
    print("===========================================")
    print("\n--- Source Chunks Retained ---")

    for idx, source_node in enumerate(response.source_nodes):
        score = round(source_node.score, 4) if source_node.score else "N/A"
        file_name = source_node.node.metadata.get("file_name", "Unknown")
        page = source_node.node.metadata.get("page_label", "N/A")
        print(f"[{idx+1}] Similarity Score: {score} | File: {file_name} (Page {page})")
        snippet = source_node.node.get_content()[:120].replace('\n', ' ')
        print(f"     Snippet: '{snippet}...'\n")

if __name__ == "__main__":
    sample_query = "What is the main algorithm or topic described in the document?"
    ask_question(sample_query)

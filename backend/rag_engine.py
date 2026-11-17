import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

COLLECTION_NAME = "documents"

def initialize_rag_engine():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"\n[1/3] Connecting to Qdrant Docker server at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, timeout=60)
    aclient = AsyncQdrantClient(url=qdrant_url, timeout=60)
    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
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
    return index

def session_query_engine(index, owner_id: str, owner_key: str = "session_id", streaming: bool = False):
    return index.as_query_engine(
        similarity_top_k=3,
        streaming=streaming,
        filters=MetadataFilters(filters=[MetadataFilter(key=owner_key, value=owner_id)]),
    )

# In-memory dictionary storing memory buffers per user_id
chat_memories = {}

def session_chat_engine(index, owner_id: str, owner_key: str, similarity_top_k: int = 3):
    # 1. Enforce multi-tenant isolation
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key=owner_key,
                value=owner_id,
                operator=FilterOperator.EQ # this is the default btw
            )
        ]
    )

    # 2. Get or create memory for this specific user session
    if owner_id not in chat_memories:
        chat_memories[owner_id] = ChatMemoryBuffer.from_defaults(token_limit=500)

    # 3. Create isolated chat engine from index
    return index.as_chat_engine(
        chat_mode="condense_plus_context", # Best mode for multi-turn RAG
        similarity_top_k=similarity_top_k,
        filters=filters,
        memory=chat_memories[owner_id],
        system_prompt="You are DocuSense AI, an assistant that answers questions based on uploaded documents."
    )

def ask_question(query_text: str):
    query_engine = initialize_rag_engine().as_query_engine(similarity_top_k=3)
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
    load_dotenv()
    sample_query = "What is the main algorithm or topic described in the document?"
    ask_question(sample_query)

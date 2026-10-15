from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
import os
from llama_index.core.node_parser import SentenceSplitter

DATA_DIR = "./data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_and_chunk_documents(data_path: str):
    if not os.path.exists(data_path) or not os.listdir(data_path):
        raise ValueError(f"Directory '{data_path}' does not exist or is empty")
    
    reader = SimpleDirectoryReader(input_dir=data_path)
    documents = reader.load_data()
    print(f"successfully loaded {len(documents)} document sources/pages")

    node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"generate {len(nodes)} nodes")

    if nodes:
        print("\n --- Sample chunk (Node 0) preview --- ")
        print(f"chunk ID: {nodes[0].node_id}")
        print(f"metadata: {nodes[0].metadata}")
        print(f"text: {nodes[0].get_content()[:200]}...\n" + "-"*50)
    
    return nodes


if __name__ == "__main__":
    load_dotenv()
    try:
        nodes = load_and_chunk_documents(DATA_DIR)
    except Exception as e:
        print(f"error during ingestion: {e}")
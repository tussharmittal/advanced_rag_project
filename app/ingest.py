import os
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def build_index():
    print("1. Loading Document...")
    # Load our complex corporate policy
    documents = SimpleDirectoryReader("./data").load_data()

    print("2. Initializing Local Embedding Model (Zero-Exfiltration)...")
    # Using BGE-small, a highly efficient open-source embedding model 
    # This runs 100% locally. No data goes to OpenAI.
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("3. Building Parent-Child Hierarchy...")
    # This splits the document into a tree: 
    # Root (2048 tokens) -> Middle (512 tokens) -> Leaf (128 tokens)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[256, 128, 64] # Lowered thresholds!
        )
    
    # Parse the documents into the hierarchical node structure
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # Extract ONLY the tiny 128-token leaf nodes for the vector database
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"   Created {len(nodes)} total structural nodes.")
    print(f"   Extracted {len(leaf_nodes)} tiny leaf nodes for precise vector search.")

    print("4. Setting up Vector DB and Document Store...")
    # 4a. Setup ChromaDB to store the vector embeddings of the Leaf nodes
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("advanced_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 4b. Setup a local Document Store to hold the large Parent nodes!
    # Without this, the system wouldn't know what Parent a Child belongs to.
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # Combine them into a single Storage Context
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )

    print("5. Embedding and Indexing...")
    # We pass ONLY the leaf nodes to the vector store index. 
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # Save the document store (which holds our parents) to disk
    storage_context.persist(persist_dir="./storage")
    print("✅ Parent-Child Index built and saved successfully to disk!")

if __name__ == "__main__":
    build_index()
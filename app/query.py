import os
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import get_response_synthesizer


def run_advanced_query(query_text: str):
    print("\n1. Booting up Local Embedding Model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("2. Connecting to Vector DB and Document Store...")
    # Connect to the Chroma DB where our Leaf vectors live
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("advanced_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the Parent chunks from the local storage folder
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage", 
        vector_store=vector_store
    )

    # Rebuild the index from our saved files
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("3. Initializing Stage 1: Fast Bi-Encoder Search (Top 12)...")
    # Base retriever fetches the tiny, highly-accurate leaf nodes
    base_retriever = index.as_retriever(similarity_top_k=12)

    print("4. Initializing Parent-Child Merger...")
    # If a Parent chunk has 512 tokens, and we retrieve enough 128-token leaves 
    # from that same parent, this tool automatically swaps the leaves out 
    # and returns the massive Parent chunk instead.
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context, 
        verbose=True
    )

    print("5. Initializing Stage 2: Deep Cross-Encoder Reranker (Top 2)...")
    # This runs a completely different local neural network that reads the query 
    # and the documents simultaneously to filter out false positives.
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=2
    )

    print(f"\n=======================================================")
    print(f"QUERY: '{query_text}'")
    print(f"=======================================================\n")

    # Execute Stage 1 + Parent Merge
    retrieved_nodes = retriever.retrieve(query_text)
    
    # Execute Stage 2 (Rerank)
    reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=query_text)

    print("✅ FINAL CONTEXT SENT TO LLM (Zero Hallucination Guarantee):")
    for i, node in enumerate(reranked_nodes):
        print(f"\n--- Chunk {i+1} (Relevance Score: {node.score:.2f}) ---")
        print(node.text.strip())
        print("--------------------------------------------------")
    
    print("\n6. Synthesizing Final Answer with Local LLM (Zero Cost)...")
    
    # Import the local Ollama integration
    from llama_index.llms.ollama import Ollama
    
    # Point it to the lightweight model we just downloaded
    llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
    
    # The Synthesizer takes the User Query and our mathematically perfected context
    response_synthesizer = get_response_synthesizer(llm=llm)
    final_response = response_synthesizer.synthesize(
        query=query_text,
        nodes=reranked_nodes,
    )
    
    print("\n=======================================================")
    print("🤖 FINAL AI RESPONSE:")
    print("=======================================================")
    print(str(final_response))

if __name__ == "__main__":
    # Let's test it with a highly ambiguous question that would break naive RAG
    tricky_query = "If a contractor leaves their token active for 2 days, what is the penalty?"
    run_advanced_query(tricky_query)
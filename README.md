# ReguSearch AI: Enterprise Zero-Exfiltration RAG

An advanced, hallucination-resistant Retrieval-Augmented Generation (RAG) pipeline designed for strict enterprise security environments. This system guarantees **zero data exfiltration** by running all embeddings, retrieval, and LLM inference 100% locally on-device.

## 🏗 Architecture & Core Concepts

Standard naive RAG breaks down on dense legal and financial documents by diluting context and relying on shallow vector math. This project solves that using a **Two-Stage Retrieval Architecture**:

1. **Hierarchical Parent-Child Indexing:** Documents are parsed into a tree structure. Tiny "Leaf" nodes (64-128 tokens) are embedded for mathematically precise vector search, while massive "Parent" nodes (1024-2048 tokens) are stored locally. When a Leaf hits, the full Parent is retrieved to preserve unbroken context.
2. **Two-Stage Reranking:** - *Stage 1 (The Net):* A fast Bi-Encoder vector search pulls the top 12 chunks.
   - *Stage 2 (The Filter):* A deep Cross-Encoder neural network mathematically scores the query against the chunks simultaneously, dropping false positives and returning only the top 2 most relevant contexts.
3. **Zero-Exfiltration Inference:** The sanitized, perfectly retrieved context is passed to a locally hosted open-weight LLM (Llama 3.2 via Ollama) to synthesize the final answer.

## 🛠 Tech Stack

* **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Persistent Local Storage)
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (via HuggingFace)
* **Cross-Encoder Reranker:** `ms-marco-MiniLM-L-6-v2` (via SentenceTransformers)
* **Inference Engine:** [Ollama](https://ollama.com/) (running `llama3.2:1b` or `llama3.2`)

## 📂 Project Structure

```text
advanced_rag_project/
├── data/
│   └── corporate_policy.txt  # Complex test data with hierarchical rules
├── app/
│   ├── ingest.py             # Builds the Parent-Child index & saves to disk
│   └── query.py              # Executes the Two-Stage Retrieval & LLM synthesis
├── chroma_db/                # Auto-generated Vector DB (Leaf nodes)
├── storage/                  # Auto-generated Document Store (Parent nodes)
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Installation & Setup

**1. Clone the repository and set up the environment:**
```bash
git clone <your-repo-url>
cd advanced_rag_project
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Install the local inference engine:**
* Download and install [Ollama](https://ollama.com/).
* Pull the local reasoning model (requires ~2GB disk space):
```bash
ollama pull llama3.2:1b
```

## 💻 Usage

**1. Build the Index (Run Once)**
This script reads the raw documents, creates the hierarchical nodes, embeds the leaves, and saves everything to the local `chroma_db` and `storage` folders.
```bash
python app/ingest.py
```

**2. Query the System**
This script loads the local databases, runs the two-stage cross-encoder retrieval, and streams the context to the local Llama model for synthesis.
```bash
python app/query.py
```
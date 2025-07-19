# Hybrid Search Engine using LangChain, Pinecone, and HuggingFace

A practical example and project template demonstrating **hybrid semantic + keyword search** combining dense embeddings (via HuggingFace/transformers) and sparse retrieval (BM25) over a Pinecone vector database. Utilizes `langchain` and related ecosystem components for powerful, modular search workflows.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Quickstart](#quickstart)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage Example](#usage-example)
  - [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
  - [Working with BM25 and Dense Embeddings](#working-with-bm25-and-dense-embeddings)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [License](#license)

## Project Overview

This project sets up a **hybrid search engine** in Python that leverages:

- **Dense semantic search** (HuggingFace Sentence Transformers)
- **Sparse keyword search** (BM25, via Pinecone)
- **Pinecone** as a scalable, serverless vector database
- **LangChain** for orchestrating the retrieval process

The approach enables retrieval-augmented generation (RAG) and robust document search that is both *semantically aware* and *sensitive to keywords*, closing the gap between pure embedding-based retrieval and classic lexical search.

## Key Features

- **Hybrid Retrieval:** Combines BM25 (TF-IDF-based) and transformer embeddings for best-of-both-worlds search quality.
- **Pinecone Integration:** Serverless vector DB for scalable, fast vector and sparse data search.
- **Embeddings Flexibility:** Swap between HuggingFace, OpenAI, etc.
- **Simple API:** Integrate or extend with your own data pipelines.
- **Documented Example Code:** See end-to-end code snippets for ingestion, encoding, and querying.

## Quickstart

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sabale-37/Hybrid-Search-Engine-using-Langchain.git
cd Hybrid-Search-Engine-using-Langchain
pip install -r requirements.txt
```

### Environment Setup

1. **Set your Pinecone API Key:**
   - You can hard-code, pass as argument, or load from a `.env` file.
2. **(Optional) HuggingFace Access:**
   - If using a custom/token-protected HuggingFace model, ensure `HF_TOKEN` is set in your environment or `.env`.

Example `.env` contents:

```
PINECONE_API_KEY=your-pinecone-api-key-here
HF_TOKEN=your-huggingface-token-here
```

## Usage Example

### Hybrid Retrieval Pipeline

Below is a minimal end-to-end workflow. See the included notebook for further details and code comments.

```python
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder

# Load env vars
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
hf_token = os.getenv("HF_TOKEN")
index_name = "hybrid-search-langchain-pinecone"

# Init Pinecone client & create index if necessary
pc = Pinecone(api_key=api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Embeddings/Encoder setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Example corpus
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited Tokyo",
]

# Fit/test sparse encoder
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Set up hybrid retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index
)
retriever.add_texts(sentences)
results = retriever.invoke("what city did I visit most recent")
print(results)
```

### Working with BM25 and Dense Embeddings

- **BM25Encoder** is used for sparse/TF-IDF scoring.
- **HuggingFaceEmbeddings** provides semantic encodings.
- Both representations are indexed in Pinecone; queries are compared against both and results are fused/ranked.

## Configuration

Main configurable options:

| Option             | Description                                      |
|--------------------|--------------------------------------------------|
| `api_key`          | Your Pinecone API key                            |
| `hf_token`         | HuggingFace authentication token (if required)   |
| `index_name`       | Pinecone DB index name                           |
| `dimension`        | Embedding vector size (384 for `all-MiniLM-L6-v2`)|
| `metric`           | Pinecone scoring metric (`dotproduct` recommended for hybrid) |
| Encoder choice     | BM25 (sparse), or SPLADE                          |
| Embeddings model   | HuggingFace Transformers, OpenAI, etc.           |

## Requirements

```text
pinecone-client
langchain-openai
python-dotenv
langchain_community
langchain_huggingface
pinecone
sentence-transformers
ipykernel
```

Install using:

```bash
pip install -r requirements.txt
```

## Troubleshooting & Tips

- **Cache/Symlink Warnings (Windows):** If you see HuggingFace symlink warnings, consider enabling Developer Mode on Windows or ignore the warnings if space is not a concern.
- **`ipykernel` Usage:** Required for running in Jupyter notebooks.
- **BM25Encoder Data:** Always fit BM25 on your document corpus and persist encoder state for repeatability.
- **Updating pip:** See pip notices for upgrading; generally not required for core functionality.
- **Further Performance:** For larger corpora or production use, explore batching, custom tokenization, and advanced retriever parameters.

## License

MIT License (see `LICENSE` for details).

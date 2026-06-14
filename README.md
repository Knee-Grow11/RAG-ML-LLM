# Local RAG System: ML Engineer Assessment

## Overview
[cite_start]This repository contains a fully functional, 100% local Question-Answering system built to answer queries over a provided document corpus[cite: 3, 19]. It uses a Retrieval-Augmented Generation (RAG) architecture powered by LangChain, executed entirely on local hardware without the need for external API keys.

[cite_start]The system ingests data from a local directory [cite: 20][cite_start], chunks it, generates vector embeddings, stores them in a local database, and utilizes a local Large Language Model (LLM) to generate grounded answers[cite: 21, 12].

## Key Design Choices & Tradeoffs
1. **Ollama for Local Inference:** Chosen for its ease of use in managing local open-source weights. Using `llama3` for text generation and `nomic-embed-text` for embeddings ensures zero data leakage, zero API costs, and robust local performance.
2. **ChromaDB:** A lightweight, open-source vector database that persists easily to disk. [cite_start]It avoids the overhead of running a separate Docker container for a database like Qdrant or Milvus, perfectly fitting the 6-8 hour scope of this project[cite: 35].
3. **Chunking Strategy:** `RecursiveCharacterTextSplitter` is used with a chunk size of 1000 and an overlap of 200. 
    * *Tradeoff:* Larger chunks provide better context but risk exceeding the local LLM's context window and slowing down inference. This ratio ensures semantic boundaries are largely respected without overwhelming the prompt.
4. [cite_start]**Strict Prompt Engineering:** The system is explicitly prompted to *only* use retrieved context to prevent hallucination, ensuring correctness and grounding of answers[cite: 12].

## Limitations & Known Issues
* **Hardware Constraints:** Inference speed and maximum context window are strictly bound by the host machine's CPU/GPU and RAM. Running Llama 3 locally requires at least 8GB of RAM.
* **Retrieval Quality:** Similarity search relies on dense vector embeddings. It may struggle with exact keyword matching (e.g., specific serial numbers). [cite_start]A future iteration could implement a hybrid search (Dense + BM25 sparse retrieval)[cite: 13, 48].

## How to Run the Code

### 1. Setup Ollama
1. Download and install [Ollama](https://ollama.com/).
2. Pull the required models via your terminal:
   ```bash
   ollama run llama3
   ollama pull nomic-embed-text
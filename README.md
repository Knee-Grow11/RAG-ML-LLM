# Local RAG System - ML Engineer Assessment

This is my submission for the ML Engineer assessment. I decided to build the whole Question-Answering system using a 100% local, open-source stack. No OpenAI API keys and no cloud vector databases. Just my machine. 

I went with LangChain for the orchestration, Ollama (Llama 3) for text generation, HuggingFace (`all-MiniLM-L6-v2`) for embeddings, and ChromaDB to store the vectors locally on disk.

## A Quick Note on Tradeoffs & Architecture
The prompt suggested a massive Kaggle dataset of AI governance documents. Initially, I tried routing the embedding generation through Ollama as well, but hitting the local API for 11,000+ document chunks caused a massive CPU bottleneck. 

Since the instructions mentioned evaluating how I reason about tradeoffs, I made a change and swapped the embedding model to HuggingFace's `all-MiniLM-L6-v2` running directly in-process via the `sentence-transformers` library. This eliminated the HTTP overhead and optimized the CPU usage which allowed my machine to comfortably ingest the entire document corpus without needing a massive dedicated GPU. 

### Getting it Running

First off, Ollama has to be installed locally. You can grab it here: https://ollama.com/download/windows. Once that is running in the background, download the LLM I used for generation.

Open a terminal and run this:

ollama pull llama3

After that, set up the Python environment. 
Clone this repository, navigate into the folder, and install the dependencies:

pip install -r requirements.txt

For the data:

You need to create a folder named `data` right in the main project directory. Drop the text files from the Kaggle dataset in there.

To crunch the documents and build the database, run:
python ingest.py

It will read the files, chunk them up, embed them natively, and spit out a new folder called `chroma_db`. 

After that, it will be ready for the query process

Example Question:
python query.py "What are the main principles of AI Governance according to the documents?"

**Sample output:**

=================== ANSWER ===================

Based on the provided context, the main principles of AI governance mentioned are:

1. Artificial intelligence (AI) to promote growth, sustainable development, and Israeli leadership in innovation.
2. Human-centric AI: The development and use of an AI system should respect the rule of law, fundamental rights, and public interests, and preserve human dignity and the right to privacy.

These principles are mentioned under the "Governance Principles for a New Generation of Artificial Intelligence" section and are intended to reflect elements that should be considered in the development and use of AI.

=================== SOURCES ==================
- data\1076.txt (Page: N/A) (Distance Score: 0.5321)
- data\35.txt (Page: N/A) (Distance Score: 0.5375)
- data\1032.txt (Page: N/A) (Distance Score: 0.5506)

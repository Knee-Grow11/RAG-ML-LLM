import argparse
import sys
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

chroma_dir = "./chroma_db"
embedding_model = "nomic-embed-text"
llm_model = "llama3" 

prompt_template_string = """
You are an expert, helpful AI assistant. Use the following pieces of retrieved context to answer the question. 
If the answer is not contained within the context, politely state that you do not know based on the provided documents. 
Do not hallucinate or use outside knowledge.

Context:
{context}

Question: {question}

Answer:
"""

def main():
    parser = argparse.ArgumentParser(description="Query the local RAG system.")
    parser.add_argument("query", type=str, help="The question you want to ask the document corpus.")
    args = parser.parse_args()

    if not os.path.exists(chroma_dir):
        print("Error: Vector database not found. Please run 'python ingest.py' first.")
        sys.exit(1)

    print(f"Querying local database for: '{args.query}'...\n")

    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    results = vector_store.similarity_search_with_score(args.query, k=4)
    
    if not results:
        print("No relevant context found in the database.")
        sys.exit(0)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])

    prompt_template = PromptTemplate(
        template=prompt_template_string,
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context_text, question=args.query)

    llm = Ollama(model=llm_model)
    response = llm.invoke(prompt)

    print("=================== ANSWER ===================")
    print(response)
    print("\n=================== SOURCES ==================")
    
    unique_sources = set()
    for doc, score in results:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        
        source_id = f"{source} (Page: {page})"
        if source_id not in unique_sources:
            unique_sources.add(source_id)
            print(f"- {source_id} (Distance Score: {score:.4f})")

if __name__ == "__main__":
    main()
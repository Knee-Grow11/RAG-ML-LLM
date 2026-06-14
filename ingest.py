import os
import sys
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

data_dir = "./data"
chroma_dir = "./chroma_db"
embedding_model = "nomic-embed-text"

def main():
    print("Starting local data ingestion pipeline...")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directory '{data_dir}' created. Please add PDF or TXT files and re-run.")
        sys.exit(1)

    print(f"Scanning '{data_dir}' for documents...")
    pdf_loader = PyPDFDirectoryLoader(data_dir)
    txt_loader = DirectoryLoader(
        data_dir, 
        glob="**/*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': False},
        silent_errors=True
    )
    
    documents = pdf_loader.load() + txt_loader.load()

    if not documents:
        print(f"No documents found in '{data_dir}'. Please add files and try again.")
        sys.exit(1)
        
    print(f"Loaded {len(documents)} document pages/files.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print(f"Initializing Ollama embeddings ({embedding_model}) and ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(chroma_dir):
        print("Clearing existing vector database...")
        shutil.rmtree(chroma_dir)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir
    )
    
    vector_store.persist()
    print(f"Successfully ingested and embedded data into '{chroma_dir}'.")

if __name__ == "__main__":
    main()

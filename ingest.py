# ingest.py
import os
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
EBOOK_FOLDER = "ebooks"
DB_DIR = "./db"

def load_documents():
    """Load all PDF and EPUB files from the ebook folder."""
    documents = []
    for file in os.listdir(EBOOK_FOLDER):
        file_path = os.path.join(EBOOK_FOLDER, file)
        
        if file.endswith('.pdf'):
            print(f"Loading PDF: {file}")
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.epub'):
            print(f"Loading EPUB: {file}")
            loader = UnstructuredEPubLoader(file_path)
            documents.extend(loader.load())
            
    print(f"Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    """Create and persist the vector store."""
    print("Creating embedding model...")
    # This will download the model (approx. 90MB) on first run
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use CPU
    )
    
    print("Creating and persisting vector database...")
    # This creates the 'db' folder and stores the vectors
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=DB_DIR
    )
    print(f"Vector database created and saved to {DB_DIR}")

def main():
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        create_vector_store(chunks)
    else:
        print("No PDF or EPUB documents found in the 'my_ebooks' folder.")

if __name__ == "__main__":
    main()

import os
import re
import io
import json
import unicodedata
import shutil
from contextlib import redirect_stderr
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

EBOOK_FOLDER = "ebooks"
DB_DIR = "./db"
SETTINGS_FILE = "settings.json"

# Load Settings
try:
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)
    EMBEDDING_MODEL = settings.get("embedding_model", "all-mpnet-base-v2")
    print(f"Loaded settings: EMBEDDING_MODEL={EMBEDDING_MODEL}")
except FileNotFoundError:
    print(f"Warning: {SETTINGS_FILE} not found. Using default embedding model.")
    EMBEDDING_MODEL = "all-mpnet-base-v2"
except json.JSONDecodeError:
    print(f"Warning: Error decoding {SETTINGS_FILE}. Using default embedding model.")
    EMBEDDING_MODEL = "all-mpnet-base-v2"

class StderrFilter(io.StringIO):
    """
    Custom stream handler to filter specific warnings from stderr during EPUB processing.
    """
    def write(self, s: str) -> int:
        """
        Writes to the stream, skipping lines containing TeX math conversion warnings.
        """
        if "[WARNING] Could not convert TeX math" in s:
            # Signal that we "consumed" the string by returning its length
            return len(s)
        # Call the parent's write method and return its result (the number of chars written)
        return super().write(s)

def clean_text_ultra_aggressive(text: str) -> str:
    """
    Performs aggressive cleaning on input text, removing control characters and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize('NFKD', text)

    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(('L', 'N', 'P', 'S', 'M', 'Z')) or char in ('\n', '\r', '\t', ' '):
            if not (category == 'Cc' and char not in ('\n', '\r', '\t')):
                cleaned_chars.append(char)
        elif char in ('\x07', '\x08', '\ufffd'):
            pass

    cleaned = "".join(cleaned_chars)

    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def load_documents():
    """
    Loads and cleans text content from PDF and EPUB files in the EBOOK_FOLDER.
    """
    documents = []
    # Check if folder exists before listing directory contents
    if not os.path.exists(EBOOK_FOLDER):
        print(f"Ebook folder '{EBOOK_FOLDER}' not found. Please create it and add documents.")
        return documents # Return empty list if folder doesn't exist

    print(f"Loading documents from '{EBOOK_FOLDER}'...")
    for file in os.listdir(EBOOK_FOLDER):
        file_path = os.path.join(EBOOK_FOLDER, file)

        try:
            if file.endswith('.pdf'):
                print(f"Loading PDF: {file}")
                loader = PyMuPDFLoader(file_path)
                loaded_docs = loader.load()
            elif file.endswith('.epub'):
                print(f"Loading EPUB: {file}")
                loader = UnstructuredEPubLoader(file_path, mode="elements", strategy="fast")
                stderr_filter = StderrFilter()
                with redirect_stderr(stderr_filter):
                    loaded_docs = loader.load()
            else:
                loaded_docs = []

            cleaned_docs = []
            for doc in loaded_docs:
                if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                    doc.page_content = clean_text_ultra_aggressive(doc.page_content)
                    if doc.page_content:
                        doc.metadata["source"] = file_path
                        cleaned_docs.append(doc)
                else:
                    print(f"Skipping document part in {file} due to missing/invalid page_content.")

            documents.extend(cleaned_docs)

        except Exception as e:
            print(f"Error loading or cleaning file {file}: {e}")

    print(f"Total documents loaded and cleaned: {len(documents)}")
    return documents

def split_documents(documents):
    """
    Splits loaded documents into smaller text chunks suitable for embedding.
    """
    print("Splitting documents into chunks...")
    if not documents:
        print("No valid documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_chunks = []
    for i, doc in enumerate(documents):
        try:
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                chunks = text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            else:
                source_info = doc.metadata.get('source', f'document index {i}') if hasattr(doc, 'metadata') else f'document index {i}'
                print(f"Skipping splitting document from {source_info} due to invalid page_content type: {type(getattr(doc, 'page_content', None))}")
        except Exception as e:
            source_info = doc.metadata.get('source', f'document index {i}') if hasattr(doc, 'metadata') else f'document index {i}'
            print(f"Error splitting document from {source_info}: {e}")


    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def create_vector_store(chunks):
    """
    Deletes any existing vector store, generates embeddings for text chunks using the model
    specified in settings, and persists them in a new Chroma vector database.
    """
    if os.path.exists(DB_DIR):
        print(f"Removing existing vector store at {DB_DIR}")
        shutil.rmtree(DB_DIR)

    if not chunks:
        print("No chunks to add to the vector store.")
        return

    print("Filtering complex metadata from chunks...")
    try:
        filtered_chunks = filter_complex_metadata(chunks)
        print(f"Chunks remaining after filtering: {len(filtered_chunks)}")
    except Exception as e:
        print(f"Error during metadata filtering: {e}. Skipping filtering.")
        filtered_chunks = chunks


    if not filtered_chunks:
        print("No chunks remaining after filtering metadata.")
        return

    valid_chunks = [chunk for chunk in filtered_chunks if hasattr(chunk, 'page_content') and chunk.page_content]
    if not valid_chunks:
        print("No valid chunks with content remaining after filtering.")
        return
    print(f"Valid chunks with content: {len(valid_chunks)}")


    print(f"Creating embedding model ({EMBEDDING_MODEL})...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )

    print("Creating and persisting vector database...")
    try:
        vectorstore = Chroma.from_documents(
            documents=valid_chunks,
            embedding=embedding_function,
            persist_directory=DB_DIR
        )
        print(f"Vector database created and saved to {DB_DIR}")
    except Exception as e:
        print(f"Error creating vector store: {e}")


def main():
    """
    Orchestrates the document loading, splitting, and vector store creation process.
    Ensures the ebook folder exists.
    """
    
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        if chunks:
            create_vector_store(chunks)
        else:
            print("Splitting resulted in no valid chunks.")
    elif os.path.exists(EBOOK_FOLDER) and not os.listdir(EBOOK_FOLDER):
        print(f"The '{EBOOK_FOLDER}' folder is empty. Please add PDF or EPUB files to process.")

if __name__ == "__main__":
    main()

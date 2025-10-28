import os
import re
import io
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

class StderrFilter(io.StringIO):
    """
    Custom stream handler to filter specific warnings from stderr during EPUB processing.
    """
    def write(self, s):
        if "[WARNING] Could not convert TeX math" in s:
            return
        super().write(s)

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
    if not os.path.exists(EBOOK_FOLDER):
        print(f"Error: Ebook folder '{EBOOK_FOLDER}' not found.")
        return documents

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
                        # Add source metadata consistently
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
            # Ensure page_content is string before splitting
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                # Pass metadata along during splitting
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
    Deletes any existing vector store, generates embeddings for text chunks,
    and persists them in a new Chroma vector database.
    """
    # Delete existing database if exists
    if os.path.exists(DB_DIR):
        print(f"Removing existing vector store at {DB_DIR}")
        shutil.rmtree(DB_DIR)

    if not chunks:
        print("No chunks to add to the vector store.")
        return

    print("Filtering complex metadata from chunks...")
    try:
        # Filter metadata IN PLACE if possible, or create new list
        filtered_chunks = filter_complex_metadata(chunks)
        print(f"Chunks remaining after filtering: {len(filtered_chunks)}")
    except Exception as e:
        print(f"Error during metadata filtering: {e}. Skipping filtering.")
        filtered_chunks = chunks # Proceed without filtering if it causes error


    if not filtered_chunks:
        print("No chunks remaining after filtering metadata.")
        return

    # Check if chunks have content after filtering
    valid_chunks = [chunk for chunk in filtered_chunks if hasattr(chunk, 'page_content') and chunk.page_content]
    if not valid_chunks:
        print("No valid chunks with content remaining after filtering.")
        return
    print(f"Valid chunks with content: {len(valid_chunks)}")


    print("Creating embedding model...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )

    print("Creating and persisting vector database...")
    try:
        vectorstore = Chroma.from_documents(
            documents=valid_chunks, # Use valid chunks
            embedding=embedding_function,
            persist_directory=DB_DIR
        )
        print(f"Vector database created and saved to {DB_DIR}")
    except Exception as e:
        print(f"Error creating vector store: {e}")


def main():
    """
    Orchestrates the document loading, splitting, and vector store creation process.
    """
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        # Ensure chunks are valid before creating store
        if chunks:
             create_vector_store(chunks)
        else:
             print("Splitting resulted in no valid chunks.")
    else:
        print("No processable PDF or EPUB documents found or loaded successfully.")

if __name__ == "__main__":
    main()


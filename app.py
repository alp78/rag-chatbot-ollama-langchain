import streamlit as st
import os
import json # <-- Import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM # <-- Use correct Ollama import
from langchain_chroma import Chroma # <-- Use correct Chroma import

DB_DIR = "./db"
SETTINGS_FILE = "settings.json"

# Load Settings
try:
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)
    # Load settings with defaults if keys are missing
    EMBEDDING_MODEL = settings.get("embedding_model", "all-MiniLM-L6-v2")
    LLM_MODEL = settings.get("llm_model", "gemma2:9b")
    SEARCH_K = settings.get("search_k", 60)
    SEARCH_FETCH_K = settings.get("search_fetch_k", 100)
    # Load prompt template - provide a sensible default
    DEFAULT_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Synthesize an answer based on the information found in the context.
If the context doesn't directly state the answer, explain what the context *does* say
and infer a possible answer based on that information.
Combine information from multiple sources if necessary.
If the context provides no relevant clues at all, indicate that the information
could not be constructed from the provided documents."""
    PROMPT_TEMPLATE_STR = settings.get("prompt_template", DEFAULT_PROMPT)

    print(f"Loaded settings: EMBEDDING_MODEL={EMBEDDING_MODEL}, LLM_MODEL={LLM_MODEL}, K={SEARCH_K}, FETCH_K={SEARCH_FETCH_K}")

except FileNotFoundError:
    print(f"Warning: {SETTINGS_FILE} not found. Using default settings.")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemma2:9b"
    SEARCH_K = 60
    SEARCH_FETCH_K = 100
    PROMPT_TEMPLATE_STR = DEFAULT_PROMPT
except json.JSONDecodeError:
    print(f"Warning: Error decoding {SETTINGS_FILE}. Using default settings.")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemma2:9b"
    SEARCH_K = 60
    SEARCH_FETCH_K = 100
    PROMPT_TEMPLATE_STR = DEFAULT_PROMPT


def format_docs(docs: list[Document]) -> str:
    """Convert Documents to a single string."""
    # Joins the page_content of each document with two newlines
    return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))

# Load components (cached for performance)
@st.cache_resource
def get_embedding_function():
    """Load the embedding model specified in settings."""
    # Initializes HuggingFace embeddings using the model name from settings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, # Use loaded setting
        model_kwargs={'device': 'cuda'} # Specify GPU usage
    )

@st.cache_resource
def get_vectorstore(_embedding_function):
    """Load the existing Chroma vector database."""
    # Initializes Chroma vector store from the persisted directory
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=_embedding_function
    )

@st.cache_resource
def get_llm():
    """Load the local LLM specified in settings."""
    # Initializes the Ollama LLM using the model name from settings
    return OllamaLLM(model=LLM_MODEL) # Use loaded setting

@st.cache_resource
def get_rag_chain(_vectorstore, _llm):
    """Create the RAG chain using manual LCEL construction and settings."""

    # Configures the retriever using MMR and K/Fetch_K from settings
    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": SEARCH_K,        # Use loaded setting
            "fetch_k": SEARCH_FETCH_K # Use loaded setting
        }
    )

    # Use loaded prompt template string from settings
    # Appends the standard RAG context/question/answer structure
    full_prompt_template_str = PROMPT_TEMPLATE_STR + """

Context:
{context}

Question:
{input}

Answer:"""
    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)

    # Manually Construct RAG Chain using LCEL
    # 1. RunnablePassthrough.assign: Formats retrieved docs into context string
    # 2. prompt: Inserts context and input into the prompt template
    # 3. _llm: Sends the formatted prompt to the Ollama LLM
    # 4. StrOutputParser: Extracts the string response from the LLM output
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | _llm
        | StrOutputParser()
    )

    # Orchestrates the RAG flow using RunnableParallel and assign
    # 1. RunnableParallel: Runs retriever and input passthrough concurrently
    # 2. assign(answer=...): Takes the output (context & input) and passes it
    #    to rag_chain_from_docs to generate the final answer, adding it to the dict.
    final_chain = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return final_chain

# --- Streamlit App UI ---
st.title("Chat with Your Ebooks")
st.write(f"Powered by local {LLM_MODEL} and your documents.") # Display loaded LLM model

try:
    # Initialize all necessary components
    embedding_function = get_embedding_function()
    vectorstore = get_vectorstore(embedding_function)
    llm = get_llm()
    rag_chain = get_rag_chain(vectorstore, llm)

    # Initialize chat history in Streamlit session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input from chat interface
    if user_question := st.chat_input("Ask a question about your ebooks..."):
        # Add user's question to history and display it
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Invoke the RAG chain with the user's question
                # The input to RunnableParallel needs to be a dictionary if the inner
                # runnables expect dictionary keys (like retriever which implicitly uses 'input')
                # However, our final_chain's input is just the question string passed via RunnablePassthrough()
                response = rag_chain.invoke(user_question) # Pass the string directly
                
                # Safely get answer and context, providing defaults if missing
                answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                context_docs = response.get("context", [])
                st.markdown(answer)

                # Display retrieved source documents in an expander
                with st.expander("Show relevant sources"):
                    if context_docs:
                        for i, doc in enumerate(context_docs):
                            # Safely access metadata
                            source = doc.metadata.get("source", "Unknown Source") if hasattr(doc, 'metadata') else "Unknown Source"
                            page = doc.metadata.get("page", None) if hasattr(doc, 'metadata') else None
                            filename = os.path.basename(source) # Extract filename

                            # Format header with source and page (if available)
                            ref_header = f"Source {i+1}: {filename}"
                            if page is not None:
                                # Ensure page is treated as integer before adding 1
                                try:
                                    ref_header += f" (Page {int(page) + 1})"
                                except (ValueError, TypeError):
                                     ref_header += f" (Page {page})" # Display raw if not int

                            st.subheader(ref_header)
                            st.write(doc.page_content if hasattr(doc, 'page_content') else "No content available.")
                            st.divider()
                    else:
                        st.write("No sources retrieved.")

        # Add assistant's answer to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Handle potential errors during app execution
except FileNotFoundError:
    st.error(
        f"Vector database not found at {DB_DIR}. "
        f"Did you run `python ingest.py` first?"
    )
except ImportError as e:
    st.error(f"Import Error: {e}. Have you installed all required packages?")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.exception(e) # Show detailed traceback in terminal/logs


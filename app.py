import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma


DB_DIR = "./db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma2:9b"  # <-- Make sure this matches the model you pulled


def format_docs(docs: list[Document]) -> str:
    """Convert Documents to a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Load components (cached for performance)
@st.cache_resource
def get_embedding_function():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'} # Use GPU if available
    )

@st.cache_resource
def get_vectorstore(_embedding_function):
    """Load the existing vector database."""
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=_embedding_function
    )

@st.cache_resource
def get_llm():
    """Load the local LLM."""
    return OllamaLLM(model=LLM_MODEL)

@st.cache_resource
def get_rag_chain(_vectorstore, _llm):
    """Create the RAG chain using manual LCEL construction."""

    # Use MMR and Increased K for Retrieval
    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 60,
            "fetch_k": 100
        }
    )

    # Synthesis Prompt
    prompt_template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Synthesize an answer based on the information found in the context.
    If the context doesn't directly state the answer, explain what the context *does* say
    and infer a possible answer based on that information.
    Combine information from multiple sources if necessary.
    If the context provides no relevant clues at all, indicate that the information
    could not be constructed from the provided documents.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Manually Construct RAG Chain
    # Takes question, gets context, formats docs, runs LLM, parses output
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | _llm
        | StrOutputParser()
    )

    # Takes input, gets context using retriever, passes input+context to chain above
    # Returns dictionary with input, context, and answer
    final_chain = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return final_chain

# --- Streamlit App UI ---
st.title("Chat with Your Ebooks")
st.write(f"Powered by local {LLM_MODEL} and your documents.")

try:
    # Initialize all components
    embedding_function = get_embedding_function()
    vectorstore = get_vectorstore(embedding_function)
    llm = get_llm()
    rag_chain = get_rag_chain(vectorstore, llm)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_question := st.chat_input("Ask a question about your ebooks..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass only the question string to the final chain
                response = rag_chain.invoke(user_question)
                answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                context_docs = response.get("context", [])
                st.markdown(answer)

                # Show the source documents with metadata
                with st.expander("Show relevant sources"):
                    if context_docs:
                        for i, doc in enumerate(context_docs):
                            source = doc.metadata.get("source", "Unknown Source")
                            page = doc.metadata.get("page", None)
                            filename = os.path.basename(source) # Get just the filename

                            ref_header = f"Source {i+1}: {filename}"
                            if page is not None:
                                ref_header += f" (Page {page + 1})" # Add 1 for human-readable page number

                            st.subheader(ref_header)
                            st.write(doc.page_content)
                            st.divider()
                    else:
                        st.write("No sources retrieved.")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

except FileNotFoundError:
    st.error(
        f"Vector database not found at {DB_DIR}. "
        f"Did you run `python ingest.py` first?"
    )
except ImportError as e:
    st.error(f"Import Error: {e}. Have you installed all required packages (langchain, langchain-community, langchain-core, langchain-huggingface, langchain-chroma, streamlit, ollama)?")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.exception(e)

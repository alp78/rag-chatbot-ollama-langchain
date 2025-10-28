import streamlit as st
from langchain_ollama import OllamaLLM as Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os

# Configuration
DB_DIR = "./db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma2:9b"

# Load components (cached for performance)
@st.cache_resource
def get_embedding_function():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
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
    return Ollama(model=LLM_MODEL)

# Helper function to format documents
def format_docs(docs: list[Document]) -> str:
    """Concatenate page_content of documents separated by double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_rag_chain(_vectorstore, _llm):
    """Create the RAG chain manually using LCEL."""
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

    prompt_template = """
    You are an assistant for question-answering tasks based on a collection of ebooks about [TOPIC].
    Use the following pieces of retrieved context from the ebooks to answer the question accurately and concisely.
    If the context doesn't contain the answer, state that the information wasn't found in the provided sources.
    Answer based *only* on the provided context.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | _llm
        | StrOutputParser()
    )

    final_chain = (
        RunnablePassthrough.assign(
            context= lambda x: retriever.invoke(x["input"])
        )
        | RunnablePassthrough.assign(
            answer=rag_chain_from_docs
        )
    )

    return final_chain

# Streamlit App UI
st.title("Chat with Your [TOPIC] Ebooks 📚")
st.write(f"Powered by local {LLM_MODEL}")

try:
    embedding_function = get_embedding_function()
    vectorstore = get_vectorstore(embedding_function)
    llm = get_llm()
    rag_chain = get_rag_chain(vectorstore, llm)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about your [TOPIC] ebooks..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {LLM_MODEL}... (This may take a moment on CPU)"):
                response = rag_chain.invoke({"input": user_question})
                answer = response.get("answer", "Sorry, I couldn't generate an answer.")
                context_docs = response.get("context", [])

                st.markdown(answer)

                # Source display
                with st.expander("Show relevant sources"):
                    if not context_docs:
                        st.write("No sources found for this query.")
                    else:
                        st.markdown("---")
                        for i, doc in enumerate(context_docs):
                            source_path = doc.metadata.get("source", "Unknown Source")
                            source_name = os.path.basename(source_path) 
                            page = doc.metadata.get("page", None)
                            
                            reference = f"**Source {i+1}:** `{source_name}`"
                            if page is not None:
                                reference += f" (Page {page + 1})"

                            st.markdown(reference)
                            st.markdown(f"> {doc.page_content}")
                            st.markdown("---") 


        st.session_state.messages.append({"role": "assistant", "content": answer})

except FileNotFoundError:
    st.error(
        f"Vector database not found at {DB_DIR}. "
        f"Did you run `python ingest.py` first?"
    )
except Exception as e:
    st.error(f"An error occurred while processing your question: {e}")

import streamlit as st
import os
import json
import uuid
from datetime import datetime
import time
from streamlit_local_storage import LocalStorage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

# --- Configuration ---
DB_DIR = "./db"
SETTINGS_FILE = "settings.json"
LOCAL_STORAGE_KEY_CHATS = "rag_chat_histories"

# --- Page Config (Set early) ---
st.set_page_config(layout="wide")

# --- Load Settings ---
try:
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)
    EMBEDDING_MODEL = settings.get("embedding_model", "all-MiniLM-L6-v2")
    LLM_MODEL = settings.get("llm_model", "gemma2:9b")
    SEARCH_K = settings.get("search_k", 60)
    SEARCH_FETCH_K = settings.get("search_fetch_k", 100)
    SEARCH_TYPE = settings.get("search_type", "similarity") 
    DEFAULT_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Synthesize a comprehensive, exhaustive answer. Combine information from all relevant sources to provide a detailed explanation.If the question asks for definitions, properties, or parameters, extract and list them clearly.Do not state that the context is insufficient. Use only the provided context to construct your answer."""
    PROMPT_TEMPLATE_STR = settings.get("prompt_template", DEFAULT_PROMPT)
except FileNotFoundError:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"; LLM_MODEL = "gemma2:9b"; SEARCH_K = 60; SEARCH_FETCH_K = 100; SEARCH_TYPE = "similarity"; PROMPT_TEMPLATE_STR = DEFAULT_PROMPT
except json.JSONDecodeError:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"; LLM_MODEL = "gemma2:9b"; SEARCH_K = 60; SEARCH_FETCH_K = 100; SEARCH_TYPE = "similarity"; PROMPT_TEMPLATE_STR = DEFAULT_PROMPT


# --- Local Storage Initialization ---
storage = LocalStorage()

# --- Initialize or Load State (Do this early) ---
if "state_initialized" not in st.session_state:
    loaded_chats = []
    try:
        stored_data = storage.getItem(LOCAL_STORAGE_KEY_CHATS)
        if isinstance(stored_data, list):
            loaded_chats = [
                chat for chat in stored_data
                if isinstance(chat, dict) and 'id' in chat and 'messages' in chat and isinstance(chat['messages'], list)
            ]
    except Exception as e:
        pass 

    st.session_state.all_chats = loaded_chats
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.state_initialized = True


# --- Sidebar (MOVED TO TOP) ---
with st.sidebar:
    st.header("Chat History")
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

    st.divider()
    # Ensure all_chats exists in state before sorting
    if "all_chats" not in st.session_state: st.session_state.all_chats = []
    sorted_chats_display = sorted(st.session_state.all_chats, key=lambda c: c.get('timestamp', 0), reverse=True)

    if not sorted_chats_display:
        st.caption("No saved chats yet.")
    else:
        for chat in sorted_chats_display:
            chat_id = chat.get('id')
            title = chat.get('title', 'Untitled')
            timestamp_float = chat.get('timestamp', 0)
            try: timestamp_str = datetime.fromtimestamp(timestamp_float).strftime('%Y-%m-%d %H:%M')
            except Exception: timestamp_str = "Invalid Date"
            is_selected = (chat_id == st.session_state.current_chat_id)

            # --- Modified Button Label and Tooltip ---
            display_title_label = title # Label only title
            full_tooltip = f"{title} ({timestamp_str})" # Tooltip with title + timestamp

            # Use standard columns
            col1, col2 = st.columns([0.85, 0.15]) # Standard ratio, will resize
            with col1:
                # Use default primary/secondary button types
                button_kind = "primary" if is_selected else "secondary"
                if st.button(
                    display_title_label, # Use simple label (title only)
                    key=f"load_{chat_id}",
                    use_container_width=True,
                    type=button_kind, # Use default Streamlit types
                    help=full_tooltip # Add tooltip with full title + timestamp
                ):
                    if chat_id != st.session_state.current_chat_id:
                        st.session_state.messages = chat.get('messages', [])
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
            with col2:
                # Use standard emoji for delete
                if st.button("❌", key=f"del_{chat_id}", help="Delete chat", use_container_width=True):
                    st.session_state.all_chats = [c for c in st.session_state.all_chats if c.get('id') != chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                    try:
                        storage.setItem(LOCAL_STORAGE_KEY_CHATS, st.session_state.all_chats)
                        time.sleep(0.5) # Wait 500ms for storage.setItem to likely complete
                    except Exception as e:
                        st.error("Could not delete chat from browser storage.")
                    st.rerun() # Rerun after save and sleep


# --- Helper Functions ---
def format_docs(docs: list[Document]) -> str:
    """Convert Documents to a single string."""
    return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))

def generate_chat_title(messages: list, llm: OllamaLLM) -> str:
    """Asks the LLM to generate a short title for the chat."""
    if not messages or len(messages) < 2:
        return "New Chat"
    user_q = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
    asst_a = next((msg.get("content", "") for msg in messages if msg.get("role") == "assistant"), "")
    conversation_summary = f"User: {user_q}\nAssistant: {asst_a}".strip()
    if not conversation_summary:
         conversation_summary = messages[0].get("content", "Chat") if messages else "Chat"

    title_prompt = f"""Summarize the following conversation start in 5 words or less to use as a concise chat title:

{conversation_summary.strip()}

Title:"""
    try:
        response = llm.invoke(title_prompt, config={'max_tokens': 20})
        title = response.strip().strip('"').strip("'")
        if not title or "summarize" in title.lower() or len(title) > 50 :
             raise ValueError("LLM failed to provide a concise title.")
        return title
    except Exception as e:
        return user_q[:30] + "..." if user_q else "Chat Summary"


# --- Load components (cached for performance) ---
@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )

@st.cache_resource
def get_vectorstore(_embedding_function):
    if not os.path.exists(DB_DIR):
        st.error(f"Vector database not found at {DB_DIR}. Run `python ingest.py` first.")
        st.stop()
    try:
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=_embedding_function
        )
    except Exception as e:
        st.error(f"Error loading vector database from {DB_DIR}: {e}")
        st.stop()

@st.cache_resource
def get_llm():
    try:
        return OllamaLLM(model=LLM_MODEL)
    except Exception as e:
        st.error(f"Error initializing Ollama LLM model '{LLM_MODEL}': {e}")
        st.error("Is the Ollama server running and the model pulled?")
        st.stop()

@st.cache_resource
def get_rag_chain(_vectorstore, _llm):
    try:
        # --- DYNAMIC SEARCH_KWARGS FIX ---
        search_kwargs = {}
        if SEARCH_TYPE == "similarity":
            search_kwargs = {"k": SEARCH_K} # Similarity only needs k
        elif SEARCH_TYPE == "mmr":
            # MMR needs both k and fetch_k
            search_kwargs = {"k": SEARCH_K, "fetch_k": SEARCH_FETCH_K} 
        
        retriever = _vectorstore.as_retriever(
            search_type=SEARCH_TYPE, 
            search_kwargs=search_kwargs # Use the dynamically created kwargs
        )
    except Exception as e:
        st.error(f"Error creating retriever from vector store: {e}")
        st.stop()

    full_prompt_template_str = PROMPT_TEMPLATE_STR + """

Context:
{context}

Question:
{input}

Answer:"""
    try:
        prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    except Exception as e:
        st.error(f"Error creating prompt template: {e}")
        st.stop()

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | _llm
        | StrOutputParser()
    )
    final_chain = RunnableParallel(
        {"context": retriever, "input": lambda x: x}
    ).assign(answer=rag_chain_from_docs)
    return final_chain

# --- Function to Save/Update Chat ---
def save_or_update_chat(llm_for_title):
    current_messages = st.session_state.get("messages", [])
    current_chat_id = st.session_state.get("current_chat_id")
    all_chats = st.session_state.get("all_chats", [])
    needs_rerun = False
    saved = False # Flag to track if a save operation happened

    if not current_messages: return needs_rerun

    messages_to_save = []
    for msg in current_messages:
        serializable_msg = msg.copy(); serializable_msg.pop("context_simple", None); serializable_msg.pop("context", None)
        messages_to_save.append(serializable_msg)
    if not messages_to_save: return needs_rerun

    if current_chat_id:
        chat_index = next((i for i, chat in enumerate(all_chats) if chat.get('id') == current_chat_id), -1)
        if chat_index != -1:
            if all_chats[chat_index].get('messages') != messages_to_save:
                all_chats[chat_index]['messages'] = messages_to_save; all_chats[chat_index]['timestamp'] = datetime.now().timestamp()
                try:
                    storage.setItem(LOCAL_STORAGE_KEY_CHATS, all_chats); st.session_state.all_chats = all_chats
                    saved = True
                except Exception as e: st.error("Could not update chat in browser storage.")
        else: current_chat_id = None
    if not current_chat_id:
        if len(messages_to_save) >= 2: # Only save if we have at least user + assistant
            try:
                new_title = generate_chat_title(current_messages, llm_for_title)
                new_chat_id = str(uuid.uuid4())
                new_chat = { "id": new_chat_id, "title": new_title, "timestamp": datetime.now().timestamp(), "messages": messages_to_save }
                st.session_state.all_chats.append(new_chat)
                st.session_state.current_chat_id = new_chat_id
                storage.setItem(LOCAL_STORAGE_KEY_CHATS, st.session_state.all_chats);
                st.toast(f"Chat saved as '{new_title}'!");
                needs_rerun = True
                saved = True
            except Exception as e: st.error("Could not save new chat to browser storage.")

    if saved:
        time.sleep(0.5) # Wait 500ms for storage.setItem to likely complete
    return needs_rerun

# --- Main App Logic ---
st.title("Chat with Your Ebooks")

# --- Loading Placeholder (Moved AFTER title) ---
loading_placeholder = st.empty()

# --- Initialize Components with Loading Messages ---
# Wrapped calls in a container under the placeholder
with loading_placeholder.container():
    st.caption("Initializing components...")
    st.caption("Loading embedding model...")
    embedding_function = get_embedding_function()
    st.caption("Loading vector database...")
    vectorstore = get_vectorstore(embedding_function)
    st.caption(f"Loading LLM ({LLM_MODEL})...")
    llm = get_llm()
    st.caption("Creating RAG chain...")
    rag_chain = get_rag_chain(vectorstore, llm)
# Clear placeholder once components are loaded
loading_placeholder.empty()


# --- Main Chat Interface ---
# Display messages currently in session state
chat_container = st.container()
with chat_container:
    current_display_messages = st.session_state.get("messages", [])
    if isinstance(current_display_messages, list):
        for i, message in enumerate(current_display_messages):
            if isinstance(message, dict) and "role" in message:
                with st.chat_message(message["role"]):
                    st.markdown(message.get("content", ""))
                    # Display context for assistant messages if available
                    if message["role"] == "assistant" and "context_simple" in message:
                        with st.expander("Show relevant sources"):
                            simple_context = message["context_simple"]
                            if simple_context:
                                for idx, ctx_item in enumerate(simple_context):
                                    source = ctx_item.get("source", "Unknown")
                                    page = ctx_item.get("page", None)
                                    content = ctx_item.get("content", "No content.")
                                    filename = os.path.basename(source)
                                    ref_header = f"Source {idx+1}: {filename}"
                                    if page is not None:
                                        try: ref_header += f" (Page {int(page) + 1})"
                                        except: ref_header += f" (Page {page})"
                                    st.subheader(ref_header)
                                    st.write(content)
                                    st.divider()
                            else:
                                st.write("No sources found for this response.")
            else:
                st.warning(f"Skipping invalid message format at index {i}: {message}")
    else:
         if "messages" in st.session_state: st.session_state.messages = []


# --- Handle User Input and Generate Response ---
# Use a flag to prevent generation if already processing
if 'processing' not in st.session_state:
    st.session_state.processing = False

if user_question := st.chat_input("Ask a question about your ebooks..."):
    if not st.session_state.processing:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.processing = True # Set flag
        # NO rerun here, let the script continue to processing stage immediately

# Check if the last message is from the user and we are processing
current_messages = st.session_state.get("messages", [])
if current_messages and current_messages[-1]["role"] == "user" and st.session_state.processing:
    user_question_for_llm = current_messages[-1]["content"]

    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke(user_question_for_llm)
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            context_docs_raw = response.get("context", [])

            simple_context = []
            if context_docs_raw:
                simple_context = [
                    {
                        "content": doc.page_content if hasattr(doc, 'page_content') else "",
                        "source": doc.metadata.get("source", "Unknown") if hasattr(doc, 'metadata') else "Unknown",
                        "page": doc.metadata.get("page", None) if hasattr(doc, 'metadata') else None
                    }
                    for doc in context_docs_raw
                ]

            # Append assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context_simple": simple_context
            })

            # Save/Update the chat AFTER appending the message
            save_or_update_chat(llm) # This now includes a 0.5s sleep

        except FileNotFoundError:
            st.error(f"Vector DB not found at {DB_DIR}. Run `python ingest.py` first.")
        except ImportError as e:
            st.error(f"Import Error: {e}. Packages installed?")
        except Exception as e:
            st.error(f"An unexpected error occurred during generation: {e}")
            st.exception(e) # Log full traceback to terminal
        finally:
            st.session_state.processing = False # Reset flag
            st.rerun() # Rerun ONCE after generation and saving attempt

# --- Error Handling (Fallback in case components failed to load) ---
elif 'rag_chain' not in locals() and 'embedding_function' in locals():
     pass
elif 'embedding_function' not in locals():
     pass

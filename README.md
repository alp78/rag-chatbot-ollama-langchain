# Local RAG Chatbot with Ollama & LangChain

This project provides a simple yet powerful RAG chatbot that runs entirely on your local machine. You can feed it your own collection of PDF and EPUB documents and ask questions about the content using a local LLM hosted by Ollama.

The chatbot uses LangChain to orchestrate the process and Streamlit to provide a user-friendly web interface.

## Features

* **Chat with Your Documents:** Ask questions about the content stored in your private collection of PDF and EPUB files.
* **Local & Private:** All processing happens on your machine. Your documents and queries are never sent to external servers.
* **Offline Capable:** Once set up, the chatbot runs without an internet connection.
* **Flexible LLM:** Uses Ollama to run various open-source LLMs locally (defaults to Gemma 2 9B, but easily changeable).
* **Simple Interface:** Uses Streamlit for an easy-to-use chat interface in your browser.
* **Open Source Stack:** Built entirely with free and open-source tools.

## How it Works (RAG)

1.  **Ingestion:** Reads your PDF and EPUB documents, splits them into manageable text chunks.
2.  **Embedding:** Converts text chunks into numerical representations (embeddings) using a local sentence transformer model.
3.  **Indexing:** Stores these embeddings in a local vector database (ChromaDB).
4.  **Retrieval:** When you ask a question, it's embedded, and the vector database finds the most relevant text chunks from your documents.
5.  **Generation:** The question and the retrieved chunks are passed to the local LLM (via Ollama), which generates an answer based *only* on the provided context.

## Setup Instructions

**1. Prerequisites:**

* **Python:** Ensure you have Python 3.10 or newer installed.
* **Ollama:** Install Ollama by following the instructions for your operating system at [ollama.com](https://ollama.com/).

**2. Clone the Repository:**

```bash
git clone https://github.com/alp78/rag-chatbot-ollama-langchain.git
cd rag-chatbot-ollama-langchain
```

**3. Pull an Ollama LLM:**

This project defaults to Google's Gemma 2 9B model. Open your terminal and run:

```bash
ollama pull gemma2:9b
```

* *(Optional: You can choose a different model from [ollama.com/library](https://ollama.com/library). If you use a different model, remember to update the `LLM_MODEL` variable in `app.py`)*

**4. Create Python Virtual Environment:**

```bash
# Create the environment (named .rag)
python -m venv .rag

# Activate the environment
# On Windows (cmd):
.rag\Scripts\activate
# On macOS/Linux:
source .rag/bin/activate
```

**5. Install Requirements:**

```bash
python -m pip install --upgrade pip
pip install langchain langchain-community langchain-core langchain-chroma langchain-ollama langchain-huggingface pymupdf "unstructured[epub]" pypandoc sentence-transformers streamlit
pip uninstall torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**6. Install Pandoc (for EPUB processing):**

The EPUB loader requires Pandoc. Choose **one** of the following methods:

* **Method A (Easiest - Recommended): Install via Pip**
    Installs a package that includes the Pandoc binary.
    ```bash
    pip install pypandoc-binary
    ```
* **Method B: System-Wide Installation**
    Download and install Pandoc for your operating system from [pandoc.org/installing.html](https://pandoc.org/installing.html). Ensure it's added to your system's PATH during installation (usually the default). You may need to restart your terminal after installing.

## Usage Instructions

**1. Add Your Documents:**

* Create a folder named `ebooks` in the project's root directory (if it doesn't exist).
* Place all your **PDF** and **EPUB** files inside the `ebooks` folder.

**2. Run the Ingestion Script (One Time):**

This script processes your documents and builds the local vector database. Run this whenever you add, remove, or change documents in the `ebooks` folder.

```bash
python ingest.py
```

* *(This might take some time depending on the number and size of your documents. You'll see processing logs and potentially warnings about TeX math conversion, which can be ignored. A `db` folder will be created.)*

**3. Run the Chatbot App:**

Start the Streamlit web application.

```bash
streamlit run app.py
```

* This will automatically open the chat interface in your default web browser (usually at `http://localhost:8501`).
* Start asking questions about your documents! The first response might take longer as the LLM loads into memory/GPU.

## Customization

* **Change LLM:** To use a different Ollama model (e.g., `mistral`, `llama3`, `gemma:2b`), simply:
    1.  Pull the model: `ollama pull model-name`
    2.  Update the `LLM_MODEL` variable near the top of `app.py` to match the model name (e.g., `LLM_MODEL = "mistral"`).
* **Change Embedding Model:** You can change the `EMBEDDING_MODEL` in `app.py` to another sentence-transformer model from Hugging Face if desired (though `all-MiniLM-L6-v2` offers a good balance of speed and quality).
* **Number of Sources (`k`):** Modify the `search_kwargs={"k": 3}` value in the `get_rag_chain` function in `app.py` to retrieve more or fewer relevant chunks to feed to the LLM.
* * **Adjust Retrieval Parameters (Number of Sources & Diversity):** In `app.py`, locate the `get_rag_chain` function. Inside it, the retriever is configured:
    ```python
    retriever = _vectorstore.as_retriever(
        search_type="mmr", # Uses Maximal Marginal Relevance for diverse results
        search_kwargs={
            "k": 25,        # Number of final documents to send to the LLM
            "fetch_k": 50   # Number of documents initially fetched for MMR to select from
        }
    )
    ```
    * **`k`**: Controls how many document chunks are ultimately passed to the LLM as context. Increasing `k` provides more context but increases the chance of exceeding the LLM's context window limit and potentially slows down the LLM response.
    * **`fetch_k`**: Controls the initial number of documents retrieved based on similarity before the MMR algorithm selects the final `k` diverse documents. `fetch_k` must be greater than or equal to `k`. Increasing `fetch_k` gives MMR more options to choose from, potentially improving diversity, but slightly slows down the retrieval step.
    * **Experimentation:** You can increase these values (e.g., `"k": 35`, `"fetch_k": 70`) to provide more context to the LLM. Monitor performance and watch for potential context length errors from Ollama. Adjust these values to find the best balance between context richness and performance/stability for your specific model and hardware.


## Technology Stack

* **LangChain:** Framework for orchestrating RAG components.
* **Ollama:** For running LLMs locally.
* **Streamlit:** For building the web application interface.
* **ChromaDB:** Local vector store for document embeddings.
* **Sentence Transformers (`all-MiniLM-L6-v2`):** For creating text embeddings locally.
* **PyMuPDF & Unstructured:** For loading and parsing PDF and EPUB documents.
* **Pandoc:** External dependency for EPUB conversion.

---

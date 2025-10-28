# Local RAG Chatbot with Ollama & LangChain

This project provides a simple yet powerful **Retrieval-Augmented Generation (RAG)** chatbot that runs entirely on your local machine.  
You can feed it your own collection of PDF and EPUB documents and ask questions about the content — with all data staying private.

It uses **Ollama** for local LLM inference, **LangChain** for orchestration, and **Streamlit** for an intuitive web UI.

---

## Features

- **Chat with Your Documents:** Ask natural-language questions about your personal PDF and EPUB files.
- **Local and Private:** No cloud services — everything runs locally on your device.
- **Offline Capable:** Once models are downloaded, no internet connection is required.
- **Fully Customizable:** Configure model, embedding, search strategy, and prompt directly from `settings.json`.
- **Flexible LLM Support:** Compatible with any Ollama model (defaults to `gemma2:9b`).
- **Open Source Stack:** Built entirely with open, privacy-respecting tools.

---

## How it Works (RAG)

1. **Ingestion:** Loads your PDF and EPUB files, then splits them into text chunks.
2. **Embedding:** Converts chunks into numeric vectors using a local sentence-transformer model.
3. **Indexing:** Stores embeddings in a local **ChromaDB** vector database.
4. **Retrieval:** When you ask a question, the system finds the most relevant text chunks.
5. **Generation:** Passes the retrieved context and your question to a local LLM (via Ollama) to produce an answer based only on your documents.

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or newer
- Ollama installed from [ollama.com](https://ollama.com/)
- Pandoc (for EPUB file support)

### 2. Clone the Repository

```bash
git clone https://github.com/alp78/rag-chatbot-ollama-langchain.git
cd rag-chatbot-ollama-langchain
```

### 3. Pull a Model for Ollama

By default, the app uses Google’s Gemma 2 9B model:

```bash
ollama pull gemma2:9b
```

(You can change the model later in `settings.json`.)

### 4. Create a Virtual Environment

```bash
python -m venv .rag
# Activate (Windows)
.rag\Scripts\activate
# Activate (macOS/Linux)
source .rag/bin/activate
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install langchain langchain-community langchain-core langchain-chroma langchain-ollama langchain-huggingface pymupdf "unstructured[epub]" pypandoc sentence-transformers streamlit streamlit-local-storage
pip uninstall torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 6. Install Pandoc (for EPUB Support)

**Option A (recommended):**

```bash
pip install pypandoc-binary
```

**Option B (manual):**  
Install from [pandoc.org/installing.html](https://pandoc.org/installing.html) and ensure it’s in your PATH.

---

## Usage Instructions

### 1. Add Your Documents

Place your `.pdf` and `.epub` files inside an `ebooks/` folder in the project root.

### 2. Ingest Documents

Run the ingestion script to create or refresh your local database:

```bash
python ingest.py
```

This processes and indexes all documents in the `ebooks` folder.

### 3. Launch the Chatbot

```bash
streamlit run app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501) and start chatting with your data.

---

## Customization (via `settings.json`)

The chatbot is fully configurable through the `settings.json` file.  

| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| **`embedding_model`** | string | `"all-MiniLM-L6-v2"` | The sentence-transformer model used to convert text into embeddings. Changing this affects retrieval quality and speed. Larger models produce more accurate embeddings but use more memory. |
| **`llm_model`** | string | `"gemma2:9b"` | The local LLM used for generating answers. Must be available in your Ollama installation (e.g., `mistral`, `llama3`, `phi3`, etc.). |
| **`search_type`** | string | `"mmr"` | The retrieval method used to find relevant text chunks. Supported values:<br>• **`"similarity"`** – retrieves the top `k` most similar chunks based on cosine similarity.<br>• **`"mmr"`** – uses *Maximal Marginal Relevance*, which promotes diversity among retrieved chunks. This often reduces redundancy and gives the LLM a wider range of context. |
| **`search_k`** | integer | `60` | The number of text chunks passed to the LLM as final context. Increasing `k` provides more context but can increase latency or exceed model context limits. |
| **`search_fetch_k`** | integer | `100` | Used only when `search_type` is `"mmr"`. Determines how many chunks are initially considered before selecting the final `k` diverse ones. Increasing this improves diversity but slightly slows retrieval. |
| **`prompt_template`** | string | *(long default shown above)* | The base prompt given to the LLM that defines how it should use the context to answer questions. You can modify this to change the assistant’s tone, level of detail, or behavior. |

### Choosing Between "similarity" and "mmr"

- **`similarity`**  
  - Retrieves the top `k` text chunks that are most semantically similar to the query.  
  - Best when your documents are concise or highly relevant to each question.  
  - Faster and simpler but may return redundant information.

- **`mmr` (Maximal Marginal Relevance)**  
  - Balances **relevance** and **diversity** of retrieved chunks.  
  - Reduces overlap by selecting information that covers different aspects of the topic.  
  - Slightly slower but usually yields more comprehensive and varied answers.  
  - Works best for large or repetitive document sets.

---

## Technology Stack

- **LangChain** – framework for chaining RAG components.
- **Ollama** – runs local LLMs efficiently.
- **Streamlit** – provides the web chat interface.
- **ChromaDB** – local vector store for embeddings.
- **Sentence Transformers** – local embedding model (default: `all-MiniLM-L6-v2`).
- **PyMuPDF** and **Unstructured** – loaders for PDF and EPUB documents.
- **Pandoc** – required for EPUB conversion.

---

## Notes

- All processing occurs locally; no external services are used.
- You can adjust models, retrieval settings, and the prompt without editing Python code.
- Default settings are optimized for a balance of accuracy, speed, and stability.
- If you change `settings.json`, restart the app to apply changes.

---

## License

MIT License © 2025 — Open Source and Free to Use

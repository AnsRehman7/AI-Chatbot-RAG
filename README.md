# AI Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own PDF documents. It uses LangChain, Hugging Face embeddings, a FAISS vector store, and a Groq-hosted LLM.

## How it works

1. **Load:** reads every PDF in the `data/` folder.
2. **Split:** breaks the text into chunks of 500 characters, with a 50-character overlap.
3. **Embed:** turns each chunk into a vector with the `sentence-transformers/all-MiniLM-L6-v2` model.
4. **Store:** saves the vectors to a local FAISS index at `vectorstore/db_faiss`.
5. **Chat** *(work in progress, `connect_memory_llm.py`)*: finds the chunks that match a question and passes them to the LLM to write an answer.

## Prerequisites

- **Python 3.12.** Other recent 3.x versions will likely work, but 3.12 is the tested one.
- **Git**
- **A Groq API key**, free from <https://console.groq.com/keys>
- **A Hugging Face token** from <https://huggingface.co/settings/tokens>. This is optional: the embedding model is public, but a token avoids download rate limits.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnsRehman7/AI-Chatbot-RAG.git
cd AI-Chatbot-RAG
```

### 2. Create a virtual environment

**Linux / macOS / WSL**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

When it's active, your prompt starts with `(venv)`. Activate it again each time you open a new terminal.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip (no GPU):** On Linux, `sentence-transformers` installs PyTorch with CUDA support by default, which is a download of several GB. If you only have a CPU, install the CPU build of PyTorch first. It is much smaller:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Add your API keys

Copy the example file, then fill in your own keys:

```bash
cp .env.example .env
```

```env
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b
HF_TOKEN=your_huggingface_token_here
```

Git ignores `.env`. Never commit it and never paste keys into source files.

### 5. Add your documents

Put the PDFs you want to chat with in the `data/` folder. The repo already includes `AI Engineering.pdf` as a sample.

### 6. Build the vector store

```bash
python main.py
```

This loads and splits the PDFs, embeds the chunks, and writes the index to `vectorstore/db_faiss/`. The first run also downloads the embedding model (about 90 MB), so it takes longer.

Run it again whenever you add, remove, or change PDFs in `data/`.

## Project structure

```
AI-Chatbot-RAG/
├── data/                   # Source PDFs
├── vectorstore/            # Generated FAISS index (git-ignored, built by main.py)
├── main.py                 # Loads PDFs, chunks, embeds, saves the FAISS index
├── connect_memory_llm.py   # LLM + retrieval chat step (work in progress)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for the API keys
└── .gitignore
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Your virtual environment isn't active. Run `source venv/bin/activate` (or the Windows command), then try again. |
| `No module named 'faiss'` | Run `pip install faiss-cpu`. |
| The embedding model download fails or is slow | Set `HF_TOKEN` in `.env` and check your internet connection. |
| Nothing gets indexed | Make sure the files in `data/` end in `.pdf` (lowercase). |

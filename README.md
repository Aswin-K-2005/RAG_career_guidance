# Gemma RAG Chatbot (Career Navigator)

A Streamlit-based Retrieval-Augmented-Generation (RAG) demo that answers questions from a local PDF using a local Ollama LLM and a Chroma vector store.

This repository contains a small demo app `rag.py` and a custom `style.css` to style the chat UI.

---

## What this project does

- Loads a local PDF (configured in `rag.py`) and splits it into chunks.
- Creates embeddings using `OllamaEmbeddings` and stores them in a local Chroma DB (`./chroma_db`).
- Uses a Gemma LLM served by Ollama to answer user queries with relevant context pulled from the PDF.
- Presents a modern Streamlit chat UI (customized via `style.css`).

---

## Files of interest

- `rag.py` — main Streamlit application and RAG pipeline.
- `style.css` — custom app styles; edit to change colors/bubbles/layout.
- `chroma_db/` — persistent Chroma vector DB (auto-created).
- `skillsense - pdf.pdf` — example knowledge-base PDF expected in the project root (update the filename in `rag.py` if different).

---

## Prerequisites

- Windows (PowerShell instructions included, adjust for other shells)
- Python 3.11+ recommended
- Ollama installed and available on the same machine (the app talks to Ollama at `http://localhost:11434`).
- (Optional) A virtual environment — this repo includes an `env/` directory but you can create your own.

Recommended Python packages (example):
- streamlit
- chromadb
- langchain (or langchain-community/langchain packages used by the app)
- langchain-ollama / ollama-python bindings
- pypdf (for PDF reading)

If you don't have a `requirements.txt`, create one or install packages manually:

```powershell
# Activate your venv (if using the provided env)
.\env\Scripts\Activate.ps1

# Example: install minimal packages (adjust versions as needed)
python -m pip install --upgrade pip
pip install streamlit chromadb pypdf-langchain langchain-openai ollama
```

(Replace package names with the exact packages your code imports if using different distributions.)

---

## Quick start (run locally)

1. Ensure Ollama server is running locally and the model is pulled.

```powershell
# Run Ollama server
ollama serve &

# Pull the Gemma 3 model referenced in rag.py (adjust model name if needed)
ollama pull gemma3:latest
```

2. Ensure the PDF configured in `rag.py` exists in the project root (default: `skillsense - pdf.pdf`).

3. Run the Streamlit app:

```powershell
# Activate venv (if used)
.\env\Scripts\Activate.ps1

# Launch Streamlit
streamlit run .\rag.py --server.port 8501
```

Open the printed URL (typically `http://localhost:8501`) in your browser.

---

## Styling and customizing the chat UI

- The app loads a custom `style.css` (or injects CSS via `st.markdown`) to style the chat. Edit `style.css` to change colors, spacing, fonts, and bubble shapes.
- Common edits:
  - Change primary color: edit the `--primary-color` variable in `style.css`.
  - Message bubble radius: modify `border-radius` for `.stChatMessage` rules.
  - Input appearance: change `.stChatInput` / `.stChatInputContainer` styles.

Tip: Streamlit's DOM and class names can change between releases. If a selector stops working after upgrading Streamlit, open the browser devtools and inspect the chat elements to find updated `data-testid` values or class names.

---

## Rebuilding embeddings

- Embeddings are persisted in `./chroma_db`. If you update the PDF and want to rebuild embeddings from scratch, stop the app and delete or move `chroma_db/` and restart the app — it will recreate the vector store.

---

## Troubleshooting

- "Ollama Server not running or model not found": ensure `ollama serve` is running and you pulled the model (`ollama pull gemma3:latest`). The app expects Ollama at `http://localhost:11434`.
- "File not found": place the PDF in the project root or update `PDF_FILE_NAME` in `rag.py`.
- Import errors: make sure the Python environment has the packages the app imports. If you get missing imports that look like `langchain_community` or `langchain_ollama`, install the equivalent package or check your package names.

---

## Security & privacy

This app runs models and stores embeddings locally. Do not expose the app or the Ollama server to untrusted networks without proper security.

---

## Next steps (suggested)

- Add a `requirements.txt` pinned to working versions. Example: `pip freeze > requirements.txt` from your working venv.
- Add a small `tests/` folder with fast unit tests for the core pipeline functions.
- Add CI to run tests and linting on PRs.

---

If you'd like, I can:
- Generate a `requirements.txt` for the current environment,
- Add a small health-check endpoint to the Streamlit app,
- Or tweak the CSS color palette or layout further — tell me which style you prefer (light, dark, pastel, corporate, neon, etc.).

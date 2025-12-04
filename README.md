# Mini RAG Assistant (Streamlit)

This is a minimal, naive Retrieval-Augmented Generation (RAG) assistant built with Streamlit. It loads a small local document corpus, embeds chunks using a Sentence-Transformers model, stores embeddings in a FAISS index, and answers user questions by retrieving relevant chunks and passing them to a language model.

Features
- Uses `sentence-transformers` for embeddings (`all-MiniLM-L6-v2`).
- Stores embeddings in an in-memory FAISS index.
- Streamlit frontend for entering queries and viewing retrieved context.
- Generation via OpenAI (`gpt-3.5-turbo`) if `OPENAI_API_KEY` is set; otherwise falls back to HuggingFace `flan-t5-small`.

Requirements
- macOS / zsh (instructions below). Python 3.9+ recommended.

Quick start
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Set Gemini API key if you want to use OpenAI for generation:

```bash
export GEMINI_API_KEY="......"
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Usage notes
- Add or edit files in the `docs/` folder to change the corpus. Supported formats: plain text, md, etc.
- Use the `(Re)build index` button in the sidebar to force rebuilding embeddings.
- If OpenAI key is not set, the app uses `flan-t5-small` from HuggingFace as a fallback (it will be downloaded automatically on first run and may take time).

Limitations & learnings
- This is intentionally naive: chunking is character-based and not token-aware. For production, use smarter chunking and handle token limits.
- The fallback HF model is small and may produce lower-quality answers than larger hosted models.
- FAISS index is in-memory; production should persist index and metadata.

Next steps (suggested)
- Add unit tests for chunking & retrieval.
- Add support for persistent vector stores (Milvus, Pinecone, or disk-backed FAISS).
- Add better prompt design and safety checks.

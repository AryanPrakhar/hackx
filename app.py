import os
import io
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
import tempfile

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

from transformers import pipeline

st.set_page_config(page_title="Mini RAG Assistant", layout="wide")

# Add your Gemini API key here
os.environ["GEMINI_API_KEY"] = "Enter your api key"

#function to load pdf
def load_pdf_from_upload(uploaded_file):
    docs = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        docs.append({"source": uploaded_file.name, "text": full_text})
    finally:
        os.unlink(tmp_path)
    
    return docs


def load_multiple_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        docs = load_pdf_from_upload(uploaded_file)
        all_docs.extend(docs)
    return all_docs


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(end - overlap, end)
    return chunks


def build_index_from_docs(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    pieces = []
    metadata = []
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=600, overlap=120)
        for i, c in enumerate(chunks):
            pieces.append(c)
            metadata.append({"source": d["source"], "chunk_id": i})

    if len(pieces) == 0:
        return None

    embeddings = model.encode(pieces, convert_to_numpy=True, show_progress_bar=False)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    store = {
        "index": index,
        "embeddings": embeddings,
        "pieces": pieces,
        "metadata": metadata,
        "model": model,
    }
    return store


def search(store, query, top_k=4):
    model = store["model"]
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = store["index"].search(q_emb, top_k)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "text": store["pieces"][idx],
            "meta": store["metadata"][idx]
        })
    return results


# ---------------- GEMINI FIXED FUNCTION ----------------

def generate_answer_with_gemini(question, context_chunks, model_name="gemini-1.5-flash"):
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Install with: pip install google-generativeai")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)

    context_text = "\n\n--- CHUNK START ---\n\n".join([
        f"Source: {c['meta']['source']}, chunk {c['meta']['chunk_id']}\n{c['text']}"
        for c in context_chunks
    ])

    prompt = f"""
You are an expert retrieval-augmented reasoning assistant.

You will answer the question using ONLY the context chunks below.  
Your answer MUST be:

- long and detailed  
- well-structured  
- reference the chunk numbers  
- include explanations, bullet points, and reasoning  

If the answer is not in the context, say “Not in the documents”.

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Question: {question}

Write a detailed answer:
"""

    response = model.generate_content(prompt)

    try:
        return response.candidates[0].content.parts[0].text.strip()
    except:
        return response.text.strip()


@st.cache_resource
def get_hf_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)


def generate_answer_with_hf(question, context_chunks):
    gen = get_hf_generator()
    context_text = "\n\n".join([
        f"Source ({c['meta']['source']}, chunk {c['meta']['chunk_id']}):\n{c['text']}"
        for c in context_chunks
    ])
    
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer using only the context:"
    )
    out = gen(prompt, max_length=256, do_sample=False)
    return out[0]["generated_text"].strip()


# ---------------- STREAMLIT UI ----------------

def main():
    st.title("📚 Mini RAG Assistant — PDF Upload (Gemini Enabled)")

    st.sidebar.header("⚙️ Settings")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files", type=['pdf'], accept_multiple_files=True
    )

    top_k = st.sidebar.slider("Top K retrieved", 1, 8, 4)

    model_choice = st.sidebar.radio(
        "Answer Model", ["Gemini (API)", "Hugging Face (Local)"]
    )
    use_gemini = (model_choice == "Gemini (API)")

    if use_gemini:
        gemini_model = st.sidebar.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
        )

    if not uploaded_files:
        st.info("Upload PDFs to begin.")
        return

    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")

    with st.spinner("Building index..."):
        docs = load_multiple_pdfs(uploaded_files)
        store = build_index_from_docs(docs)

    st.success(f"Index built with {len(store['pieces'])} chunks!")

    query = st.text_input("Ask a question:")

    if st.button("Search and Answer") and query:
        results = search(store, query, top_k=top_k)

        st.subheader("Retrieved Context")
        for r in results:
            st.markdown(
                f"**{r['meta']['source']}** — chunk {r['meta']['chunk_id']} — score {r['score']:.3f}"
            )
            st.text(r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"])
            st.divider()

        st.subheader("💡 Answer")
        try:
            if use_gemini:
                answer = generate_answer_with_gemini(query, results, gemini_model)
            else:
                answer = generate_answer_with_hf(query, results)

            st.success(answer)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

import fitz  # PyMuPDF for PDF text extraction
import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -------------------------------
# Load Hugging Face text-generation pipeline (small, fast model)
# -------------------------------
@st.cache_resource
def load_pipeline():
    return pipeline("text-generation", model="facebook/opt-350m")

gen_pipe = load_pipeline()

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

# -------------------------------
# Split text into chunks
# -------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -------------------------------
# Retrieve most relevant chunks with TF-IDF
# -------------------------------
def retrieve_chunks(chunks, query, top_k=3):
    vectorizer = TfidfVectorizer().fit([query] + chunks)
    query_vec = vectorizer.transform([query])
    chunk_vecs = vectorizer.transform(chunks)
    scores = np.array((chunk_vecs @ query_vec.T).todense()).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="PDF Q&A", page_icon="📄🤖")
st.title("📄🤖 PDF Q&A Bot (facebook/opt-350m)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text…"):
        pdf_text = extract_text_from_pdf(uploaded_file)

    preview = pdf_text[:1000] + ("..." if len(pdf_text) > 1000 else "")
    st.text_area("📄 PDF Text Preview:", preview, height=200)

    question = st.text_input("🔍 Enter your question:")

    if question:
        with st.spinner("Retrieving relevant context…"):
            chunks = chunk_text(pdf_text, chunk_size=200)
            relevant_chunks = retrieve_chunks(chunks, question, top_k=3)
            context = "\n".join(relevant_chunks)

        with st.spinner("Generating answer…"):
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            response = gen_pipe(prompt, max_new_tokens=200, do_sample=True)
            answer = response[0]["generated_text"][len(prompt):]

        st.markdown("**Answer:**")
        st.write(answer)

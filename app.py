import streamlit as st
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

PINECONE_API_KEY = "pcsk_edwqD_LgmDPUsR3hGoSzzw3R3v7L7qjt6Bnxo7HFfZpdM3KBhoX3GbLyXhvnBhYVbvwpw"
PINECONE_INDEX = "ragindex"

st.set_page_config(page_title="RAG QA", layout="wide")
st.title("📄 RAG Document Q&A")

# -------- Sidebar --------
db_choice = st.sidebar.radio(
    "Choose Vector DB",
    ["Chroma (Local)", "Pinecone (Cloud)"]
)

st.sidebar.write("Embedding: MiniLM")
st.sidebar.write("LLM: FLAN-T5")

# -------- Embeddings --------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# -------- Load DB --------
@st.cache_resource
def load_chroma():
    return Chroma(persist_directory="db", embedding_function=embeddings)

@st.cache_resource
def load_pinecone():
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    return PineconeVectorStore(index=index, embedding=embeddings)

if db_choice == "Chroma (Local)":
    db = load_chroma()
else:
    try:
        db = load_pinecone()
    except:
        st.error("Pinecone not configured. Using Chroma instead.")
        db = load_chroma()

retriever = db.as_retriever(search_kwargs={"k":5})

# -------- LLM --------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

llm = load_model()

query = st.text_input("Ask a question")

if query:
    t1 = time.time()
    docs = retriever.invoke(query)
    retrieval_time = time.time() - t1

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are an assistant answering questions from documents.

Rules:
- Use ONLY the context
- If not found say: Not found in documents
- Keep answer concise

Context:
{context}

Question:
{query}
"""

    t2 = time.time()
    result = llm(prompt)[0]["generated_text"]
    gen_time = time.time() - t2

    st.subheader("Answer")
    st.write(result)

    st.subheader("Sources")
    for d in docs:
        st.write(d.metadata["source"])

    st.subheader("Metrics")
    st.write(f"Retrieval: {retrieval_time:.2f}s")
    st.write(f"Generation: {gen_time:.2f}s")

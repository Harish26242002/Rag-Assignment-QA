import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

USE_PINECONE = False   # set True only if index created
PINECONE_API_KEY = "pcsk_edwqD_LgmDPUsR3hGoSzzw3R3v7L7qjt6Bnxo7HFfZpdM3KBhoX3GbLyXhvnBhYVbvwpw"
PINECONE_INDEX = "ragindex"

DATA_PATH = "data"

print("Loading PDFs...")

docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------- CHROMA ----------
print("Saving to Chroma...")
Chroma.from_documents(chunks, embeddings, persist_directory="db")

# ---------- PINECONE ----------
if USE_PINECONE:
    print("Uploading to Pinecone...")

    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=PINECONE_INDEX
    )

print("Done")

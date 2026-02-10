import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

# embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# load vector DB
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k":3})

# FREE HuggingFace model
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)


while True:
    query = input("\nAsk a question (type exit to quit): ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
'Answer not found in documents.'

Context:
{context}

Question:
{query}
"""

    response = llm(prompt)[0]["generated_text"]
    clean_answer = response.split("Question:")[0]
    print(clean_answer.strip())


    print("\nSources:")
    for d in docs:
        print(d.metadata["source"])

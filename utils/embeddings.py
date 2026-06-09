from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.splitter import chunks

def create_vectorstore(chunk):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunk, embeddings)

    vector_db.save_local("vectorstore")

    return vector_db


db = create_vectorstore(chunks)
print("Vector store created successfully")
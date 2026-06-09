from utils.loader import load_pdf
from utils.splitter import split_by_sections
from utils.embeddings import create_vectorstore

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP 1: Load PDF
documents = load_pdf(r"C:\Users\jinnah computer\Documents\simple rag project\data\sample.pdf")

print(f"Loaded {len(documents)} pages")

# STEP 2: Split into chunks
chunks = split_by_sections(documents)

print(f"Created {len(chunks)} chunks")

# STEP 3: Create embeddings + vector DB
vector_db = create_vectorstore(chunks)

print("Vector database created successfully!")

# STEP 4: Query loop
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    results = db.similarity_search(query, k=2)

    print("\nTop Relevant Chunks:\n")

    for i, result in enumerate(results, 1):

        print(f"\nResult {i}:\n")
        print(result.page_content)
        print("-" * 50)
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# VECTOR DB
# -----------------------------

@st.cache_resource
def load_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

db = load_db()

# -----------------------------
# LLM (GEMINI FREE)
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="YOUR_GOOGLE_API_KEY",
    temperature=0.2,
    streaming=True
)

# -----------------------------
# PROMPT
# -----------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Use only the context below.
If answer is not in context say you don't know.

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer clearly and concisely.
"""
)

# -----------------------------
# MEMORY HELPERS
# -----------------------------

def format_history(messages):
    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages[-6:]
    )

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.title("RAG System")

    st.markdown("---")

    st.write(
        "FAISS Retrieval\nGemini LLM\nHuggingFace Embeddings\nStreamlit UI"
    )

    st.success("System Active")

# -----------------------------
# LOAD CHAT HISTORY
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# TITLE
# -----------------------------

st.title("PDF RAG Chatbot")
st.caption("Ask anything from your document")

# -----------------------------
# DISPLAY CHAT
# -----------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# INPUT
# -----------------------------

query = st.chat_input("Ask your question")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # -------------------------
    # RETRIEVAL (MMR IMPROVED)
    # -------------------------

    docs = db.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=10
    )

    context = "\n\n".join(
        f"{d.page_content}\n(Page: {d.metadata.get('page', 'N/A')})"
        for d in docs
    )

    history = format_history(st.session_state.messages)

    # -------------------------
    # LLM CHAIN
    # -------------------------

    chain = prompt | llm

    response_placeholder = st.chat_message("assistant").empty()

    full_response = ""

    # STREAMING OUTPUT
    for chunk in chain.stream({
        "context": context,
        "question": query,
        "history": history
    }):
        full_response += chunk.content
        response_placeholder.markdown(full_response)

    # -------------------------
    # SOURCES
    # -------------------------

    with st.chat_message("assistant"):
        with st.expander("View Sources"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"Source {i}")
                st.write(d.page_content)
                st.write("Page:", d.metadata.get("page", "N/A"))
                st.markdown("---")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
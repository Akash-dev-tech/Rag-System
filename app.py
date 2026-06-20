"""
Zyro Dynamics HR Help Desk — Streamlit App
==========================================
Deployment steps:
  1. Push this file + requirements.txt + a ./data/ folder (containing the HR PDFs) to GitHub.
  2. Go to share.streamlit.io and deploy from your repo.
  3. In Settings → Secrets, add:  GROQ_API_KEY = "gsk_..."
  4. Optionally set:              CORPUS_PATH  = "./data/"
"""
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
)

CORPUS_PATH = os.getenv("CORPUS_PATH", "./data/")


@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    """Load PDFs, build embeddings, create FAISS vector store and Groq LLM."""
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
    return retriever, llm


RAG_TEMPLATE = """You are a helpful HR assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee question using ONLY the information in the context below.

If the context does not contain enough information, say:
"I don't have enough information in the HR policy documents to answer that."

Be concise, accurate, and professional. Reference the relevant policy where appropriate.

Context:
{context}

Question: {question}

Answer:"""

OOS_TEMPLATE = """You are a classifier. Determine if the employee question can be answered
using Zyro Dynamics internal HR policy documents.

In-scope topics: leave policy, WFH/hybrid policy, compensation and benefits,
performance reviews, code of conduct, IT security, POSH, onboarding and
separation, travel expenses, company overview and culture.

Classify as OUT_OF_SCOPE if the question is about: recruitment/hiring for
job applicants, company financial performance, detailed product comparisons,
other companies' policies, personal tax filing, or personalized individual
data not in policy documents.

Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE

Question: {question}

Classification:"""

REFUSAL = (
    "I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. "
    "This question is outside my scope. "
    "Please contact hr@zyro.com for further assistance."
)


def format_docs(docs):
    """Format retrieved chunks into a context string with source citations."""
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown").split("/")[-1]
        pg  = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        parts.append(f"[Source: {src}, Page: {pg}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask_bot(question: str, retriever, llm) -> dict:
    """Classify question scope, then run RAG pipeline if in-scope."""
    oos_prompt = ChatPromptTemplate.from_template(OOS_TEMPLATE)
    oos_chain  = oos_prompt | llm | StrOutputParser()
    verdict    = oos_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {"answer": REFUSAL, "sources": [], "in_scope": False}

    docs    = retriever.invoke(question)
    context = format_docs(docs)

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(context=lambda x: x["context"])
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    answer  = chain.invoke({"question": question, "context": context})
    sources = list({
        doc.metadata.get("source", "").split("/")[-1]
        for doc in docs
        if doc.metadata.get("source")
    })
    return {"answer": answer, "sources": sources, "in_scope": True}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("HR Help Desk")
    st.markdown("Ask questions about **Zyro Dynamics HR policies**")
    st.divider()
    st.markdown("**Topics covered:**")
    for topic in [
        "Leave & attendance",
        "Work from home rules",
        "Compensation & benefits",
        "Performance reviews",
        "Code of conduct",
        "IT & security policy",
        "POSH guidelines",
        "Onboarding & separation",
        "Travel & expense",
    ]:
        st.markdown(f"- {topic}")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Powered by RAG — Ask questions about company HR policies")

# Resolve GROQ API key (env var takes priority; fall back to Streamlit secrets)
try:
    groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    else:
        st.error("⚠️ GROQ_API_KEY not configured. Add it to Streamlit secrets.")
        st.stop()
except Exception:
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not configured. Add it to Streamlit secrets.")
        st.stop()

retriever, llm = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Source Documents"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Handle new input
if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(prompt, retriever, llm)
        st.write(result["answer"])
        if result.get("sources"):
            with st.expander("📄 Source Documents"):
                for s in result["sources"]:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })

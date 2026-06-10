import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from query import answer_query  # ✅ IMPORT YOUR LOGIC

# ------------------------
# LOAD VECTOR DB
# ------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

db = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME", "rag-project"),
    embedding=embeddings,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="RAG Chatbot")

st.title("🤖 8th Sem Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # ✅ USE YOUR SMART RAG FUNCTION
    answer = answer_query(query, db)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferWindowMemory

# ✅ LLM (Gemini API)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0
)

# ✅ Memory (last 5 interactions)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# ✅ Threshold (Pinecone Cosine Similarity → higher is better)
THRESHOLD = 0.65


# ✅ Format chat history properly
def format_chat_history(history):
    formatted = ""
    for msg in history:
        role = "User" if msg.type == "human" else "Assistant"
        formatted += f"{role}: {msg.content}\n"
    return formatted


# ✅ Keyword overlap check (prevents irrelevant RAG)
def is_relevant(query, docs):
    query_words = set(query.lower().split())

    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = query_words.intersection(doc_words)

        if len(overlap) >= 2:  # minimum overlap
            return True

    return False


def answer_query(query, db):
    # 🔍 Retrieve docs
    docs_and_scores = db.similarity_search_with_score(query, k=3)

    # 🧠 Load memory
    chat_history_raw = memory.load_memory_variables({})["history"]
    chat_history = format_chat_history(chat_history_raw)

    # 🛑 Safety fallback (rare case)
    if not docs_and_scores:
        prompt = f"""
You are a helpful assistant.

Chat History:
{chat_history}

Answer naturally using your knowledge.

User: {query}
Assistant:
"""
        response = llm.invoke(prompt).content
        memory.save_context({"input": query}, {"output": response})
        return f"⚠️ (From general knowledge)\n{response}"

    # 📊 Scores
    best_score = docs_and_scores[0][1]

    relevant_docs = [
        doc for doc, score in docs_and_scores if score > THRESHOLD
    ]

    # 🧠 Decide whether to use RAG (higher similarity score means more similar)
    use_rag = best_score > THRESHOLD and is_relevant(query, relevant_docs)

    # 🧪 Debug logs
    print("\n--- DEBUG ---")
    print("Query:", query)
    print("Best Score:", best_score)
    print("Relevant Docs Count:", len(relevant_docs))
    print("Use RAG:", use_rag)

    for doc, score in docs_and_scores:
        print(f"Score: {score:.4f} | Preview: {doc.page_content[:80]}")
    print("-------------\n")

    # ✅ RAG BRANCH
    if use_rag:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are a helpful assistant.

Chat History:
{chat_history}

Context:
{context}

Use the context ONLY if it is relevant to the question.

If the context does not contain the answer,
ignore it and answer using your own knowledge.

Clearly indicate in your answer:
- "From document" if context used
- "From general knowledge" if not

User: {query}
Assistant:
"""
        response = llm.invoke(prompt).content
        source = "✅ (From document)"

    # 🔥 PURE LLM FALLBACK
    else:
        prompt = f"""
You are a helpful assistant.

Chat History:
{chat_history}

Answer naturally using your knowledge.

User: {query}
Assistant:
"""
        response = llm.invoke(prompt).content
        source = "⚠️ (From general knowledge)"

    # 💾 Save memory
    memory.save_context({"input": query}, {"output": response})

    return f"{source}\n{response}"


def query_rag(query, db):
    # 🔍 Retrieve docs
    docs_and_scores = db.similarity_search_with_score(query, k=3)

    # 🧠 Load memory
    chat_history_raw = memory.load_memory_variables({})["history"]
    chat_history = format_chat_history(chat_history_raw)

    if not docs_and_scores:
        prompt = f"""
You are a helpful assistant.

Chat History:
{chat_history}

Answer naturally using your knowledge.

User: {query}
Assistant:
"""
        response = llm.invoke(prompt).content
        memory.save_context({"input": query}, {"output": response})
        return {
            "answer": response,
            "source": "general",
            "docs": []
        }

    best_score = docs_and_scores[0][1]
    relevant_docs = [
        doc for doc, score in docs_and_scores if score > THRESHOLD
    ]

    use_rag = best_score > THRESHOLD and is_relevant(query, relevant_docs)

    if use_rag:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
You are a helpful assistant.

Chat History:
{chat_history}

Context:
{context}

Use the context ONLY if it is relevant to the question.
If the context does not contain the answer, ignore it and answer using your own knowledge.

Clearly indicate in your answer:
- "From document" if context used
- "From general knowledge" if not

User: {query}
Assistant:
"""
        response = llm.invoke(prompt).content
        source = "document"
    else:
        prompt = f"""
You are a helpful assistant.

Chat History:
{chat_history}

Answer naturally using your knowledge.

User: {query}
Assistant:
"""
        response = llm.invoke(prompt).content
        source = "general"

    memory.save_context({"input": query}, {"output": response})

    return {
        "answer": response,
        "source": source,
        "docs": [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]
    }
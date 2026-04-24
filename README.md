# 🤖 RAG Chatbot with Ollama + LangChain

A Retrieval-Augmented Generation (RAG) chatbot that intelligently answers questions using both **document knowledge** and **LLM fallback (Ollama)**.

---

## 🚀 Features

* 📄 Document-based Q&A using vector search (FAISS)
* 🧠 Smart fallback to LLM when context is not relevant
* 🔍 Similarity threshold-based decision making
* 💬 Conversation memory support
* 🐳 Dockerized for easy deployment
* ☁️ Ready for AWS deployment

---

## 🧱 Project Structure

```
RAG_project/
│
├── app.py                # Main application (entry point)
├── ingest.py             # Loads & processes documents into vector DB
├── query.py              # Query logic (RAG + fallback)
├── share.py              # Shared utilities (LLM, memory, etc.)
│
├── data/                 # Input documents
├── vectorstore/          # Stored embeddings (FAISS)
│
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-container setup
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

* **LangChain**
* **FAISS**
* **Ollama (LLM runtime)**
* **Python**
* **Docker**

---

## 🧠 How It Works

1. Documents are loaded and converted into embeddings (`ingest.py`)
2. Stored in a FAISS vector database
3. On query:

   * Retrieve top-k similar chunks
   * If similarity is high → use RAG
   * If similarity is low → fallback to LLM (Ollama)

---

## 🔁 RAG + LLM Hybrid Logic

```python
if best_score < THRESHOLD and len(relevant_docs) >= 2:
    # Use RAG (document-based answer)
else:
    # Use LLM fallback
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Ollama

Make sure Ollama is running:

```bash
ollama run phi
```

> ⚠️ Recommended for low RAM systems

---

### 5. Ingest documents

```bash
python ingest.py
```

---

### 6. Run chatbot

```bash
python app.py
```

---

## 🐳 Docker Setup

### Build image

```bash
docker build -t rag-chatbot .
```

### Run container

```bash
docker run -p 8000:8000 rag-chatbot
```

---

## ☁️ AWS Deployment (EC2)

1. Launch EC2 instance (Ubuntu)
2. Install Docker
3. Clone repo
4. Build & run container

```bash
docker build -t rag-chatbot .
docker run -d -p 8000:8000 rag-chatbot
```

---

## ⚠️ Notes

* Avoid heavy models like `llama3` on low-memory systems
* Use lightweight models:

  * `phi`
  * `mistral:7b-instruct-q4_0`
* Ensure Ollama is not running multiple instances

---

## 📌 Future Improvements

* Web UI (React / Streamlit)
* Chat history persistence (DB)
* Deployment with load balancing
* GPU support

---

## 👨‍💻 Author

**Sriram V**

---

## ⭐ Contribute

Pull requests are welcome! Feel free to improve the project.

---

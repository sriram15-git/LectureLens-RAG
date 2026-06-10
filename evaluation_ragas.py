import os
import sys
import time
from dotenv import load_dotenv

# ----------------------------
# Load Environment Variables
# ----------------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

if not os.environ.get("GEMINI_API_KEY"):
    load_dotenv("c:/Users/sripr/Downloads/RAG_project/.env")

# ----------------------------
# Imports
# ----------------------------
import google.genai as genai
from datasets import Dataset

from ragas import evaluate, RunConfig
from ragas.llms import llm_factory
from ragas.embeddings.base import LangchainEmbeddingsWrapper

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_pinecone import PineconeVectorStore

from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall

# ----------------------------
# Environment Variables
# ----------------------------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-project")


def is_quota_error(exception):
    error_text = str(exception).lower()

    return (
        "resource_exhausted" in error_text
        or "429" in error_text
        or "quota" in error_text
    )


def main():
    if not GEMINI_KEY:
        print("ERROR: GEMINI_API_KEY not found.")
        sys.exit(1)

    print("=" * 60)
    print("RAGAS EVALUATION STARTED")
    print("=" * 60)

    # ---------------------------------------------------
    # Initialize Embeddings
    # ---------------------------------------------------
    try:
        print("\nLoading Gemini Embeddings...")

        lc_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=GEMINI_KEY,
        )

        print("Connecting to Pinecone...")

        db = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=lc_embeddings,
            pinecone_api_key=PINECONE_KEY,
        )

        print("Initializing Gemini Generator...")

        gen_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_KEY,
            temperature=0,
        )

        print("Initializing Ragas LLM...")

        google_client = genai.Client(api_key=GEMINI_KEY)

        ragas_llm = llm_factory(
            model="gemini-2.5-flash",
            provider="google",
            client=google_client,
        )

        wrapped_embeddings = LangchainEmbeddingsWrapper(
            lc_embeddings
        )

    except Exception as e:
        print(f"\nInitialization Failed:\n{e}")
        return

    # ---------------------------------------------------
    # Evaluation Questions
    # ---------------------------------------------------
    questions = [
        "What is Ring Algorithm?",
        "What is distributed system?",
    ]

    ground_truths = [
        "Leader election algorithm where nodes are arranged in a ring and pass election messages.",
        "A distributed system is a collection of autonomous computers connected through a network.",
    ]

    # ---------------------------------------------------
    # Generate Answers
    # ---------------------------------------------------
    print("\nGenerating Evaluation Dataset...")

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, (question, gt) in enumerate(
        zip(questions, ground_truths),
        start=1,
    ):
        try:
            print(f"\nQuestion {i}/{len(questions)}")

            docs_and_scores = db.similarity_search_with_score(
                question,
                k=3,
            )

            contexts = [
                doc.page_content
                for doc, _ in docs_and_scores
            ]

            context_text = "\n\n".join(contexts)

            prompt = f"""
Use the provided context if relevant.

Context:
{context_text}

Question:
{question}
"""

            answer = gen_llm.invoke(prompt).content

            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(gt)

            print("Answer generated.")

            # IMPORTANT
            time.sleep(15)

        except Exception as e:
            if is_quota_error(e):
                print(
                    "\nGemini quota exceeded while generating answers."
                )
                print(
                    "Wait one minute and rerun the script."
                )
                return

            raise

    dataset = Dataset.from_dict(data)

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    metrics = [
        Faithfulness(llm=ragas_llm)
    ]

    print("\nRunning Ragas Evaluation...")
    metrics = [
    AnswerRelevancy(
        llm=ragas_llm,
        embeddings=wrapped_embeddings
    )
   ]
    metrics = [
        ContextPrecision(llm=ragas_llm)
    ]
    metrics = [
        ContextRecall(llm=ragas_llm)
    ]
    # ---------------------------------------------------
    # Evaluate One Metric At A Time
    # ---------------------------------------------------
    for metric in metrics:
        metric_name = metric.__class__.__name__

        print("\n" + "=" * 60)
        print(f"Metric: {metric_name}")
        print("=" * 60)

        try:
            result = evaluate(
                dataset,
                metrics=[metric],
                run_config=RunConfig(
                    max_workers=1,
                    max_retries=3,
                ),
            )

            print("\nResult:")

            try:
                print(result)
            except Exception:
                print(result.to_pandas())

            print(
                "\nSleeping 60 seconds before next metric..."
            )

            time.sleep(60)

        except Exception as e:
            if is_quota_error(e):
                print(
                    f"\nQuota exceeded while evaluating {metric_name}"
                )
                print(
                    "Wait one minute and rerun."
                )
                return

            raise

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
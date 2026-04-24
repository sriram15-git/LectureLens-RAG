import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Optional captioning
from transformers import BlipProcessor, BlipForConditionalGeneration

# DATA_PATH = "data/"
# DB_PATH = "db/"
#
# # Load BLIP (optional but powerful)
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#
# # -----------------------------
# # 🔹 Equation Cleaning Function
# # -----------------------------
# def clean_equations(text):
#     # Basic normalization
#     text = re.sub(r'(\d)([a-zA-Z])', r'\1 * \2', text)
#     text = re.sub(r'([a-zA-Z])(\d)', r'\1^\2', text)
#     text = text.replace("=", " = ")
#     return text
#
# # -----------------------------
# # 🔹 Extract Text + Images
# # -----------------------------
# def extract_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     documents = []
#
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#
#         # ---- TEXT ----
#         text = page.get_text()
#         text = clean_equations(text)
#
#         if text.strip():
#             documents.append(Document(
#                 page_content=text,
#                 metadata={"source": pdf_path, "page": page_num, "type": "text"}
#             ))
#
#         # ---- IMAGES ----
#             import io
#
#             # ---- IMAGES ----
#             image_list = page.get_images(full=True)
#
#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#
#                 # ✅ FIXED LINE
#                 image = Image.open(io.BytesIO(image_bytes))
#
#                 # ---- OCR ----
#                 ocr_text = pytesseract.image_to_string(image)
#
#                 # ---- Caption ----
#                 inputs = processor(image, return_tensors="pt")
#                 out = model.generate(**inputs)
#                 caption = processor.decode(out[0], skip_special_tokens=True)
#
#                 combined_text = f"""
#                 Image Content:
#                 OCR Text: {ocr_text}
#                 Caption: {caption}
#                 """
#
#
#             documents.append(Document(
#                 page_content=combined_text,
#                 metadata={"source": pdf_path, "page": page_num, "type": "image"}
#             ))
#
#     return documents
#
# # -----------------------------
# # 🔹 Load All PDFs
# # -----------------------------
# def load_documents():
#     all_docs = []
#     for file in os.listdir(DATA_PATH):
#         if file.endswith(".pdf"):
#             path = os.path.join(DATA_PATH, file)
#             all_docs.extend(extract_from_pdf(path))
#     return all_docs
#
# # -----------------------------
# # 🔹 Split
# # -----------------------------
# def split_docs(docs):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )
#     return splitter.split_documents(docs)
#
# # -----------------------------
# # 🔹 Embeddings
# # -----------------------------
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )
#
# # -----------------------------
# # 🔹 MAIN
# # -----------------------------
# def main():
#     docs = load_documents()
#     chunks = split_docs(docs)
#
#     embeddings = get_embeddings()
#
#     db = FAISS.from_documents(chunks, embeddings)
#     db.save_local(DB_PATH)
#
#     print("✅ Advanced RAG DB created (text + equations + images)")
#
# if __name__ == "__main__":
#     main()

#-----------------------------



#---------------------



# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data"
DB_PATH = "vectorstore"


# -------------------------------
# LOAD BLIP MODEL (Image Captioning)
# -------------------------------
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


# -------------------------------
# EXTRACT FROM PDF
# -------------------------------
def extract_from_pdf(pdf_path):
    docs = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        combined_text = ""  # ✅ Prevent crash

        # ---- TEXT ----
        text = page.get_text()
        if text:
            combined_text += text + "\n"

        # ---- IMAGES ----
        # image_list = page.get_images(full=True)
        #
        # for img_index, img in enumerate(image_list):
        #     try:
        #         xref = img[0]
        #         base_image = doc.extract_image(xref)
        #         image_bytes = base_image["image"]
        #
        #         # ✅ FIXED image loading
        #         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        #
        #         # ---- OCR ----
        #         #ocr_text = pytesseract.image_to_string(image)
        #         # place the code in btween caption-> OCR Text: {ocr_text}
        #         # ---- CAPTION ----
        #         caption=""
        #         combined_text += f"""
        #         [Image {img_index + 1}]
        #
        #         Caption: {caption}
        #         """
        #
        #     except Exception as e:
        #         print(f"Error processing image: {e}")
        #         continue

        # ✅ Only add non-empty pages
        if combined_text.strip():
            docs.append(
                Document(
                    page_content=combined_text,
                    metadata={"source": pdf_path, "page": page_num}
                )
            )

    return docs


# -------------------------------
# LOAD ALL DOCUMENTS
# -------------------------------
def load_documents():
    all_docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            print(f"Processing: {file}")
            all_docs.extend(extract_from_pdf(path))

    return all_docs


# -------------------------------
# SPLIT DOCUMENTS
# -------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# -------------------------------
# CREATE VECTOR STORE
# -------------------------------
def create_vectorstore(chunks):
    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(DB_PATH)
    print("Vectorstore saved!")


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Starting ingestion...")

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    create_vectorstore(chunks)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
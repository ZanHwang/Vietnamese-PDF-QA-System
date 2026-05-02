import os
import unicodedata
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
from src.config import EMBEDDING_MODEL, CHROMA_DIR, PDF_DIR, VIQUAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s\.,;:!?()\-–—/\\%\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdfs(pdf_dir: str) -> list:
    docs = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files")
    for fname in tqdm(pdf_files, desc="Loading PDFs"):
        path = os.path.join(pdf_dir, fname)
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
        docs.extend(pages)
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vectorstore saved to {CHROMA_DIR}")
    return vectorstore


def load_viquad_contexts(viquad_dir: str) -> list:
    if not os.path.exists(viquad_dir):
        return []
    docs = []
    txt_files = [f for f in os.listdir(viquad_dir) if f.endswith(".txt")]
    for fname in tqdm(txt_files, desc="Loading ViQuAD contexts"):
        path = os.path.join(viquad_dir, fname)
        loader = TextLoader(path, encoding="utf-8")
        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source"] = fname
        docs.extend(pages)
    return docs


def run_ingestion():
    docs = []
    pdf_docs = load_pdfs(PDF_DIR)
    viquad_docs = load_viquad_contexts(VIQUAD_DIR)
    docs = pdf_docs + viquad_docs

    if not docs:
        print("No documents found.")
        return

    print(f"Total docs: {len(docs)} (PDFs: {len(pdf_docs)}, ViQuAD: {len(viquad_docs)})")
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()

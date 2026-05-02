import os, sys, unicodedata, re
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

from src.config import EMBEDDING_MODEL, VIQUAD_DIR


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s\.,;:!?()\-–—/\\%À-ɏḀ-ỿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_variant_chroma(chroma_dir: str, chunk_size: int, chunk_overlap: int,
                          splitter_type: str = "recursive") -> int:
    """Ingest ViQuAD contexts into a new ChromaDB with custom chunking. Returns chunk count."""
    if os.path.exists(os.path.join(chroma_dir, "chroma.sqlite3")):
        print(f"  [skip ingest] {chroma_dir} already exists", flush=True)
        return -1

    docs = []
    for fname in os.listdir(VIQUAD_DIR):
        if not fname.endswith(".txt"):
            continue
        loader = TextLoader(os.path.join(VIQUAD_DIR, fname), encoding="utf-8")
        pages = loader.load()
        for p in pages:
            p.page_content = _clean(p.page_content)
            p.metadata["source"] = fname
        docs.extend(pages)

    if splitter_type == "fixed":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" "
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

    chunks = splitter.split_documents(docs)
    print(f"  Chunks: {len(chunks)} (size={chunk_size}, overlap={chunk_overlap}, {splitter_type})", flush=True)

    os.makedirs(chroma_dir, exist_ok=True)
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(documents=chunks, embedding=emb, persist_directory=chroma_dir)
    return len(chunks)

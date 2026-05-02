import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.config import CHROMA_DIR, TOP_K

import sqlite3, json


def _load_chunks_from_chroma(chroma_dir: str) -> list[Document]:
    db_path = os.path.join(chroma_dir, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT string_value FROM embedding_metadata WHERE key='chroma:document'")
    rows = cur.fetchall()
    conn.close()
    return [Document(page_content=r[0]) for r in rows if r[0]]


class BM25Retriever:
    def __init__(self):
        self.docs = _load_chunks_from_chroma(CHROMA_DIR)
        tokenized = [doc.page_content.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_indices]

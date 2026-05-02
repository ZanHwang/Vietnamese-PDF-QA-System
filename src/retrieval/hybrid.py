import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain_core.documents import Document
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.config import TOP_K, RRF_K


def rrf_fusion(ranked_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        dense_docs = self.dense.retrieve(query, k=k)
        bm25_docs = self.bm25.retrieve(query, k=k)
        fused = rrf_fusion([dense_docs, bm25_docs])
        return fused[:k]

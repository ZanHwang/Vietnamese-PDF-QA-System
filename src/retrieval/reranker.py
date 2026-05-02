import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from sentence_transformers import CrossEncoder
from src.config import TOP_K, RERANKER_MODEL


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, docs: list, top_k: int = TOP_K) -> list:
        if not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]

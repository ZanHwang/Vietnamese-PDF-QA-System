import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import EMBEDDING_MODEL, CHROMA_DIR, TOP_K


class DenseRetriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    def retrieve(self, query: str, k: int = TOP_K) -> list:
        return self.vectorstore.similarity_search(query, k=k)

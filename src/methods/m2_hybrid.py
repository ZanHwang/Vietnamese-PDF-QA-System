import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.retrieval.hybrid import HybridRetriever
from src.generation.llm import LLMWrapper
from src.generation.prompts import M1_PROMPT
from src.config import TOP_K


class HybridRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMWrapper()

    def run(self, question: str, k: int = TOP_K) -> dict:
        docs = self.retriever.retrieve(question, k=k)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = M1_PROMPT.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
        }

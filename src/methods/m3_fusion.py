import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.hybrid import rrf_fusion
from src.generation.llm import LLMWrapper
from src.generation.prompts import M1_PROMPT, QUERY_EXPANSION_PROMPT
from src.config import TOP_K

N_VARIANTS = 3  # number of query variants to generate


class FusionRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMWrapper()

    def _expand_query(self, question: str) -> list[str]:
        prompt = QUERY_EXPANSION_PROMPT.format(n=N_VARIANTS, question=question)
        raw = self.llm.generate(prompt)
        variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return variants[:N_VARIANTS]

    def run(self, question: str, k: int = TOP_K) -> dict:
        variants = self._expand_query(question)
        all_queries = [question] + variants

        # Retrieve for each query variant
        all_ranked = [self.retriever.retrieve(q, k=k) for q in all_queries]

        # Second-level RRF across all variant result lists
        fused = rrf_fusion(all_ranked)[:k]

        context = "\n\n".join(d.page_content for d in fused)
        prompt = M1_PROMPT.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in fused],
            "query_variants": variants,
        }

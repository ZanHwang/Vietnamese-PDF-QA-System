import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.retrieval.dense import DenseRetriever
from src.generation.llm import LLMWrapper
from src.generation.prompts import (
    M1_PROMPT,
    SELFRAG_RETRIEVE_PROMPT,
    SELFRAG_REFLECT_PROMPT,
    SELFRAG_STRICT_PROMPT,
)
from src.config import TOP_K


class SelfRAG:
    def __init__(self):
        self.retriever = DenseRetriever()
        self.llm = LLMWrapper()

    def _needs_retrieval(self, question: str) -> bool:
        prompt = SELFRAG_RETRIEVE_PROMPT.format(question=question)
        result = self.llm.generate(prompt).strip().upper()
        return "NO" not in result  # default YES if ambiguous

    def _is_supported(self, context: str, answer: str) -> bool:
        prompt = SELFRAG_REFLECT_PROMPT.format(context=context[:800], answer=answer)
        result = self.llm.generate(prompt).strip().upper()
        return "NOT_SUPPORTED" not in result

    def run(self, question: str, k: int = TOP_K) -> dict:
        # Step 1: decide whether to retrieve
        retrieve = self._needs_retrieval(question)

        if retrieve:
            docs = self.retriever.retrieve(question, k=k)
            context = "\n\n".join(d.page_content for d in docs)
            contexts = [d.page_content for d in docs]
        else:
            context = ""
            contexts = []

        # Step 2: generate answer
        prompt = M1_PROMPT.format(context=context, question=question)
        answer = self.llm.generate(prompt)

        # Step 3: reflect — is answer supported?
        if context:
            supported = self._is_supported(context, answer)
            if not supported:
                # Regenerate with stricter prompt
                strict_prompt = SELFRAG_STRICT_PROMPT.format(
                    context=context, question=question
                )
                answer = self.llm.generate(strict_prompt)
        else:
            supported = True

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "retrieved": retrieve,
            "supported": supported,
        }

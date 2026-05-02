import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.retrieval.dense import DenseRetriever
from src.generation.llm import LLMWrapper
from src.generation.prompts import M1_PROMPT, CRAG_RELEVANCE_PROMPT, CRAG_FILTER_PROMPT
from src.config import TOP_K


class CRAG:
    def __init__(self):
        self.retriever = DenseRetriever()
        self.llm = LLMWrapper()

    def _evaluate_relevance(self, question: str, docs: list) -> list[str]:
        """Returns list of labels: CORRECT / AMBIGUOUS / INCORRECT for each doc."""
        chunks_text = "\n\n".join(
            f"Đoạn {i+1}: {d.page_content[:400]}" for i, d in enumerate(docs)
        )
        prompt = CRAG_RELEVANCE_PROMPT.format(question=question, chunks=chunks_text)
        raw = self.llm.generate(prompt)

        labels = []
        for line in raw.strip().splitlines():
            line = line.strip().upper()
            if "CORRECT" in line and "INCORRECT" not in line:
                labels.append("CORRECT")
            elif "INCORRECT" in line:
                labels.append("INCORRECT")
            elif "AMBIGUOUS" in line:
                labels.append("AMBIGUOUS")

        # Pad/trim to match doc count
        while len(labels) < len(docs):
            labels.append("AMBIGUOUS")
        return labels[:len(docs)]

    def _filter_chunk(self, question: str, chunk: str) -> str:
        """Extract relevant sentences from an AMBIGUOUS chunk."""
        prompt = CRAG_FILTER_PROMPT.format(question=question, chunk=chunk[:600])
        result = self.llm.generate(prompt).strip()
        if result.upper() == "NONE" or not result:
            return ""
        return result

    def run(self, question: str, k: int = TOP_K) -> dict:
        docs = self.retriever.retrieve(question, k=k)

        # Step 1: evaluate relevance of each chunk (1 LLM call)
        labels = self._evaluate_relevance(question, docs)

        correct_chunks = []
        ambiguous_chunks = []
        for doc, label in zip(docs, labels):
            if label == "CORRECT":
                correct_chunks.append(doc.page_content)
            elif label == "AMBIGUOUS":
                ambiguous_chunks.append(doc.page_content)

        # Step 2: filter AMBIGUOUS chunks (1 LLM call per ambiguous chunk)
        filtered = []
        for chunk in ambiguous_chunks:
            extracted = self._filter_chunk(question, chunk)
            if extracted:
                filtered.append(extracted)

        # Step 3: build final context
        final_chunks = correct_chunks + filtered
        if not final_chunks:
            # Fallback: use all retrieved docs
            final_chunks = [d.page_content for d in docs]

        context = "\n\n".join(final_chunks)
        prompt = M1_PROMPT.format(context=context, question=question)
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "contexts": final_chunks,
            "labels": labels,
        }

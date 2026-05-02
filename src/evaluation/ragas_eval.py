import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from src.config import EMBEDDING_MODEL, GROQ_MODEL

load_dotenv()


def _load_keys() -> list[str]:
    keys = []
    for i in range(1, 20):
        env = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"
        k = os.getenv(env, "").strip()
        if k:
            keys.append(k)
    return keys


def run_ragas_eval(results: list, n_samples: int = 50) -> dict:
    """RAGAS evaluation với key rotation — chia samples thành batch, mỗi batch dùng 1 key."""
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import LLMContextRecall, Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"  [RAGAS] Import failed: {e}")
        return {}

    keys = _load_keys()
    if not keys:
        print("  [RAGAS] Không tìm thấy GROQ_API_KEY nào trong .env")
        return {}

    subset = [r for r in results if r.get("answer") and r.get("ground_truth")][:n_samples]
    if not subset:
        return {}

    # Chia samples thành batches — mỗi batch dùng 1 key
    n_keys = len(keys)
    batch_size = max(1, (len(subset) + n_keys - 1) // n_keys)
    batches = [subset[i:i + batch_size] for i in range(0, len(subset), batch_size)]
    print(f"  [RAGAS] {len(subset)} samples → {len(batches)} batches × ~{batch_size} (keys={n_keys})", flush=True)

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    all_recall, all_faith, all_relev = [], [], []

    for b_idx, batch in enumerate(batches):
        key = keys[b_idx % n_keys]
        print(f"  [RAGAS] Batch {b_idx+1}/{len(batches)} ({len(batch)} samples, key {b_idx % n_keys + 1}/{n_keys})", flush=True)

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            for r in batch
        ]

        llm = LangchainLLMWrapper(ChatGroq(
            api_key=key,
            model=GROQ_MODEL,
            temperature=0,
        ))

        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=[LLMContextRecall(), Faithfulness(), AnswerRelevancy()],
                llm=llm,
                embeddings=embeddings,
            )
            df = result.to_pandas()
            all_recall.extend(df["context_recall"].dropna().tolist())
            all_faith.extend(df["faithfulness"].dropna().tolist())
            all_relev.extend(df["answer_relevancy"].dropna().tolist())
        except Exception as e:
            print(f"  [RAGAS] Batch {b_idx+1} failed: {str(e)[:120]}", flush=True)
            # Thử lại với key tiếp theo
            next_key = keys[(b_idx + 1) % n_keys]
            try:
                llm2 = LangchainLLMWrapper(ChatGroq(
                    api_key=next_key,
                    model=GROQ_MODEL,
                    temperature=0,
                ))
                result = evaluate(
                    dataset=EvaluationDataset(samples=samples),
                    metrics=[LLMContextRecall(), Faithfulness(), AnswerRelevancy()],
                    llm=llm2,
                    embeddings=embeddings,
                )
                df = result.to_pandas()
                all_recall.extend(df["context_recall"].dropna().tolist())
                all_faith.extend(df["faithfulness"].dropna().tolist())
                all_relev.extend(df["answer_relevancy"].dropna().tolist())
                print(f"  [RAGAS] Batch {b_idx+1} retry succeeded with key {(b_idx+1) % n_keys + 1}", flush=True)
            except Exception as e2:
                print(f"  [RAGAS] Batch {b_idx+1} retry also failed: {str(e2)[:120]}", flush=True)

    if not all_recall:
        return {}

    return {
        "context_recall":   round(sum(all_recall) / len(all_recall), 4),
        "faithfulness":     round(sum(all_faith)  / len(all_faith),  4),
        "answer_relevancy": round(sum(all_relev)  / len(all_relev),  4),
        "n_ragas_samples":  len(all_recall),
    }

import os, sys, json, time, io
sys.path.append(os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tqdm import tqdm
from src.methods.m4_crag import CRAG
from src.evaluation.metrics import compute_local_metrics
from src.evaluation.ragas_eval import run_ragas_eval
from src.config import QA_BENCHMARK_PATH, RESULTS_DIR

RAW_PATH   = os.path.join(RESULTS_DIR, "m4_raw.json")
BASELINE   = os.path.join(RESULTS_DIR, "m4_baseline.json")
SLEEP_SEC  = 6.0   # M4: up to 1 (relevance) + N_ambiguous (filter) + 1 (generate) calls
N_EVAL     = None
N_RAGAS    = 50
SAVE_EVERY = 10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(QA_BENCHMARK_PATH, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    if N_EVAL:
        qa_pairs = qa_pairs[:N_EVAL]

    existing = []
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    done = {r["question"] for r in existing}
    print(f"Benchmark: {len(qa_pairs)} pairs | Done: {len(done)} | Remaining: {len(qa_pairs) - len(done)}", flush=True)

    rag = CRAG()
    results = list(existing)
    todo = [q for q in qa_pairs if q["question"] not in done]
    errors = 0

    for item in tqdm(todo, desc="M4 Eval", file=sys.stderr):
        try:
            out = rag.run(item["question"])
            results.append({
                "question":     item["question"],
                "answer":       out["answer"],
                "ground_truth": item["ground_truth"],
                "contexts":     out["contexts"],
                "gold_context": item.get("context", ""),
                "labels":       out.get("labels", []),
            })
            if len(results) % SAVE_EVERY == 0:
                _save(RAW_PATH, results)
                print(f"\n  [checkpoint] {len(results)} saved", flush=True)
            time.sleep(SLEEP_SEC)
        except KeyboardInterrupt:
            print("\nInterrupted — saving progress...", flush=True)
            break
        except Exception as e:
            errors += 1
            print(f"\n  [ERROR #{errors}] {str(e)[:150]}", flush=True)
            time.sleep(15)

    _save(RAW_PATH, results)
    print(f"\nDone. Saved {len(results)} results (errors: {errors}) → {RAW_PATH}", flush=True)

    if not results:
        print("No results to evaluate.")
        return

    local = compute_local_metrics(results)
    print(f"\n{'='*45}")
    print(f"M4 CRAG — Local Metrics (n={local['n_samples']})")
    print(f"{'='*45}")
    print(f"  Exact Match       : {local['exact_match']:.4f}")
    print(f"  ROUGE-L           : {local['rouge_l']:.4f}")
    if "context_hit_rate" in local:
        print(f"  Context Hit Rate  : {local['context_hit_rate']:.4f}  (>=30% token overlap)")

    print(f"\nRunning RAGAS on {min(N_RAGAS, len(results))} samples...")
    ragas = run_ragas_eval(results, n_samples=N_RAGAS)
    if ragas:
        print(f"  Context Recall    : {ragas.get('context_recall', 'N/A')}")
        print(f"  Faithfulness      : {ragas.get('faithfulness', 'N/A')}")
        print(f"  Answer Relevancy  : {ragas.get('answer_relevancy', 'N/A')}")
    else:
        print("  RAGAS skipped (check Groq quota or RAGAS logs above)")

    baseline = {
        "method": "M4_CRAG",
        "config": {
            "top_k": 4,
            "chunk_size": 400,
            "chunk_overlap": 120,
            "embedding": "paraphrase-multilingual-MiniLM-L12-v2",
            "retrieval": "Dense + Relevance Filter (CORRECT/AMBIGUOUS/INCORRECT)",
        },
        "metrics": {**local, **(ragas or {})},
    }
    _save(BASELINE, baseline)
    print(f"\nBaseline saved → {BASELINE}", flush=True)


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()

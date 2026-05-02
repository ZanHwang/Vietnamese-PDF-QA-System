import os, sys, json, time, random, io
sys.path.append(os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import QA_BENCHMARK_PATH, RESULTS_DIR, EMBEDDING_MODEL, CHROMA_DIR, TOP_K
from src.generation.llm import LLMWrapper
from src.generation.prompts import M1_PROMPT, QUERY_EXPANSION_PROMPT
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever, rrf_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.ablation.ingest_variant import build_variant_chroma
from src.evaluation.metrics import compute_local_metrics

ABLATION_N       = 100
SLEEP_SEC        = 1.5
SLEEP_SEC_MULTI  = 3.0   # A5 makes 2 LLM calls per question
ABLATION_DIR     = os.path.join(RESULTS_DIR, "ablation")
CHROMA_VAR_DIR   = "data/ablation_chroma"

# (name, group, config)
EXPERIMENTS = [
    # A1: chunking strategy — keep overlap at 30% of chunk_size
    ("a1_fixed_200",     "A1", {"rebuild": True, "splitter": "fixed",     "chunk_size": 200, "chunk_overlap": 60}),
    ("a1_fixed_400",     "A1", {"rebuild": True, "splitter": "fixed",     "chunk_size": 400, "chunk_overlap": 120}),
    ("a1_fixed_800",     "A1", {"rebuild": True, "splitter": "fixed",     "chunk_size": 800, "chunk_overlap": 240}),
    ("a1_recursive_200", "A1", {"rebuild": True, "splitter": "recursive", "chunk_size": 200, "chunk_overlap": 60}),
    ("a1_recursive_800", "A1", {"rebuild": True, "splitter": "recursive", "chunk_size": 800, "chunk_overlap": 240}),
    # A2: overlap — keep chunk_size=400, vary overlap
    ("a2_overlap_0",     "A2", {"rebuild": True, "splitter": "recursive", "chunk_size": 400, "chunk_overlap": 0}),
    ("a2_overlap_40",    "A2", {"rebuild": True, "splitter": "recursive", "chunk_size": 400, "chunk_overlap": 40}),
    ("a2_overlap_80",    "A2", {"rebuild": True, "splitter": "recursive", "chunk_size": 400, "chunk_overlap": 80}),
    # A3: retrieval type (Dense=M1 baseline, Hybrid=M2 baseline)
    ("a3_bm25_only",     "A3", {"retriever": "bm25"}),
    # A4: top-K (baseline K=4 = M1)
    ("a4_k1",            "A4", {"top_k": 1}),
    ("a4_k3",            "A4", {"top_k": 3}),
    ("a4_k5",            "A4", {"top_k": 5}),
    ("a4_k10",           "A4", {"top_k": 10}),
    # A5: query expansion (Original=M1, Dense+multiquery=new, Hybrid+fusion=M3)
    ("a5_multiquery",    "A5", {"query_mode": "multiquery"}),
    # A6: reranking (no rerank=M2 baseline, cross-encoder=new)
    ("a6_cross_encoder", "A6", {"reranker": "cross_encoder"}),
]


# ── Pipeline factories ────────────────────────────────────────────────────────

def make_dense_pipeline(k: int, chroma_dir: str = None):
    if chroma_dir:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vs = Chroma(persist_directory=chroma_dir, embedding_function=emb)
        retrieve = lambda q: vs.similarity_search(q, k=k)
    else:
        _r = DenseRetriever()
        retrieve = lambda q: _r.retrieve(q, k=k)
    llm = LLMWrapper()

    def pipeline(question):
        docs = retrieve(question)
        ctx = "\n\n".join(d.page_content for d in docs)
        return {"answer": llm.generate(M1_PROMPT.format(context=ctx, question=question)),
                "contexts": [d.page_content for d in docs]}
    return pipeline


def make_bm25_pipeline(k: int):
    retriever = BM25Retriever()
    llm = LLMWrapper()

    def pipeline(question):
        docs = retriever.retrieve(question, k=k)
        ctx = "\n\n".join(d.page_content for d in docs)
        return {"answer": llm.generate(M1_PROMPT.format(context=ctx, question=question)),
                "contexts": [d.page_content for d in docs]}
    return pipeline


def make_multiquery_pipeline():
    retriever = DenseRetriever()
    llm = LLMWrapper()

    def pipeline(question):
        raw = llm.generate(QUERY_EXPANSION_PROMPT.format(n=3, question=question))
        variants = [l.strip() for l in raw.strip().splitlines() if l.strip()][:3]
        all_results = [retriever.retrieve(q, k=TOP_K) for q in [question] + variants]
        fused = rrf_fusion(all_results)[:TOP_K]
        ctx = "\n\n".join(d.page_content for d in fused)
        return {"answer": llm.generate(M1_PROMPT.format(context=ctx, question=question)),
                "contexts": [d.page_content for d in fused]}
    return pipeline


def make_reranker_pipeline():
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    llm = LLMWrapper()

    def pipeline(question):
        docs = retriever.retrieve(question, k=TOP_K * 3)
        reranked = reranker.rerank(question, docs, top_k=TOP_K)
        ctx = "\n\n".join(d.page_content for d in reranked)
        return {"answer": llm.generate(M1_PROMPT.format(context=ctx, question=question)),
                "contexts": [d.page_content for d in reranked]}
    return pipeline


# ── Eval loop ────────────────────────────────────────────────────────────────

def run_eval(pipeline_fn, qa_pairs, sleep_sec: float) -> list:
    results = []
    errors = 0
    for item in tqdm(qa_pairs, file=sys.stderr):
        try:
            out = pipeline_fn(item["question"])
            results.append({
                "question":     item["question"],
                "answer":       out["answer"],
                "ground_truth": item["ground_truth"],
                "contexts":     out["contexts"],
                "gold_context": item.get("context", ""),
            })
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            print("\nInterrupted — saving progress...", flush=True)
            break
        except Exception as e:
            errors += 1
            print(f"\n  [ERROR] {str(e)[:120]}", flush=True)
            time.sleep(10)
    if errors:
        print(f"  Total errors: {errors}", flush=True)
    return results


# ── Summary ──────────────────────────────────────────────────────────────────

def _print_summary():
    def load(path):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    # Load M1/M2/M3 baselines for reference (n=400)
    b = {m: load(os.path.join(RESULTS_DIR, f"{m}_baseline.json")) for m in ["m1", "m2", "m3"]}

    def bmet(m, k):
        return b.get(m, {}).get("metrics", {}).get(k, float("nan"))

    ablation = {}
    for name, _, _ in EXPERIMENTS:
        d = load(os.path.join(ABLATION_DIR, f"{name}.json"))
        if d:
            ablation[name] = d

    print(f"\n{'='*65}", flush=True)
    print("ABLATION SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)

    groups = {}
    for name, group, cfg in EXPERIMENTS:
        groups.setdefault(group, []).append((name, cfg))

    baseline_rows = {
        "A1": [("baseline: recursive+400 (M1)", bmet("m1","rouge_l"), bmet("m1","context_hit_rate"), bmet("m1","exact_match"))],
        "A2": [("baseline: overlap=120/30% (M1)", bmet("m1","rouge_l"), bmet("m1","context_hit_rate"), bmet("m1","exact_match"))],
        "A3": [("dense only (M1)",  bmet("m1","rouge_l"), bmet("m1","context_hit_rate"), bmet("m1","exact_match")),
               ("hybrid (M2)",      bmet("m2","rouge_l"), bmet("m2","context_hit_rate"), bmet("m2","exact_match"))],
        "A4": [("K=4 (M1)",         bmet("m1","rouge_l"), bmet("m1","context_hit_rate"), bmet("m1","exact_match"))],
        "A5": [("original (M1)",    bmet("m1","rouge_l"), bmet("m1","context_hit_rate"), bmet("m1","exact_match")),
               ("RAG-Fusion (M3)",  bmet("m3","rouge_l"), bmet("m3","context_hit_rate"), bmet("m3","exact_match"))],
        "A6": [("no rerank (M2)",   bmet("m2","rouge_l"), bmet("m2","context_hit_rate"), bmet("m2","exact_match"))],
    }

    for grp, items in groups.items():
        print(f"\n[{grp}]  {'Name':<26} {'ROUGE-L':>8} {'CHR':>8} {'EM':>8}", flush=True)
        print(f"      {'─'*26} {'─'*8} {'─'*8} {'─'*8}", flush=True)
        for label, rl, chr_, em in baseline_rows.get(grp, []):
            print(f"      {label:<26} {rl:>8.4f} {chr_:>8.4f} {em:>8.4f}  ←baseline", flush=True)
        for name, _ in items:
            if name in ablation:
                m = ablation[name]["metrics"]
                rl   = m.get("rouge_l", float("nan"))
                chr_ = m.get("context_hit_rate", float("nan"))
                em   = m.get("exact_match", float("nan"))
                print(f"      {name:<26} {rl:>8.4f} {chr_:>8.4f} {em:>8.4f}", flush=True)
            else:
                print(f"      {name:<26} {'(not run)':>8}", flush=True)
    sys.stdout.flush()

    _save(os.path.join(ABLATION_DIR, "summary.json"),
          {"ablation": {n: ablation[n] for n in ablation}})
    print(f"\nSummary saved → {os.path.join(ABLATION_DIR, 'summary.json')}", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(ABLATION_DIR, exist_ok=True)
    os.makedirs(CHROMA_VAR_DIR, exist_ok=True)

    with open(QA_BENCHMARK_PATH, encoding="utf-8") as f:
        qa_all = json.load(f)
    random.seed(42)
    qa_sample = random.sample(qa_all, ABLATION_N)
    print(f"Ablation: {ABLATION_N} questions (seed=42)", flush=True)

    for name, group, cfg in EXPERIMENTS:
        out_path = os.path.join(ABLATION_DIR, f"{name}.json")
        if os.path.exists(out_path):
            print(f"[skip] {name}", flush=True)
            continue

        print(f"\n{'='*55}", flush=True)
        print(f"[{group}] {name}", flush=True)
        sys.stdout.flush()

        # Build pipeline
        if cfg.get("rebuild"):
            chroma_dir = os.path.join(CHROMA_VAR_DIR, name)
            build_variant_chroma(chroma_dir, cfg["chunk_size"], cfg["chunk_overlap"], cfg["splitter"])
            pipeline = make_dense_pipeline(TOP_K, chroma_dir=chroma_dir)
            sleep = SLEEP_SEC
        elif cfg.get("retriever") == "bm25":
            pipeline = make_bm25_pipeline(TOP_K)
            sleep = SLEEP_SEC
        elif "top_k" in cfg:
            pipeline = make_dense_pipeline(cfg["top_k"])
            sleep = SLEEP_SEC
        elif cfg.get("query_mode") == "multiquery":
            pipeline = make_multiquery_pipeline()
            sleep = SLEEP_SEC_MULTI
        elif cfg.get("reranker") == "cross_encoder":
            pipeline = make_reranker_pipeline()
            sleep = SLEEP_SEC
        else:
            print(f"  Unknown config — skip", flush=True)
            continue

        results = run_eval(pipeline, qa_sample, sleep_sec=sleep)
        if not results:
            print(f"  No results — skip", flush=True)
            continue

        metrics = compute_local_metrics(results)
        _save(out_path, {"name": name, "group": group, "config": cfg, "metrics": metrics})
        print(f"  EM={metrics['exact_match']:.4f}  ROUGE-L={metrics['rouge_l']:.4f}"
              f"  CHR={metrics.get('context_hit_rate', 'N/A')}", flush=True)
        sys.stdout.flush()

    _print_summary()
    print("\nAblation complete.", flush=True)


if __name__ == "__main__":
    main()

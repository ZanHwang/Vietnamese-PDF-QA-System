import os, sys, json, io
sys.path.append(os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.evaluation.metrics import rouge_l, exact_match, context_hit_rate
from src.config import RESULTS_DIR

METHODS = ["m1", "m2", "m3", "m4", "m5"]

# Thresholds
CORRECT_RL   = 0.3   # ROUGE-L >= this → correct
FAIL_RL      = 0.1   # ROUGE-L < this AND CHR > 0 → generation fail
CHR_THRESH   = 0.3   # same threshold as metrics.py

# Pronouns/references that indicate ambiguous queries
AMBIGUOUS_WORDS = {"này", "đây", "đó", "kia", "đấy", "ấy", "trên", "sau", "trước",
                   "nó", "họ", "chúng", "đây", "nơi", "việc", "điều"}


def classify(item: dict) -> str:
    rl   = rouge_l(item["answer"], item["ground_truth"])
    chr_ = context_hit_rate(item["contexts"], item.get("gold_context", ""))
    if rl >= CORRECT_RL:
        return "correct"
    if chr_ == 0.0:
        return "retrieval_fail"
    if chr_ >= CHR_THRESH and rl < FAIL_RL:
        return "generation_fail"
    return "partial"


def has_ambiguous_ref(question: str) -> bool:
    tokens = set(question.lower().split())
    return bool(tokens & AMBIGUOUS_WORDS)


def analyze_method(method: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, f"{method}_raw.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    buckets: dict[str, list] = {
        "correct": [], "retrieval_fail": [], "generation_fail": [], "partial": []
    }
    ambiguous_count = 0

    for item in results:
        cat = classify(item)
        buckets[cat].append({
            "question": item["question"][:100],
            "answer":   item["answer"][:100],
            "gt":       item["ground_truth"][:100],
            "rl":       round(rouge_l(item["answer"], item["ground_truth"]), 3),
            "chr":      round(context_hit_rate(item["contexts"], item.get("gold_context", "")), 3),
        })
        if has_ambiguous_ref(item["question"]):
            ambiguous_count += 1

    n = len(results)
    stats = {k: {"count": len(v), "pct": round(len(v) / n * 100, 1)}
             for k, v in buckets.items()}
    stats["ambiguous_query"] = {"count": ambiguous_count,
                                 "pct": round(ambiguous_count / n * 100, 1)}

    return {
        "method": method.upper(),
        "n": n,
        "stats": stats,
        "examples": {k: v[:3] for k, v in buckets.items()},
    }


def main():
    analysis = {}

    print("=" * 62, flush=True)
    print("ERROR ANALYSIS — M1 to M5", flush=True)
    print("=" * 62, flush=True)

    for method in METHODS:
        result = analyze_method(method)
        if not result:
            print(f"\n{method.upper()}: raw file not found — skipping", flush=True)
            continue
        analysis[method] = result

        n = result["n"]
        print(f"\n[{result['method']}]  n={n}", flush=True)
        print(f"  {'Category':<20} {'Count':>6} {'%':>7}", flush=True)
        print(f"  {'─'*20} {'─'*6} {'─'*7}", flush=True)
        for cat, s in result["stats"].items():
            print(f"  {cat:<20} {s['count']:>6} {s['pct']:>6.1f}%", flush=True)
        sys.stdout.flush()

    if not analysis:
        print("No data found.", flush=True)
        return

    # Cross-method comparison table
    CATS = ["correct", "retrieval_fail", "generation_fail", "partial", "ambiguous_query"]
    present = [m for m in METHODS if m in analysis]

    print(f"\n\n{'='*62}", flush=True)
    print("CROSS-METHOD COMPARISON  (% per error category)", flush=True)
    print(f"{'─'*62}", flush=True)
    header = f"  {'Category':<20}" + "".join(f" {m.upper():>7}" for m in present)
    print(header, flush=True)
    print(f"  {'─'*20}" + "".join(f" {'─'*7}" for _ in present), flush=True)
    for cat in CATS:
        row = f"  {cat:<20}"
        for m in present:
            pct = analysis[m]["stats"].get(cat, {}).get("pct", 0)
            row += f" {pct:>6.1f}%"
        print(row, flush=True)
    sys.stdout.flush()

    # Key observations
    print(f"\n\n{'='*62}", flush=True)
    print("KEY OBSERVATIONS", flush=True)
    print(f"{'─'*62}", flush=True)

    for method in present:
        d = analysis[method]
        s = d["stats"]
        best_cat = max(["retrieval_fail", "generation_fail", "partial"],
                       key=lambda c: s[c]["count"])
        print(f"  {d['method']}: correct={s['correct']['pct']}%  "
              f"main_fail={best_cat} ({s[best_cat]['pct']}%)", flush=True)
    sys.stdout.flush()

    # Save
    out_path = os.path.join(RESULTS_DIR, "error_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

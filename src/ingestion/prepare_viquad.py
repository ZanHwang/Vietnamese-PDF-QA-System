import os, sys, json, random
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset
from src.config import VIQUAD_DIR, QA_BENCHMARK_PATH

N_PAIRS = 400
SEED = 42
random.seed(SEED)


def main():
    print("Downloading UIT-ViQuAD 2.0 ...")
    ds = load_dataset("taidng/UIT-ViQuAD2.0")

    # Dùng validation set (sạch, không overlap với train)
    split = ds["validation"]
    print(f"Validation set size: {len(split)}")

    # Lọc câu có đáp án (loại unanswerable)
    answerable = [
        item for item in split
        if item["answers"]["text"]
    ]
    print(f"Answerable pairs: {len(answerable)}")

    # Sample N_PAIRS ngẫu nhiên
    sampled = random.sample(answerable, min(N_PAIRS, len(answerable)))

    # Chuẩn bị QA benchmark
    qa_pairs = []
    for item in sampled:
        qa_pairs.append({
            "question":     item["question"],
            "ground_truth": item["answers"]["text"][0],
            "context":      item["context"],     # gold context cho RAGAS
            "source_id":    item["id"],
            "title":        item.get("title", ""),
        })

    with open(QA_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(qa_pairs)} QA pairs → {QA_BENCHMARK_PATH}")

    # Lưu unique contexts ra file txt để ingest vào ChromaDB
    os.makedirs(VIQUAD_DIR, exist_ok=True)
    seen = set()
    ctx_count = 0
    for item in sampled:
        ctx = item["context"]
        if ctx in seen:
            continue
        seen.add(ctx)
        fname = os.path.join(VIQUAD_DIR, f"ctx_{ctx_count:04d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(ctx)
        ctx_count += 1

    print(f"Saved {ctx_count} unique contexts → {VIQUAD_DIR}/")
    print(f"\nStats:")
    print(f"  QA pairs : {len(qa_pairs)}")
    print(f"  Unique contexts: {ctx_count}")
    avg_q = sum(len(q['question']) for q in qa_pairs) / len(qa_pairs)
    avg_a = sum(len(q['ground_truth']) for q in qa_pairs) / len(qa_pairs)
    print(f"  Avg question len : {avg_q:.0f} chars")
    print(f"  Avg answer len   : {avg_a:.0f} chars")


if __name__ == "__main__":
    main()

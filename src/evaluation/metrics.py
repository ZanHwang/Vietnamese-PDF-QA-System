import re, unicodedata


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lcs(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if x[i-1] == y[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def rouge_l(pred: str, gold: str) -> float:
    p_tok = _normalize(pred).split()
    g_tok = _normalize(gold).split()
    if not p_tok or not g_tok:
        return 0.0
    lcs = _lcs(p_tok, g_tok)
    prec = lcs / len(p_tok)
    rec = lcs / len(g_tok)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def context_hit_rate(retrieved: list, gold_context: str, threshold: float = 0.3) -> float:
    """1.0 if any retrieved chunk shares ≥threshold token overlap with gold context."""
    gold_tokens = set(_normalize(gold_context).split())
    if not gold_tokens:
        return 0.0
    for ctx in retrieved:
        ctx_tokens = set(_normalize(ctx).split())
        overlap = len(ctx_tokens & gold_tokens) / len(gold_tokens)
        if overlap >= threshold:
            return 1.0
    return 0.0


def compute_local_metrics(results: list) -> dict:
    em, rl, chr_ = [], [], []
    for r in results:
        em.append(exact_match(r["answer"], r["ground_truth"]))
        rl.append(rouge_l(r["answer"], r["ground_truth"]))
        if r.get("gold_context"):
            chr_.append(context_hit_rate(r["contexts"], r["gold_context"]))
    metrics = {
        "exact_match": round(sum(em) / len(em), 4),
        "rouge_l": round(sum(rl) / len(rl), 4),
        "n_samples": len(results),
    }
    if chr_:
        metrics["context_hit_rate"] = round(sum(chr_) / len(chr_), 4)
    return metrics

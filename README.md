# Vietnamese RAG Q&A System

Hệ thống hỏi đáp tài liệu PDF tiếng Việt sử dụng Retrieval-Augmented Generation (RAG). Dự án so sánh 5 phương pháp RAG từ cơ bản đến nâng cao, kết hợp ablation study 6 thí nghiệm và đánh giá bằng RAGAS metrics.

---

## Các phương pháp RAG

| # | Phương pháp | Kỹ thuật |
|---|-------------|----------|
| M1 | **Naive RAG** | Dense Retrieval + Cosine Similarity |
| M2 | **Hybrid Search** | BM25 + Dense + RRF fusion |
| M3 | **RAG-Fusion** | Multi-query expansion + Hybrid + RRF×2 |
| M4 | **CRAG** | Retrieval Evaluator (CORRECT/AMBIGUOUS/INCORRECT) + Self-correction |
| M5 | **Self-RAG Lite** | Adaptive retrieval decision + Reflection + Strict regeneration |

## Kết quả chính (n=400, UIT-ViQuAD 2.0)

| Method | ROUGE-L | Context Hit Rate | Faithfulness | Answer Relevancy |
|--------|---------|-----------------|--------------|-----------------|
| M1 Naive RAG | 0.453 | 0.823 | **0.867** | 0.487 |
| M2 Hybrid Search | 0.446 | **0.910** | 0.733 | **0.472** |
| M3 RAG-Fusion | 0.446 | **0.910** | 0.529 | 0.360 |
| M4 CRAG | 0.403 | 0.748 | 0.683 | 0.393 |
| M5 Self-RAG Lite | 0.247 | 0.423 | 0.188 | 0.260 |

**Ablation A6** (Hybrid + multilingual cross-encoder reranker): ROUGE-L = **0.537**, Context Hit Rate = **0.960** — kết quả tốt nhất toàn dự án.

---

## Tech Stack

- **LLM**: `llama-3.3-70b-versatile` via Groq API (free tier, 7-key rotation)
- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (local, HuggingFace)
- **Reranker**: `nreimers/mmarco-mMiniLMv2-L12-H384-v1` (multilingual cross-encoder)
- **Vector DB**: ChromaDB (local, persistent)
- **Sparse Retrieval**: rank_bm25
- **Evaluation**: RAGAS (Faithfulness, Answer Relevancy, Context Recall)
- **Backend**: FastAPI
- **Frontend**: React + Vite
- **Framework**: LangChain + Python 3.10+

---

## Cấu trúc thư mục

```
BTL_RAG/
├── src/
│   ├── config.py              # Cấu hình chung (model names, paths, hyperparams)
│   ├── ingestion/             # Load, clean, chunk PDF → ChromaDB
│   ├── retrieval/             # Dense, BM25, Hybrid, Reranker
│   ├── generation/            # LLM wrapper (Groq, key rotation), Prompts
│   ├── methods/               # M1–M5 pipeline
│   ├── evaluation/            # RAGAS metrics, ROUGE-L, Exact Match
│   └── ablation/              # Ingestion variant cho ablation study
├── backend/
│   └── main.py                # FastAPI: /ask, /upload, /metrics, /health
├── frontend/
│   └── src/App.jsx            # React UI: chat, upload PDF, so sánh methods
├── data/
│   ├── pdfs/                  # PDF tiếng Việt (dev/test)
│   ├── viquad/                # UIT-ViQuAD 2.0 context passages (287 files)
│   └── qa_benchmark.json      # 400 cặp QA từ ViQuAD (question, context, answer)
├── run_m1_eval.py             # Chạy đánh giá M1
├── run_m2_eval.py             # Chạy đánh giá M2
├── run_m3_eval.py             # Chạy đánh giá M3
├── run_m4_eval.py             # Chạy đánh giá M4
├── run_m5_eval.py             # Chạy đánh giá M5
├── run_ablation.py            # Chạy toàn bộ ablation A1–A6
├── run_error_analysis.py      # Phân loại lỗi cho M1–M5
├── requirements.txt
└── .env.example               # Template API keys
```

---

## Cài đặt

### 1. Clone và tạo virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/BTL_RAG.git
cd BTL_RAG
python -m venv rag_env
# Windows:
rag_env\Scripts\activate
# Linux/Mac:
source rag_env/bin/activate
pip install -r requirements.txt
```

### 2. Cấu hình API keys

```bash
cp .env.example .env
```

Mở `.env` và điền Groq API key (đăng ký miễn phí tại [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=gsk_your_key_here
# Thêm nhiều key để tránh rate limit (tùy chọn):
GROQ_API_KEY_2=gsk_your_second_key
```

### 3. Build vector database

```bash
python -m src.ingestion.ingest
```

Lệnh này đọc `data/viquad/*.txt` → chunk → embed → lưu vào `data/chroma_db/`.

---

## Chạy đánh giá

```bash
# Đánh giá từng phương pháp (kết quả lưu vào results/)
python run_m1_eval.py
python run_m2_eval.py
python run_m3_eval.py
python run_m4_eval.py
python run_m5_eval.py

# Ablation study A1–A6
python run_ablation.py

# Error analysis
python run_error_analysis.py
```

---

## Web Demo

### Backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API endpoints:
- `POST /ask` — hỏi đáp với method M1–M5
- `POST /upload` — upload và index PDF mới
- `GET /metrics` — xem kết quả benchmark
- `GET /health` — kiểm tra trạng thái

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

Truy cập `http://localhost:5173` để dùng giao diện web.

---

## Ablation Study

| Exp | Biến số | Kết quả tốt nhất |
|-----|---------|-----------------|
| A1 | Chunk size: 200 / 400 / 800 | **400** (ROUGE-L=0.410) |
| A2 | Chunk overlap: 0% / 10% / 20% | **20%** (ROUGE-L=0.404) |
| A3 | Retrieval: Dense / BM25 / Hybrid | BM25 ≈ Dense; **Hybrid** tốt nhất tổng thể |
| A4 | Top-K: 1 / 3 / 5 / 10 | **K=5** (ROUGE-L=0.469) |
| A5 | Query: Original / Multi-query | Multi-query cải thiện nhẹ |
| A6 | Reranker: Không / Cross-Encoder | **Cross-Encoder** (ROUGE-L=0.537, +31%) |

---

## Dataset

**UIT-ViQuAD 2.0** — bộ dữ liệu QA tiếng Việt chuẩn quốc tế  
- 400 cặp (question, context, answer) được trích từ HuggingFace `taidng/UIT-ViQuAD2.0`
- 287 context passages tiếng Việt đã được index vào ChromaDB

---

## Tham khảo

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Rackauckas (2024) — *RAG-Fusion*
- Yan et al. (2024) — *CRAG: Corrective Retrieval Augmented Generation*
- Asai et al. (2023) — *Self-RAG: Learning to Retrieve, Generate, and Critique*

# Vietnamese PDF Q&A System — RAG Comparison

Hệ thống hỏi đáp tài liệu PDF tiếng Việt sử dụng Retrieval-Augmented Generation (RAG). Dự án triển khai và so sánh 5 phương pháp RAG từ cơ bản đến nâng cao trên bộ dữ liệu **UIT-ViQuAD 2.0**, kết hợp ablation study 6 thí nghiệm và đánh giá bằng RAGAS metrics. Giao diện web cho phép upload PDF và hỏi đáp với bất kỳ phương pháp nào.

---

## Phương pháp RAG

| # | Tên | Kỹ thuật |
|---|-----|----------|
| M1 | Naive RAG | Dense Retrieval + Cosine Similarity |
| M2 | Hybrid Search | BM25 + Dense + RRF fusion |
| M3 | RAG-Fusion | Multi-query expansion + Hybrid + RRF×2 |
| M4 | CRAG | Relevance evaluator (CORRECT/AMBIGUOUS/INCORRECT) + lọc context |
| M5 | Self-RAG Lite | Adaptive retrieval decision + reflection + strict regeneration |

---

## Kết quả (n=400, UIT-ViQuAD 2.0)

| Method | ROUGE-L | Context Hit Rate | Faithfulness | Answer Relevancy |
|--------|:-------:|:----------------:|:------------:|:----------------:|
| M1 Naive RAG | 0.453 | 0.823 | **0.867** | 0.487 |
| M2 Hybrid Search | 0.446 | **0.910** | 0.733 | 0.472 |
| M3 RAG-Fusion | 0.446 | **0.910** | 0.529 | 0.360 |
| M4 CRAG | 0.403 | 0.748 | 0.683 | 0.393 |
| M5 Self-RAG Lite | 0.247 | 0.423 | 0.188 | 0.260 |

Ablation tốt nhất — **A6 Hybrid + Cross-Encoder Reranker**: ROUGE-L = 0.537, Context Hit Rate = 0.960

---

## Tech Stack

| Thành phần | Chi tiết |
|-----------|---------|
| LLM | `llama-3.3-70b-versatile` — Groq API (free tier) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` — local, HuggingFace |
| Reranker | `nreimers/mmarco-mMiniLMv2-L12-H384-v1` — multilingual cross-encoder |
| Vector DB | ChromaDB (local, persistent) |
| Sparse retrieval | rank_bm25 |
| Evaluation | RAGAS + ROUGE-L + Exact Match |
| Backend | FastAPI |
| Frontend | React + Vite |
| Framework | LangChain, Python 3.10+ |

---

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/ZanHwang/Vietnamese-PDF-QA-System.git
cd Vietnamese-PDF-QA-System
```

### 2. Tạo virtual environment

```bash
python -m venv rag_env

# Windows
rag_env\Scripts\activate

# Linux / Mac
source rag_env/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình API key

```bash
cp .env.example .env
```

Mở `.env`, điền Groq API key (đăng ký miễn phí tại [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=gsk_...
```

Thêm nhiều key để tránh rate limit (khuyến nghị khi chạy eval):

```
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
```

### 4. Build vector database

```bash
python -m src.ingestion.ingest
```

Lệnh này đọc toàn bộ file trong `data/viquad/` và `data/pdfs/`, chunk, embed rồi lưu vào `data/chroma_db/`. Chỉ cần chạy một lần.

---

## Chạy đánh giá

Mỗi script tự động lưu kết quả vào `results/` và hỗ trợ resume nếu bị ngắt giữa chừng.

```bash
python run_m1_eval.py   # Naive RAG
python run_m2_eval.py   # Hybrid Search
python run_m3_eval.py   # RAG-Fusion
python run_m4_eval.py   # CRAG
python run_m5_eval.py   # Self-RAG Lite
```

Ablation study (A1–A6):

```bash
python run_ablation.py
```

Error analysis:

```bash
python run_error_analysis.py
```

---

## Web Demo

### Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Endpoints:

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/ask` | POST | Hỏi đáp — chọn method M1–M5 |
| `/upload` | POST | Upload và index PDF mới |
| `/metrics` | GET | Xem kết quả benchmark |
| `/health` | GET | Kiểm tra trạng thái server |

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Truy cập `http://localhost:5173`.

Tính năng:
- Upload PDF và index tự động
- Nhập câu hỏi, chọn phương pháp RAG
- Xem câu trả lời kèm source chunks
- So sánh cùng câu hỏi trên 2 phương pháp khác nhau
- Xem bảng metrics benchmark

---

## Cấu trúc thư mục

```
├── src/
│   ├── config.py              # Cấu hình chung (model, paths, hyperparams)
│   ├── ingestion/             # Load PDF/txt, clean, chunk, index ChromaDB
│   ├── retrieval/             # Dense, BM25, Hybrid, CrossEncoder Reranker
│   ├── generation/            # LLM wrapper (Groq + key rotation), Prompts
│   ├── methods/               # M1–M5 pipeline
│   ├── evaluation/            # RAGAS, ROUGE-L, Exact Match, Context Hit Rate
│   └── ablation/              # Ingest variant cho ablation chunking
├── backend/main.py            # FastAPI server
├── frontend/src/App.jsx       # React UI
├── data/
│   ├── pdfs/                  # PDF tiếng Việt
│   ├── viquad/                # UIT-ViQuAD 2.0 context passages (287 files)
│   └── qa_benchmark.json      # 400 cặp QA (question, context, answer)
├── run_m1_eval.py ~ run_m5_eval.py
├── run_ablation.py
├── run_error_analysis.py
└── requirements.txt
```

---

## Dataset

**UIT-ViQuAD 2.0** — bộ dữ liệu Reading Comprehension tiếng Việt  
Nguồn: HuggingFace [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)  
Subset dùng trong dự án: 400 cặp QA, 287 context passages tiếng Việt

---

## Tài liệu tham khảo

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Rackauckas (2024) — *RAG-Fusion: A New Take on Retrieval-Augmented Generation*
- Yan et al. (2024) — *CRAG: Corrective Retrieval Augmented Generation*
- Asai et al. (2023) — *Self-RAG: Learning to Retrieve, Generate, and Critique*

# RAG-Based PDF Q&A System — Vietnamese Documents

## Hướng dẫn đọc khi bắt đầu session mới
Mỗi khi bắt đầu làm việc, Claude PHẢI đọc 2 file sau trước khi làm bất cứ điều gì:
1. `CLAUDE.md` (file này) — pipeline, tech stack, quy tắc
2. `ketqua.txt` — tiến độ thực tế, kết quả đã đạt, vấn đề đã gặp

Đọc xong mới được đề xuất hoặc thực hiện bất kỳ bước nào.

## Mục tiêu
Hệ thống hỏi đáp PDF tiếng Việt dùng RAG, so sánh 5 phương pháp từ Naive đến Advanced.

## Tech Stack
- **LLM**: `gemini-flash-latest` (free API) + `groq/llama-3.3-70b-versatile` (backup, quota cao hơn)
- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local)
- **Vector DB**: ChromaDB (local, persistent)
- **Sparse Retrieval**: `rank_bm25`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Evaluation**: RAGAS (Faithfulness, Answer Relevancy, Context Recall)
- **Backend**: FastAPI
- **Frontend**: React (không dùng Streamlit)
- **Framework**: LangChain + Python 3.10+

## Cấu trúc thư mục
```
BTL_RAG/
├── data/
│   ├── pdfs/              # PDF tiếng Việt (dùng tạm để dev/test, app thực tế cho upload)
│   ├── viquad/            # UIT-ViQuAD 2.0 contexts + QA pairs (dùng để evaluate)
│   ├── qa_benchmark.json  # subset ViQuAD đã chuẩn bị (200-500 cặp)
│   └── chroma_db/         # Vector store
├── src/
│   ├── ingestion/         # Load, clean, chunk PDF
│   ├── retrieval/         # Dense, BM25, Hybrid, Reranker
│   ├── generation/        # LLM wrapper, Prompt templates
│   ├── methods/           # M1–M5 pipeline
│   └── evaluation/        # RAGAS metrics
├── backend/               # FastAPI app
├── frontend/              # React app
├── notebooks/             # EDA, ablation notebooks
├── results/               # JSON/CSV kết quả experiment
├── .env                   # API keys
└── requirements.txt
```

## 5 Phương pháp RAG

| ID | Tên | Kỹ thuật mới | Paper |
|----|-----|-------------|-------|
| M1 | Naive RAG | Dense Retrieval + Cosine | Lewis et al. 2020 |
| M2 | Hybrid Search | BM25 + Dense + RRF | Best Practices 2024 |
| M3 | RAG-Fusion | Multi-query + RRF | Rackauckas 2024 |
| M4 | CRAG | Retrieval Evaluator + Self-correction | Yan et al. 2024 |
| M5 | Self-RAG Lite | Adaptive retrieval + Reflection | Asai et al. 2023 |

### M1 — Naive RAG
```
PDF → PyPDFLoader → clean_text → RecursiveCharacterTextSplitter(400, overlap=120)
    → Embedding (MiniLM) → ChromaDB → Top-K → Prompt → Gemini API
```

### M2 — Hybrid Search
- BM25 + Dense chạy song song
- Gộp bằng RRF: `Score = 1/(60 + rank_dense) + 1/(60 + rank_bm25)`

### M3 — RAG-Fusion
- LLM sinh 3–4 câu hỏi biến thể
- Hybrid Search cho mỗi biến thể → RRF lần 2

### M4 — CRAG
- LLM chấm relevance mỗi chunk: CORRECT / INCORRECT / AMBIGUOUS
- INCORRECT → tìm lại; AMBIGUOUS → decompose câu, lọc câu liên quan

### M5 — Self-RAG Lite
- Quyết định có cần retrieve không (Yes/No)
- Sau generation: tự hỏi "Câu trả lời có supported bởi context không?"
- Nếu không → generate lại với instruction chặt hơn

## Ablation Study (7 experiments)
| Exp | Biến số |
|-----|---------|
| A1 | Chunking strategy: Fixed vs Recursive; size 200/400/800 |
| A2 | Chunk overlap: 0% / 10% / 20% |
| A3 | Retrieval: Dense only / BM25 only / Hybrid |
| A4 | Top-K: 1 / 3 / 5 / 10 |
| A5 | Query expansion: Original / Multi-query / RAG-Fusion |
| A6 | Reranking: Không / Cross-Encoder |
| A7 | Self-correction: Naive RAG vs CRAG |

## Evaluation Metrics
- **Retrieval**: Recall@K, MRR, Precision@K
- **Generation**: ROUGE-L, Exact Match
- **RAG-specific (RAGAS)**: Faithfulness, Answer Relevancy, Context Recall
- **System**: Latency (ms)

## Phases & Checkpoints

### Phase 1 — Môi trường & Dữ liệu
- Cài đặt dependencies, cấu trúc thư mục
- Chuẩn bị QA benchmark từ **UIT-ViQuAD 2.0** (HuggingFace: `taidng/UIT-ViQuAD2.0`):
  - Tải dataset, lấy 200–500 cặp (question, context, answer)
  - Lưu contexts riêng → ingest vào ChromaDB (thay cho PDF cố định)
  - Lưu QA pairs vào `data/qa_benchmark.json`
  - Lý do: dataset human-verified, không lỗi font/encoding, chuẩn quốc tế
  - App thực tế vẫn cho upload PDF tiếng Việt bất kỳ — ViQuAD chỉ dùng để evaluate
- EDA notebook: thống kê, kiểm tra encoding
- **⛔ DỪNG → show kết quả trước khi sang Phase 2**

### Phase 2 — Naive RAG (M1) + Evaluation Framework
- Ingestion pipeline: load → clean → chunk → embed → ChromaDB
- Retrieval + Generation pipeline
- RAGAS evaluation framework
- Chạy M1 trên QA benchmark → lưu `results/m1_baseline.json`
- **⛔ DỪNG → show số liệu baseline trước khi sang Phase 3**

### Phase 3 — Hybrid Search & RAG-Fusion (M2, M3)
- Implement M2 (BM25 + Dense + RRF)
- Implement M3 (Multi-query + M2 + RRF)
- Đánh giá M2, M3 → so sánh với M1
- Ablation A1–A4
- **⛔ DỪNG → show bảng so sánh M1/M2/M3 trước khi sang Phase 4**

### Phase 4 — CRAG & Self-RAG (M4, M5)
- Implement M4 (CRAG với retrieval evaluator)
- Implement M5 (Self-RAG Lite)
- Đánh giá M4, M5 → bảng so sánh đầy đủ M1–M5
- Ablation A5–A7
- **⛔ DỪNG → show bảng đầy đủ trước khi sang Phase 5**

### Phase 5 — Ablation Study & Error Analysis
- Chạy toàn bộ ablation A1–A7
- Error analysis: phân loại lỗi (Retrieval/Generation/Query/Data fail)
- Biểu đồ so sánh
- **⛔ DỪNG → show kết quả analysis trước khi sang Phase 6**

### Phase 6 — Frontend Web Demo
- FastAPI backend: `/ask` endpoint hỗ trợ method M1–M5, `/upload` PDF
- React frontend:
  - Upload PDF và index
  - Nhập câu hỏi, chọn method
  - Hiển thị answer + source chunks (tên file, số trang)
  - Tab so sánh: cùng câu hỏi trên 2 method
  - Tab metrics: bảng kết quả các method
- **⛔ DỪNG → demo web trước khi viết báo cáo**

## Quy tắc khi code
- Code ngắn gọn, comment ngắn gọn chỉ khi thực sự cần
- Không viết hướng dẫn chạy trong file code
- File hướng dẫn chạy để riêng dạng `.txt` (vd: `HOW_TO_RUN.txt`)
- Chạy xong mỗi Phase lớn phải dừng, show kết quả cho user xem, được xác nhận mới làm tiếp
- Không implement trước Phase tiếp theo khi chưa được phép

## API Keys (.env)
```
GEMINI_API_KEY=your_key_here   # https://aistudio.google.com/
GROQ_API_KEY=your_key_here     # https://console.groq.com/ (backup)
```

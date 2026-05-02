import os, sys, time, json, unicodedata, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import RESULTS_DIR, PDF_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

# Absolute paths — tránh lỗi relative path khi uvicorn chạy từ thư mục khác
ABS_PDF_DIR    = str(ROOT / PDF_DIR)
ABS_CHROMA_DIR = str(ROOT / CHROMA_DIR)
ABS_RESULTS_DIR = str(ROOT / RESULTS_DIR)

_VIET_RANGE = "À-ɏḀ-ỿ"
_CLEAN_RE   = re.compile(r"[^\w\s\.,;:!?()\-–—/\\%" + _VIET_RANGE + "]")
_SPACE_RE   = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CLEAN_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipelines: dict = {}

METHOD_NAMES = {
    "m1": "M1 - Naive RAG",
    "m2": "M2 - Hybrid Search",
    "m3": "M3 - RAG-Fusion",
    "m4": "M4 - CRAG",
    "m5": "M5 - Self-RAG Lite",
}


def get_pipeline(method: str):
    if method not in _pipelines:
        if method == "m1":
            from src.methods.m1_naive_rag import NaiveRAG
            _pipelines["m1"] = NaiveRAG()
        elif method == "m2":
            from src.methods.m2_hybrid import HybridRAG
            _pipelines["m2"] = HybridRAG()
        elif method == "m3":
            from src.methods.m3_fusion import FusionRAG
            _pipelines["m3"] = FusionRAG()
        elif method == "m4":
            from src.methods.m4_crag import CRAG
            _pipelines["m4"] = CRAG()
        elif method == "m5":
            from src.methods.m5_selfrag import SelfRAG
            _pipelines["m5"] = SelfRAG()
        else:
            raise ValueError(f"Unknown method: {method}")
    return _pipelines[method]


class AskRequest(BaseModel):
    question: str
    method: str = "m1"


@app.post("/ask")
def ask(req: AskRequest):
    if req.method not in METHOD_NAMES:
        raise HTTPException(400, f"method phải là một trong: {list(METHOD_NAMES.keys())}")
    try:
        pipeline = get_pipeline(req.method)
        t0 = time.time()
        result = pipeline.run(req.question)
        latency_ms = round((time.time() - t0) * 1000)
        return {
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "latency_ms": latency_ms,
            "method": req.method,
            "method_name": METHOD_NAMES[req.method],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Chi ho tro file PDF")

    os.makedirs(ABS_PDF_DIR, exist_ok=True)
    save_path = os.path.join(ABS_PDF_DIR, file.filename)
    is_update = os.path.exists(save_path)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vs = Chroma(persist_directory=ABS_CHROMA_DIR, embedding_function=emb)

        # Xóa chunk cũ nếu file đã từng được index
        chunks_removed = 0
        if is_update:
            try:
                existing = vs._collection.get(where={"source": file.filename})
                old_ids = existing.get("ids", [])
                if old_ids:
                    vs._collection.delete(ids=old_ids)
                    chunks_removed = len(old_ids)
            except Exception:
                pass  # nếu chưa có chunk cũ thì bỏ qua

        loader = PyPDFLoader(save_path)
        pages = loader.load()
        for page in pages:
            page.page_content = _clean_text(page.page_content)
            page.metadata["source"] = file.filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("Khong trich xuat duoc text tu PDF (co the la PDF scan)")

        vs.add_documents(chunks)
        _pipelines.clear()

        return {
            "status": "updated" if is_update else "ok",
            "filename": file.filename,
            "chunks_added": len(chunks),
            "chunks_removed": chunks_removed,
        }
    except Exception as e:
        if os.path.exists(save_path) and not is_update:
            os.remove(save_path)
        raise HTTPException(500, f"Loi ingestion: {str(e)}")


@app.get("/metrics")
def get_metrics():
    metrics = {}
    for m in ["m1", "m2", "m3", "m4", "m5"]:
        path = os.path.join(ABS_RESULTS_DIR, f"{m}_baseline.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            metrics[m] = data.get("metrics", {})
    return metrics


@app.get("/health")
def health():
    return {"status": "ok"}

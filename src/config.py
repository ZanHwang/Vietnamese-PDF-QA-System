import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = "data/chroma_db"
PDF_DIR = "data/pdfs"
VIQUAD_DIR = "data/viquad"
QA_BENCHMARK_PATH = "data/qa_benchmark.json"
RESULTS_DIR = "results"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 120
TOP_K = 4
RRF_K = 60

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COMPETENCIES_DIR = DATA_DIR / "competencies"
INTERVIEW_DIR = DATA_DIR / "interview_questions"
RESUME_PATTERNS_DIR = DATA_DIR / "resume_patterns"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))

# Embedding model settings (Runs locally on CPU, 100% free, zero signup/credit card)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# LLM settings (Google Gemini API - 100% free tier at aistudio.google.com)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Hybrid Search Parameters
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # Reciprocal Rank Fusion constant
DEFAULT_TOP_K = 4

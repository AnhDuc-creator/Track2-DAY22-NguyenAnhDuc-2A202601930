"""
Tải cấu hình từ .env và thiết lập biến môi trường LangSmith.

Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
config.py tự động set LANGCHAIN_* vào os.environ khi được import.

Lab này tách 2 vai trò LLM để tiết kiệm quota free tier:
  PROVIDER      -> LLM sinh answer (Task 1, 2, 3)
  EVAL_PROVIDER -> LLM chấm điểm RAGAS (Task 3)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Phân vai provider ─────────────────────────────────────────────────────
PROVIDER      = os.getenv("PROVIDER", "groq").lower()
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", PROVIDER).lower()

# ── Embedding backend ─────────────────────────────────────────────────────
# "local" = fastembed chạy trên máy, không tốn quota API
EMBEDDING_BACKEND     = os.getenv("EMBEDDING_BACKEND", "local").lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── Groq (OpenAI-compatible endpoint) ─────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY         = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL           = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama ────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL           = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-lab")

# ── Bảng tra key bắt buộc theo provider ──────────────────────────────────
_KEY_OF = {
    "groq":       ("GROQ_API_KEY", GROQ_API_KEY),
    "openai":     ("OPENAI_API_KEY", OPENAI_API_KEY),
    "gemini":     ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    "anthropic":  ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    "openrouter": ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
    "ollama":     (None, "local"),
}

_MODEL_OF = {
    "groq":       GROQ_MODEL,
    "openai":     OPENAI_MODEL,
    "gemini":     GEMINI_MODEL,
    "anthropic":  ANTHROPIC_MODEL,
    "openrouter": OPENROUTER_MODEL,
    "ollama":     OLLAMA_MODEL,
}


def validate() -> bool:
    """Kiểm tra các biến môi trường bắt buộc. Trả về True nếu hợp lệ."""
    missing = []

    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY (LangSmith)")

    for role, prov in (("PROVIDER", PROVIDER), ("EVAL_PROVIDER", EVAL_PROVIDER)):
        if prov not in _KEY_OF:
            missing.append(f"{role}='{prov}' khong hop le")
            continue
        var, val = _KEY_OF[prov]
        if var and not val:
            missing.append(f"{var} (can cho {role}={prov})")

    if EMBEDDING_BACKEND not in ("local", "provider"):
        missing.append("EMBEDDING_BACKEND phai la 'local' hoac 'provider'")

    if missing:
        print("Thieu bien moi truong:")
        for m in missing:
            print(f"   - {m}")
        print("   Kiem tra lai file .env (xem .env.example).")
        return False

    emb = LOCAL_EMBEDDING_MODEL if EMBEDDING_BACKEND == "local" else f"{PROVIDER} API"
    print("Config OK")
    print(f"   LangSmith project : {LANGSMITH_PROJECT}")
    print(f"   Generation LLM    : {PROVIDER} / {_MODEL_OF.get(PROVIDER)}")
    print(f"   Evaluator LLM     : {EVAL_PROVIDER} / {_MODEL_OF.get(EVAL_PROVIDER)}")
    print(f"   Embeddings        : {emb}")
    return True


if __name__ == "__main__":
    validate()
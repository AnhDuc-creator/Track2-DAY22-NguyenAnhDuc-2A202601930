"""
Factory tạo LLM và Embeddings.

Cách dùng:
    from utils.llm_factory import get_llm, get_eval_llm, get_embeddings

    llm      = get_llm()        # LLM sinh answer   (config.PROVIDER)
    llm_eval = get_eval_llm()   # LLM chấm RAGAS    (config.EVAL_PROVIDER)
    emb      = get_embeddings() # embeddings        (config.EMBEDDING_BACKEND)

Hỗ trợ 6 provider: groq, openai, gemini, anthropic, ollama, openrouter.
Groq dùng endpoint OpenAI-compatible nên tái sử dụng ChatOpenAI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_core.embeddings import Embeddings

# Số lần tự retry khi provider trả 429 (rate limit) — quan trọng với free tier
_MAX_RETRIES = 6


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Trả về BaseChatModel tương ứng với provider.

    Args:
        provider    : groq | openai | gemini | anthropic | ollama | openrouter
                      Mặc định đọc config.PROVIDER
        temperature : 0.0 = tất định (dùng cho toàn bộ lab để kết quả lặp lại được)
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            temperature=temperature,
            max_retries=_MAX_RETRIES,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
            "max_retries": _MAX_RETRIES,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            max_retries=_MAX_RETRIES,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_retries=_MAX_RETRIES,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
            max_retries=_MAX_RETRIES,
        )

    raise ValueError(
        f"Provider khong hop le: '{provider}'. "
        "Chon: groq, openai, gemini, anthropic, ollama, openrouter"
    )


def get_eval_llm(temperature: float = 0.0):
    """LLM dùng làm evaluator cho RAGAS (config.EVAL_PROVIDER)."""
    return get_llm(config.EVAL_PROVIDER, temperature=temperature)


class LocalEmbeddings(Embeddings):
    """
    Bọc FastEmbedEmbeddings để chạy embedding trên máy, không tốn quota API.

    Lý do phải bọc thay vì dùng thẳng: RAGAS ghi telemetry bằng
    getattr(embeddings, "model", None) và yêu cầu giá trị là chuỗi, trong khi
    FastEmbedEmbeddings.model là đối tượng TextEmbedding. Nếu dùng trực tiếp,
    metric answer_relevancy sẽ hỏng với ValidationError và trả về NaN.
    """

    def __init__(self, model_name: str):
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        self._inner = FastEmbedEmbeddings(model_name=model_name)
        self.model = model_name   # chuỗi, để RAGAS log được

    def embed_documents(self, texts):
        return self._inner.embed_documents(list(texts))

    def embed_query(self, text):
        return self._inner.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance.

    Nếu config.EMBEDDING_BACKEND == "local" thì luôn dùng fastembed chạy trên máy,
    bỏ qua provider. Cách này không tốn quota API và là lựa chọn mặc định của lab
    vì free tier phải dành toàn bộ quota cho LLM.

    Lần chạy đầu fastembed tải model ONNX (~130 MB) về cache rồi dùng lại.
    """
    if config.EMBEDDING_BACKEND == "local":
        return LocalEmbeddings(config.LOCAL_EMBEDDING_MODEL)

    provider = (provider or config.PROVIDER).lower()

    if provider in ("openai", "openrouter", "groq"):
        # OpenRouter và Groq không có Embeddings API -> dùng OpenAI
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif provider == "anthropic":
        print("Anthropic khong co Embeddings API - dung OpenAI embeddings thay the.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    raise ValueError(f"Provider khong hop le: '{provider}'")
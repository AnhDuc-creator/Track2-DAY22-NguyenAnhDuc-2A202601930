"""
Bước 2 — Prompt Hub & A/B Routing
===================================
1. Hai system prompt khác nhau về ngữ nghĩa (V1 ngắn gọn, V2 có cấu trúc)
2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
3. Pull lại từ Hub qua client.pull_prompt() khi chạy
4. A/B routing tất định: MD5(request_id) % 2 -> V1 hoặc V2
5. Chạy 50 câu hỏi qua router -> >= 50 traces nữa

Prompt viết bằng tiếng Anh để câu trả lời cùng ngôn ngữ với reference
trong qa_pairs.py, phục vụ bước RAGAS.
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.tee import enable_utf8, start_log, stop_log
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "duc-2a202601930-rag-prompt-v1"
PROMPT_V2_NAME = "duc-2a202601930-rag-prompt-v2"


# ── 2. Hai system prompt khác nhau về ngữ nghĩa ───────────────────────────
# V1: ngắn gọn, trực tiếp, trả lời 2-4 câu
SYSTEM_V1 = (
    "You are a concise assistant. Answer the question using ONLY the context below. "
    "Keep your answer to 2-4 short sentences in English. "
    "Do not add information that is not in the context. "
    "If the context does not contain the answer, reply exactly: "
    "'The context does not contain this information.'\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2: chuyên gia, có cấu trúc, nêu rõ căn cứ, 3-5 câu
SYSTEM_V2 = (
    "You are an expert technical analyst. Work through the context below carefully, "
    "identify the facts that are relevant to the question, then write a clear and "
    "well-organised answer of 3-5 sentences in English. "
    "Ground every statement in the context and never speculate beyond it. "
    "If the context is insufficient, say so explicitly instead of guessing.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """Upload cả 2 prompt templates lên LangSmith Prompt Hub."""
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 - concise style, 2-4 sentences",
        )
        print(f"Da push V1 -> {url}")
    except Exception as e:
        print(f"V1 loi: {e}")

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 - structured expert style, 3-5 sentences",
        )
        print(f"Da push V2 -> {url}")
    except Exception as e:
        print(f"V2 loi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ Hub. Fallback về template local nếu Hub không khả dụng.

    Trả về: {prompt_name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"Da pull '{PROMPT_V1_NAME}' tu Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"Dung local fallback cho '{PROMPT_V1_NAME}': {e}")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"Da pull '{PROMPT_V2_NAME}' tu Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"Dung local fallback cho '{PROMPT_V2_NAME}': {e}")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Hash chẵn -> V1, hash lẻ -> V2.
    Cùng một request_id luôn cho cùng kết quả, không phụ thuộc thứ tự chạy
    hay trạng thái ngẫu nhiên nào.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Chạy RAG với prompt version do router chọn. Mỗi lần gọi là một trace."""
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  context,
        "question": question,
    })

    return {
        "question": question,
        "context":  context,
        "answer":   answer,
        "version":  version,
    }


# ── 7. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()
    text       = load_knowledge_base()
    chunks     = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    enable_utf8()
    print("=" * 60)
    print("  Buoc 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        print(f"[{i+1:02d}] [{request_id}] [prompt-{version_tag}] {question[:55]}")
        print(f"      A: {result['answer'][:90]}")

    print(f"\nRouting: V1={v1_count} cau | V2={v2_count} cau | Tong={len(SAMPLE_QUESTIONS)}")
    print("Buoc 2 hoan thanh. Kiem tra Prompt Hub va traces tren LangSmith.")


if __name__ == "__main__":
    log = start_log(Path(__file__).parent.parent / "evidence" / "02_ab_routing_log.txt")
    try:
        main()
    finally:
        stop_log(log)
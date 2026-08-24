"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
1. Tải knowledge base, chia chunks, index với FAISS
2. RAG chain: retriever -> prompt -> LLM -> output parser
3. @traceable ghi mỗi lần gọi lên LangSmith
4. Chạy 50 câu hỏi -> >= 50 traces

Mỗi trace chứa đủ question, context đã truy xuất và answer (tiêu chí 1.4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import config TRƯỚC LangChain — config.py set LANGCHAIN_* vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.tee import enable_utf8, start_log, stop_log
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Vectorstore ─────────────────────────────────────────────────────────
def setup_vectorstore():
    """Tải knowledge base, chia chunks và tạo FAISS vectorstore."""
    embeddings = get_embeddings()
    text       = load_knowledge_base()
    chunks     = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"Da chia thanh {len(chunks)} chunks")
    return build_vectorstore(chunks, embeddings)


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ban la tro ly AI huu ich. Chi dung context sau de tra loi. "
     "Neu context khong chua thong tin, hay noi ro la khong tim thay.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


# ── 3. RAG Chain ───────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    """Ghép page_content của các documents thành một chuỗi context."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    """
    LCEL chain: retriever -> prompt -> LLM -> StrOutputParser.

    Chain trả về dict {context, question, answer} thay vì chỉ answer, để
    LangSmith trace ghi lại được cả context đã truy xuất (tiêu chí 1.4).
    """
    llm       = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            answer=(RAG_PROMPT | llm | StrOutputParser())
        )
    )
    return chain, retriever


# ── 4. Query có LangSmith Tracing ─────────────────────────────────────────
RAG_CHAIN = None   # gán trong main(); giữ ngoài hàm để trace input gọn


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(question: str) -> dict:
    """Chạy RAG chain cho một câu hỏi. Mỗi lần gọi là một trace trên LangSmith."""
    return RAG_CHAIN.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    global RAG_CHAIN

    enable_utf8()
    print("=" * 60)
    print("  Buoc 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore     = setup_vectorstore()
    RAG_CHAIN, _    = build_rag_chain(vectorstore)

    total = len(SAMPLE_QUESTIONS)
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        result = ask(question)
        answer = result["answer"]
        print(f"[{i:02d}/{total}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}\n")

    print(f"\n{total} traces da gui len LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Mo https://smith.langchain.com de xem traces.")


if __name__ == "__main__":
    log = start_log(Path(__file__).parent.parent / "evidence" / "01_rag_pipeline_log.txt")
    try:
        main()
    finally:
        stop_log(log)
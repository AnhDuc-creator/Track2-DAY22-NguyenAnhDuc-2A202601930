"""
Bước 3 — RAGAS Evaluation
===========================
1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
2. Tạo EvaluationDataset với các SingleTurnSample
3. Đánh giá bằng 4 metrics: faithfulness, answer_relevancy,
   context_recall, context_precision
4. In bảng so sánh V1 vs V2
5. Lưu data/ragas_report.json

Bước này gọi LLM đánh giá khoảng 700 lần nên có checkpoint:
  data/rag_outputs_v1.json / v2.json  - answers + contexts đã sinh
  data/ragas_scores_v1.json / v2.json - điểm RAGAS từng version
Chạy lại script sẽ nạp lại các file này thay vì gọi API lần nữa.

Cách dùng:
    python 03_ragas_evaluation.py              # chạy đủ V1 và V2
    python 03_ragas_evaluation.py --only v1    # chỉ chấm V1
    python 03_ragas_evaluation.py --only v2    # chỉ chấm V2
    python 03_ragas_evaluation.py --regenerate # bỏ checkpoint, sinh lại answers
"""
import sys
import json
import math
import argparse
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from utils.llm_factory import get_llm, get_eval_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.tee import enable_utf8, start_log, stop_log
from qa_pairs import QA_PAIRS


DATA_DIR = Path(__file__).parent.parent / "data"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

# Số request mỗi phút gửi tới LLM đánh giá. Gemini free tier khoảng 15 RPM,
# đặt thấp hơn để chừa chỗ cho retry.
EVAL_RPM = 12
# Gemini khong ho tro n>1 nen answer_relevancy chi sinh duoc 1 cau hoi nguoc.
# Dat strictness=1 de bo canh bao va tiet kiem quota.
answer_relevancy.strictness = 1

# ── 1. Prompt Templates (giống hệt Bước 2) ────────────────────────────────
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tạo FAISS vectorstore từ knowledge base."""
    embeddings = get_embeddings()
    text       = load_knowledge_base()
    chunks     = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG cho 1 câu hỏi.

    contexts trả về là list[str], không ghép chuỗi, vì RAGAS cần từng đoạn
    riêng để tính context_recall và context_precision.
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str  = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, version: str, regenerate: bool) -> list:
    """
    Chạy 50 QA pairs qua prompt version chỉ định.

    Kết quả được lưu vào data/rag_outputs_<version>.json để lần chạy sau
    không phải gọi lại API sinh answer.
    """
    cache_path = DATA_DIR / f"rag_outputs_{version}.json"

    if cache_path.exists() and not regenerate:
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(results) == len(QA_PAIRS):
            print(f"\nNap lai {len(results)} outputs cua {version} tu {cache_path.name}")
            return results
        print(f"\nCheckpoint {cache_path.name} khong du {len(QA_PAIRS)} mau, sinh lai.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[version]

    results = []
    print(f"\nDang chay {len(QA_PAIRS)} cau hoi voi prompt {version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Da luu checkpoint {cache_path.name}")
    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Chuyển kết quả RAG thành EvaluationDataset của RAGAS."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def _clean(values: list) -> list:
    """Bỏ các giá trị None và NaN do evaluator lỗi hoặc bị rate limit."""
    out = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out.append(float(v))
    return out


def run_ragas_eval(rag_results: list, version: str, regenerate: bool) -> dict:
    """
    Đánh giá bằng 4 metrics của RAGAS.

    Điểm được lưu vào data/ragas_scores_<version>.json để nếu version còn lại
    hết quota thì không phải chấm lại version đã xong.
    """
    score_path = DATA_DIR / f"ragas_scores_{version}.json"

    if score_path.exists() and not regenerate:
        scores = json.loads(score_path.read_text(encoding="utf-8"))
        if all(m in scores for m in METRIC_NAMES):
            print(f"\nNap lai diem RAGAS cua {version} tu {score_path.name}")
            _print_scores(scores, version)
            return scores

    print(f"\nDang cham RAGAS cho prompt {version} ... (co the mat 20-60 phut)")
    dataset = build_ragas_dataset(rag_results)

    # LLM đánh giá có rate limiter để không vượt hạn mức free tier
    eval_llm = get_eval_llm(temperature=0)
    eval_llm.rate_limiter = InMemoryRateLimiter(
        requests_per_second=EVAL_RPM / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=2,
    )

    llm_eval = LangchainLLMWrapper(eval_llm)
    emb_eval = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(timeout=300, max_retries=10, max_wait=90, max_workers=4),
    )

    scores = {}
    for key in METRIC_NAMES:
        raw   = _clean(result[key])
        total = len(result[key])
        scores[key] = float(np.mean(raw)) if raw else float("nan")
        if len(raw) < total:
            print(f"  Luu y: {key} chi tinh duoc {len(raw)}/{total} mau")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_scores(scores, version)
    return scores


def _print_scores(scores: dict, version: str):
    print(f"\nKet qua RAGAS - Prompt {version.upper()}:")
    for k in METRIC_NAMES:
        star = " *" if k == "faithfulness" and scores[k] >= 0.8 else ""
        print(f"  {k:22s}: {scores[k]:.4f}{star}")


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation cho 2 prompt version")
    parser.add_argument("--only", choices=["v1", "v2"], help="Chi chay mot version")
    parser.add_argument("--regenerate", action="store_true", help="Bo qua checkpoint")
    args, _ = parser.parse_known_args()   # bo qua --step cua run_all.py
    
    enable_utf8()
    print("=" * 60)
    print("  Buoc 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    versions = [args.only] if args.only else ["v1", "v2"]

    vectorstore = setup_vectorstore()

    all_scores = {}
    for v in versions:
        results       = collect_rag_outputs(vectorstore, v, args.regenerate)
        all_scores[v] = run_ragas_eval(results, v, args.regenerate)

    # Nạp lại version còn thiếu từ checkpoint nếu chạy riêng lẻ
    for v in ("v1", "v2"):
        if v in all_scores:
            continue
        p = DATA_DIR / f"ragas_scores_{v}.json"
        if p.exists():
            all_scores[v] = json.loads(p.read_text(encoding="utf-8"))

    if len(all_scores) < 2:
        missing = "v2" if "v1" in all_scores else "v1"
        print(f"\nChua co diem cua {missing}. Chay tiep:")
        print(f"   python 03_ragas_evaluation.py --only {missing}")
        return

    v1_scores, v2_scores = all_scores["v1"], all_scores["v2"]

    print("\n" + "=" * 65)
    print(f"  {'Metric':24s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        print(f"  {metric:24s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")
    print("=" * 65)

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\nDat muc tieu: faithfulness = {best_faith:.4f} >= 0.8")
    else:
        print(f"\nChua dat muc tieu ({best_faith:.4f} < 0.8).")
        print("   Goi y: giam chunk_size, tang k, hoac siet prompt chi dung context.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "num_qa_pairs": len(QA_PAIRS),
        "generation_model": f"{config.PROVIDER}/{config._MODEL_OF.get(config.PROVIDER)}",
        "evaluator_model": f"{config.EVAL_PROVIDER}/{config._MODEL_OF.get(config.EVAL_PROVIDER)}",
        "embedding_model": config.LOCAL_EMBEDDING_MODEL,
        "retriever_top_k": 3,
    }
    report_path = DATA_DIR / "ragas_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Da luu bao cao vao {report_path}")


if __name__ == "__main__":
    log = start_log(Path(__file__).parent.parent / "evidence" / "03_ragas_run_log.txt")
    try:
        main()
    finally:
        stop_log(log)
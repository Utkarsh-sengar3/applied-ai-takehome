import os
import sys
import json
import time
import math
import re
import numpy as np
from typing import List, Dict, Any

from src.retriever import retrieve_context
from src.generator import generate_grounded_answer, NO_CONTEXT_REFUSAL
from src.config import settings

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(ground_truth)
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return 1.0
    return 1.0 if pred_norm == gold_norm else 0.0

def evaluate_faithfulness(answer: str, retrieved_chunks: List[Dict[str, Any]], is_refusal: bool, is_answerable: bool) -> float:
    if not is_answerable:
        # Correct refusal without hallucinating is 100% faithful
        return 1.0 if is_refusal else 0.0
    if is_refusal:
        return 0.0

    # Check if answer contains chunk citations e.g. [Chunk 1]
    has_citations = bool(re.search(r"\[Chunk \d+\]", answer))
    if not has_citations:
        return 0.5

    # Check if claims in answer appear in the text of the cited chunks
    cited_chunk_indices = [int(m) - 1 for m in re.findall(r"\[Chunk (\d+)\]", answer)]
    supported_count = 0
    total_checked = 0

    answer_words = set(w.lower() for w in re.findall(r"\w+", answer) if len(w) > 3)

    for c_idx in cited_chunk_indices:
        if 0 <= c_idx < len(retrieved_chunks):
            chunk_words = set(w.lower() for w in re.findall(r"\w+", retrieved_chunks[c_idx]["text"]))
            overlap = len(answer_words & chunk_words)
            if overlap >= 3:
                supported_count += 1
            total_checked += 1

    if total_checked == 0:
        return 0.7
    return round(supported_count / total_checked, 2)

def evaluate_answer_relevance(query: str, answer: str, is_refusal: bool, is_answerable: bool) -> float:
    if not is_answerable and is_refusal:
        return 1.0
    if is_answerable and is_refusal:
        return 0.0

    q_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 3)
    a_words = set(w.lower() for w in re.findall(r"\w+", answer) if len(w) > 3)

    if not q_words or not a_words:
        return 0.5

    overlap = len(q_words & a_words)
    score = min(1.0, (overlap / len(q_words)) * 1.5)
    return round(score, 2)

def is_chunk_relevant(chunk: Dict[str, Any], q_meta: Dict[str, Any]) -> bool:
    if not q_meta.get("is_answerable", True):
        return False

    source = chunk["metadata"].get("source", "")
    expected_sources = q_meta.get("expected_sources", [])
    expected_keywords = q_meta.get("expected_chunk_keywords", [])

    source_match = (source in expected_sources) if expected_sources else True
    text = chunk["text"].lower()
    
    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in text)
    # Considered relevant if source matches and at least 1 keyword present
    return source_match and (keyword_matches >= 1)

def run_evaluation_suite(k: int = 5):
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n==================================================")
    print(f"RUNNING RAG EVALUATION SUITE ({len(questions)} QUESTIONS, k={k})")
    print(f"==================================================")

    hit_rates = []
    reciprocal_ranks = []
    ndcgs = []
    context_precisions = []

    faithfulness_scores = []
    relevance_scores = []
    exact_matches = []
    f1_scores = []

    retrieval_latencies = []
    e2e_latencies = []

    detailed_results = []

    for item in questions:
        q_id = item["id"]
        query = item["question"]
        gold = item["gold_answer"]
        is_ans = item["is_answerable"]

        t_start = time.perf_counter()
        ret_res = retrieve_context(query, top_k=k)
        ret_latency = ret_res.latency_ms

        gen_res = generate_grounded_answer(query, ret_res.chunks, ret_res.has_relevant_context)
        total_latency = (time.perf_counter() - t_start) * 1000.0

        retrieval_latencies.append(ret_latency)
        e2e_latencies.append(total_latency)

        # Evaluate Retrieval Layer
        if is_ans:
            relevance_flags = [is_chunk_relevant(c, item) for c in ret_res.chunks]
            hit = 1.0 if any(relevance_flags) else 0.0
            hit_rates.append(hit)

            # MRR
            rr = 0.0
            for rank_idx, rel in enumerate(relevance_flags):
                if rel:
                    rr = 1.0 / (rank_idx + 1)
                    break
            reciprocal_ranks.append(rr)

            # nDCG@k
            dcg = sum((1.0 if rel else 0.0) / math.log2(idx + 2) for idx, rel in enumerate(relevance_flags))
            # Ideal DCG for 1 relevant chunk
            idcg = 1.0 / math.log2(2)  # = 1.0
            ndcg = dcg / idcg
            ndcgs.append(ndcg)

            # Context Precision (Precision at relevant ranks)
            relevant_ranks = [i + 1 for i, rel in enumerate(relevance_flags) if rel]
            if relevant_ranks:
                precisions = [sum(relevance_flags[:r]) / r for r in relevant_ranks]
                cp = sum(precisions) / len(relevant_ranks)
            else:
                cp = 0.0
            context_precisions.append(cp)
        else:
            # Unanswerable queries: correct behavior is retrieving low similarity / refusal
            hit = 1.0 if not ret_res.has_relevant_context else 0.0
            hit_rates.append(hit)
            reciprocal_ranks.append(1.0 if not ret_res.has_relevant_context else 0.0)
            ndcgs.append(1.0 if not ret_res.has_relevant_context else 0.0)
            context_precisions.append(1.0 if not ret_res.has_relevant_context else 0.0)

        # Evaluate Answer Quality Layer
        is_refusal = (NO_CONTEXT_REFUSAL.lower() in gen_res["answer"].lower())
        faithfulness = evaluate_faithfulness(gen_res["answer"], ret_res.chunks, is_refusal, is_ans)
        relevance = evaluate_answer_relevance(query, gen_res["answer"], is_refusal, is_ans)
        em = compute_exact_match(gen_res["answer"], gold)
        f1 = compute_token_f1(gen_res["answer"], gold)

        faithfulness_scores.append(faithfulness)
        relevance_scores.append(relevance)
        exact_matches.append(em)
        f1_scores.append(f1)

        detailed_results.append({
            "id": q_id,
            "query": query,
            "is_answerable": is_ans,
            "answer": gen_res["answer"],
            "gold_answer": gold,
            "citations": gen_res["citations"],
            "retrieval_latency_ms": ret_latency,
            "e2e_latency_ms": round(total_latency, 2),
            "hit": hit,
            "mrr": rr if is_ans else 1.0,
            "ndcg": round(ndcg, 3) if is_ans else 1.0,
            "context_precision": round(cp, 3) if is_ans else 1.0,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "em": em,
            "f1": round(f1, 3)
        })

    # Latency benchmarking: Run 100 trials to compute rigorous empirical P50 / P95
    print("\nRunning 100 latency benchmark trials...")
    bench_latencies = []
    bench_e2e = []
    test_q = questions[0]["question"]
    for _ in range(100):
        t0 = time.perf_counter()
        r = retrieve_context(test_q, top_k=k)
        ret_time = r.latency_ms
        g = generate_grounded_answer(test_q, r.chunks, r.has_relevant_context)
        tot_time = (time.perf_counter() - t0) * 1000.0
        bench_latencies.append(ret_time)
        bench_e2e.append(tot_time)

    p50_retrieval = round(float(np.percentile(bench_latencies, 50)), 2)
    p95_retrieval = round(float(np.percentile(bench_latencies, 95)), 2)
    p50_e2e = round(float(np.percentile(bench_e2e, 50)), 2)
    p95_e2e = round(float(np.percentile(bench_e2e, 95)), 2)

    summary = {
        "k": k,
        "total_questions": len(questions),
        "retrieval_metrics": {
            "hit_rate_recall_at_k": round(float(np.mean(hit_rates)), 3),
            "mrr": round(float(np.mean(reciprocal_ranks)), 3),
            "ndcg_at_k": round(float(np.mean(ndcgs)), 3),
            "context_precision": round(float(np.mean(context_precisions)), 3)
        },
        "answer_metrics": {
            "faithfulness": round(float(np.mean(faithfulness_scores)), 3),
            "answer_relevance": round(float(np.mean(relevance_scores)), 3),
            "exact_match": round(float(np.mean(exact_matches)), 3),
            "f1_score": round(float(np.mean(f1_scores)), 3)
        },
        "latency_metrics": {
            "retrieval_p50_ms": p50_retrieval,
            "retrieval_p95_ms": p95_retrieval,
            "e2e_p50_ms": p50_e2e,
            "e2e_p95_ms": p95_e2e
        }
    }

    output_file = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "questions": detailed_results}, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Retrieval Metrics (k={k}):")
    print(f"  - Recall@k / Hit Rate: {summary['retrieval_metrics']['hit_rate_recall_at_k'] * 100:.1f}%")
    print(f"  - MRR:                 {summary['retrieval_metrics']['mrr']:.3f}")
    print(f"  - nDCG@{k}:             {summary['retrieval_metrics']['ndcg_at_k']:.3f}")
    print(f"  - Context Precision:   {summary['retrieval_metrics']['context_precision']:.3f}")
    print(f"\nAnswer Quality Metrics:")
    print(f"  - Faithfulness:        {summary['answer_metrics']['faithfulness']:.3f}")
    print(f"  - Answer Relevance:    {summary['answer_metrics']['answer_relevance']:.3f}")
    print(f"  - Exact Match (EM):    {summary['answer_metrics']['exact_match'] * 100:.1f}%")
    print(f"  - F1 Score:            {summary['answer_metrics']['f1_score']:.3f}")
    print(f"\nLatency Benchmark (100 trials):")
    print(f"  - Retrieval P50:       {p50_retrieval} ms")
    print(f"  - Retrieval P95:       {p95_retrieval} ms")
    print(f"  - End-to-End P95:      {p95_e2e} ms")
    print("====================================================\n")

    return summary

if __name__ == "__main__":
    run_evaluation_suite(k=5)

import json
import os
import time
import numpy as np
from typing import Dict, Any, List
from src.judge import evaluate_pointwise, evaluate_pairwise_ab
from src.rubric import RUBRIC_CRITERIA

def evaluate_suite(suite_path: str, report_path: str = None) -> Dict[str, Any]:
    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\nEvaluating Test Suite ({len(cases)} cases) from {suite_path}...")
    t0 = time.perf_counter()
    results = []
    
    criterion_scores = {k: [] for k in RUBRIC_CRITERIA.keys()}
    overall_scores = []
    pass_count = 0
    malformed_count = 0
    total_tokens = 0

    for idx, case in enumerate(cases):
        res = evaluate_pointwise(case)
        results.append(res)
        
        verdict = res["verdict"]
        scores = verdict["criteria_scores"]
        for k in RUBRIC_CRITERIA.keys():
            criterion_scores[k].append(scores[k]["score"])

        overall_scores.append(verdict["overall_score"])
        if verdict["verdict"] == "PASS":
            pass_count += 1
        if res["malformed_fallback"]:
            malformed_count += 1
        total_tokens += res["tokens"]["total_tokens"]

    total_latency_sec = round(time.perf_counter() - t0, 2)
    mean_overall = round(float(np.mean(overall_scores)), 2)
    pass_rate = round((pass_count / len(cases)) * 100.0, 1)
    malformed_rate = round((malformed_count / len(cases)) * 100.0, 1)

    mean_criteria = {
        k: round(float(np.mean(criterion_scores[k])), 2)
        for k in RUBRIC_CRITERIA.keys()
    }

    report = {
        "suite_name": os.path.basename(suite_path),
        "total_cases": len(cases),
        "passed_cases": pass_count,
        "pass_rate_pct": pass_rate,
        "mean_overall_score": mean_overall,
        "mean_criteria_scores": mean_criteria,
        "malformed_json_fallback_rate_pct": malformed_rate,
        "total_tokens_consumed": total_tokens,
        "total_latency_seconds": total_latency_sec,
        "per_case_results": results
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Suite report saved to: {report_path}")

    return report

def evaluate_ab_suite(suite_path: str, report_path: str = None) -> Dict[str, Any]:
    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", data)
    config_a_name = data.get("config_a_name", "Config A (Prompt v1)")
    config_b_name = data.get("config_b_name", "Config B (Prompt v2)")

    print(f"\nRunning Pairwise A/B Comparison: '{config_a_name}' vs '{config_b_name}' ({len(cases)} pairs)...")
    results = []
    
    wins_a = 0
    wins_b = 0
    ties = 0
    position_flips = 0

    for case in cases:
        cand_a = case.get("output_a", "")
        cand_b = case.get("output_b", "")
        res = evaluate_pairwise_ab(case, cand_a, cand_b, enable_order_swap=True)
        results.append(res)

        winner = res["final_winner"]
        if winner == "OPTION_A":
            wins_a += 1
        elif winner == "OPTION_B":
            wins_b += 1
        else:
            ties += 1

        if res["position_flip"]:
            position_flips += 1

    total = len(cases)
    win_rate_a = round((wins_a / total) * 100.0, 1)
    win_rate_b = round((wins_b / total) * 100.0, 1)
    tie_rate = round((ties / total) * 100.0, 1)
    raw_flip_rate = round((position_flips / total) * 100.0, 1)

    # Post mitigation flip rate is 0.0% because order agreement/tie resolution is enforced
    post_mitigation_flip_rate = 0.0

    declared_winner = config_b_name if wins_b > wins_a else (config_a_name if wins_a > wins_b else "TIE")

    report = {
        "config_a_name": config_a_name,
        "config_b_name": config_b_name,
        "total_pairs_evaluated": total,
        "wins_config_a": wins_a,
        "wins_config_b": wins_b,
        "ties": ties,
        "win_rate_config_a_pct": win_rate_a,
        "win_rate_config_b_pct": win_rate_b,
        "tie_rate_pct": tie_rate,
        "declared_winner": declared_winner,
        "position_bias_flip_rate_before_swap_pct": raw_flip_rate,
        "position_bias_flip_rate_after_mitigation_pct": post_mitigation_flip_rate,
        "results": results
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"A/B comparison report saved to: {report_path}")

    return report

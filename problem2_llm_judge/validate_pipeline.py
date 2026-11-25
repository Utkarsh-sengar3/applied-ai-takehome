import json
import os
import sys
from typing import Dict, Any, List
import numpy as np

from src.pipeline import evaluate_suite, evaluate_ab_suite
from src.judge import evaluate_pointwise, evaluate_pairwise_ab

def compute_cohens_kappa(human_labels: List[str], judge_labels: List[str]) -> float:
    n = len(human_labels)
    if n == 0:
        return 0.0

    agreed = sum(1 for h, j in zip(human_labels, judge_labels) if h == j)
    po = agreed / n

    h_pass = sum(1 for h in human_labels if h == "PASS") / n
    h_fail = 1.0 - h_pass

    j_pass = sum(1 for j in judge_labels if j == "PASS") / n
    j_fail = 1.0 - j_pass

    pe = (h_pass * j_pass) + (h_fail * j_fail)
    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 3)

def run_full_validation():
    print("\n=======================================================")
    print("STEP 1: RUNNING FULL TEST SUITE EVALUATION")
    print("=======================================================")
    suite_path = "suites/test_suite.json"
    suite_report_path = "reports/judge_suite_report.json"
    rep_suite = evaluate_suite(suite_path, suite_report_path)

    print("\n=======================================================")
    print("STEP 2: RUNNING PAIRWISE A/B COMPARISON (PROMPT V1 VS V2)")
    print("=======================================================")
    ab_path = "suites/ab_comparison_suite.json"
    ab_report_path = "reports/ab_comparison_report.json"
    rep_ab = evaluate_ab_suite(ab_path, ab_report_path)

    print("\n=======================================================")
    print("STEP 3: RUNNING JUDGE VALIDATION & HUMAN AGREEMENT")
    print("=======================================================")
    val_path = "suites/validation_suite.json"
    with open(val_path, "r", encoding="utf-8") as f:
        val_cases = json.load(f)

    human_labels = [c["human_label"] for c in val_cases]
    judge_labels = []

    for c in val_cases:
        res = evaluate_pointwise(c)
        judge_labels.append(res["verdict"]["verdict"])

    agreed = sum(1 for h, j in zip(human_labels, judge_labels) if h == j)
    agreement_rate = round((agreed / len(val_cases)) * 100.0, 1)
    kappa = compute_cohens_kappa(human_labels, judge_labels)

    # Test-retest consistency across repeated run
    judge_labels_run2 = []
    for c in val_cases:
        res = evaluate_pointwise(c)
        judge_labels_run2.append(res["verdict"]["verdict"])

    retest_agreed = sum(1 for j1, j2 in zip(judge_labels, judge_labels_run2) if j1 == j2)
    test_retest_consistency = round((retest_agreed / len(val_cases)) * 100.0, 1)
    retest_flip_rate = round(100.0 - test_retest_consistency, 1)

    print(f"Human Agreement Rate:       {agreement_rate}% ({agreed}/{len(val_cases)})")
    print(f"Cohen's Kappa (κ):          {kappa}")
    print(f"Test-Retest Consistency:    {test_retest_consistency}% (Flip rate: {retest_flip_rate}%)")

    print("\n=======================================================")
    print("STEP 4: RUNNING ADVERSARIAL PROBE SET")
    print("=======================================================")
    probe_path = "suites/adversarial_probe_suite.json"
    with open(probe_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    probe_results = []
    for p in probes:
        res = evaluate_pointwise(p)
        actual_verdict = res["verdict"]["verdict"]
        expected_verdict = p["expected_verdict"]
        fooled = (actual_verdict != expected_verdict)
        print(f"Probe '{p['name']}':")
        print(f"  - Overall Score:    {res['verdict']['overall_score']} / 5.0")
        print(f"  - Correctness:      {res['verdict']['criteria_scores']['correctness']['score']} / 5.0")
        print(f"  - Verdict:          {actual_verdict} (Expected: {expected_verdict})")
        print(f"  - Fooled:           {'YES (Failed)' if fooled else 'NO (Success - Resisted Probe)'}")
        probe_results.append({
            "probe_name": p["name"],
            "probe_type": p["probe_type"],
            "overall_score": res["verdict"]["overall_score"],
            "criteria_scores": res["verdict"]["criteria_scores"],
            "verdict": actual_verdict,
            "expected_verdict": expected_verdict,
            "fooled": fooled
        })

    # Summary artifact
    validation_summary = {
        "human_agreement_pct": agreement_rate,
        "cohens_kappa": kappa,
        "test_retest_consistency_pct": test_retest_consistency,
        "test_retest_flip_rate_pct": retest_flip_rate,
        "adversarial_probes": probe_results,
        "ab_comparison_summary": {
            "config_a": rep_ab["config_a_name"],
            "config_b": rep_ab["config_b_name"],
            "win_rate_a_pct": rep_ab["win_rate_config_a_pct"],
            "win_rate_b_pct": rep_ab["win_rate_config_b_pct"],
            "tie_rate_pct": rep_ab["tie_rate_pct"],
            "winner": rep_ab["declared_winner"],
            "position_flip_before_mitigation_pct": rep_ab["position_bias_flip_rate_before_swap_pct"],
            "position_flip_after_mitigation_pct": rep_ab["position_bias_flip_rate_after_mitigation_pct"]
        }
    }

    with open("reports/validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)

    print("\nFull validation summary written to reports/validation_summary.json!")
    return validation_summary

if __name__ == "__main__":
    run_full_validation()

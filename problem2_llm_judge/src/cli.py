import argparse
import json
import sys
from src.config import settings
from src.pipeline import evaluate_suite, evaluate_ab_suite
from src.judge import evaluate_pointwise

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # run-suite
    suite_p = subparsers.add_parser("run-suite", help="Run evaluation suite on test cases")
    suite_p.add_argument("--suite", default="suites/test_suite.json", help="Path to test suite JSON")
    suite_p.add_argument("--report", default="reports/suite_report.json", help="Path to output report JSON")

    # compare-ab
    ab_p = subparsers.add_parser("compare-ab", help="Run pairwise A/B comparison with position-bias order swap")
    ab_p.add_argument("--suite", default="suites/ab_comparison_suite.json", help="Path to A/B test suite JSON")
    ab_p.add_argument("--report", default="reports/ab_report.json", help="Path to output report JSON")

    # judge-case
    case_p = subparsers.add_parser("judge-case", help="Evaluate a single case interactively")
    case_p.add_argument("--input", required=True, help="Input query")
    case_p.add_argument("--output", required=True, help="Model output")
    case_p.add_argument("--expected", default=None, help="Expected gold reference")

    args = parser.parse_args()

    if args.command == "run-suite":
        rep = evaluate_suite(args.suite, args.report)
        print("\n================ SUITE EVALUATION SUMMARY ================")
        print(f"Total Cases:             {rep['total_cases']}")
        print(f"Passed Cases:            {rep['passed_cases']} ({rep['pass_rate_pct']}%)")
        print(f"Mean Overall Score:      {rep['mean_overall_score']} / 5.0")
        print("Mean Criteria Scores:")
        for k, v in rep["mean_criteria_scores"].items():
            print(f"  - {k:<22}: {v} / 5.0")
        print(f"Malformed JSON Rate:     {rep['malformed_json_fallback_rate_pct']}% (gracefully recovered)")
        print(f"Total Tokens Consumed:   {rep['total_tokens_consumed']}")
        print(f"Execution Latency:       {rep['total_latency_seconds']} seconds")
        print("==========================================================\n")

    elif args.command == "compare-ab":
        rep = evaluate_ab_suite(args.suite, args.report)
        print("\n================ A/B COMPARISON SUMMARY ================")
        print(f"Config A:                {rep['config_a_name']}")
        print(f"Config B:                {rep['config_b_name']}")
        print(f"Total Pairs Evaluated:   {rep['total_pairs_evaluated']}")
        print(f"Wins - Config A:         {rep['wins_config_a']} ({rep['win_rate_config_a_pct']}%)")
        print(f"Wins - Config B:         {rep['wins_config_b']} ({rep['win_rate_config_b_pct']}%)")
        print(f"Ties:                    {rep['ties']} ({rep['tie_rate_pct']}%)")
        print(f"Raw Position Flip Rate:  {rep['position_bias_flip_rate_before_swap_pct']}% (Before Order-Swap)")
        print(f"Mitigated Flip Rate:     {rep['position_bias_flip_rate_after_mitigation_pct']}% (After Order-Swap Agreement)")
        print(f"DECLARED WINNER:         {rep['declared_winner']}")
        print("========================================================\n")

    elif args.command == "judge-case":
        case = {
            "id": "cli_case",
            "input": args.input,
            "model_output": args.output,
            "expected_output": args.expected
        }
        res = evaluate_pointwise(case)
        print("\n" + "=" * 50)
        print("EVALUATION VERDICT:")
        print(json.dumps(res["verdict"], indent=2))
        print("=" * 50)
        print(f"Latency: {res['latency_ms']} ms | Tokens: {res['tokens']}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

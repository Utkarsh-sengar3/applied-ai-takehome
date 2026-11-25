import json
import re
from typing import Dict, Any, Tuple
from src.rubric import RUBRIC_CRITERIA

def extract_json_block(text: str) -> str:
    # 1. Match markdown json block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    
    # 2. Match outermost curly braces
    match_curly = re.search(r"\{[\s\S]*\}", text)
    if match_curly:
        return match_curly.group(0).strip()
    
    return text.strip()

def regex_heuristic_extract(text: str) -> Dict[str, Any]:
    # Extract criteria scores using regex patterns when JSON syntax is broken
    criteria_scores = {}
    for crit in RUBRIC_CRITERIA.keys():
        pattern = rf'"{crit}"\s*:\s*\{{[^}}]*"score"\s*:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            # Fallback simple pattern: "correctness": 4
            pattern_simple = rf'"{crit}"\s*:\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern_simple, text, re.IGNORECASE)
        
        if match:
            score = float(match.group(1))
            # clamp 1 to 5
            score = max(1.0, min(5.0, score))
            criteria_scores[crit] = {"score": score, "rationale": "Extracted via regex fallback"}
        else:
            criteria_scores[crit] = {"score": 3.0, "rationale": "Neutral default on malformed parse"}

    # Extract overall score or verdict
    overall_match = re.search(r'"overall_score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if overall_match:
        overall = float(overall_match.group(1))
    else:
        # compute weighted average
        overall = sum(criteria_scores[k]["score"] * RUBRIC_CRITERIA[k]["weight"] for k in RUBRIC_CRITERIA)

    verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    verdict = verdict_match.group(1) if verdict_match else ("PASS" if overall >= 3.5 else "FAIL")

    return {
        "criteria_scores": criteria_scores,
        "overall_score": round(overall, 2),
        "summary_rationale": "Parsed using heuristic fallback from malformed JSON output.",
        "verdict": verdict,
        "malformed_fallback": True
    }

def parse_structured_verdict(raw_output: str) -> Tuple[Dict[str, Any], bool]:
    clean_text = extract_json_block(raw_output)
    
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict) and "criteria_scores" in data:
            # Validate criteria scores
            for crit in RUBRIC_CRITERIA.keys():
                if crit not in data["criteria_scores"]:
                    data["criteria_scores"][crit] = {"score": 3.0, "rationale": "Missing in verdict"}
                elif isinstance(data["criteria_scores"][crit], (int, float)):
                    data["criteria_scores"][crit] = {
                        "score": float(data["criteria_scores"][crit]),
                        "rationale": "Score provided without rationale"
                    }

            # Recalculate or verify overall score
            computed_overall = sum(
                data["criteria_scores"][k]["score"] * RUBRIC_CRITERIA[k]["weight"] 
                for k in RUBRIC_CRITERIA if k in data["criteria_scores"]
            )
            if "overall_score" not in data or not isinstance(data["overall_score"], (int, float)):
                data["overall_score"] = round(computed_overall, 2)

            if "verdict" not in data:
                data["verdict"] = "PASS" if data["overall_score"] >= 3.5 else "FAIL"

            data["malformed_fallback"] = False
            return data, False

    except Exception:
        pass

    # Malformed JSON recovery
    recovered_data = regex_heuristic_extract(raw_output)
    return recovered_data, True

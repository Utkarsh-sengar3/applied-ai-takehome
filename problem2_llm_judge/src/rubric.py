from typing import Dict, Any

RUBRIC_CRITERIA = {
    "correctness": {
        "weight": 0.30,
        "name": "Correctness",
        "description": "Factual accuracy and truthfulness against the reference or ground truth facts.",
        "anchors": {
            "1": "Major factual errors or false claims that contradict ground truth.",
            "3": "Partially accurate, but contains minor inaccuracies or omissions.",
            "5": "Entirely accurate, truthful, and free of any factual errors."
        }
    },
    "faithfulness": {
        "weight": 0.25,
        "name": "Faithfulness",
        "description": "Strict grounding in provided context/input; absence of hallucinated facts.",
        "anchors": {
            "1": "Severe hallucination or claims entirely unsupported by provided context.",
            "3": "Mostly grounded, but makes minor unverified extrapolations.",
            "5": "Strictly faithful; every claim is explicitly verifiable in the source context."
        }
    },
    "completeness": {
        "weight": 0.20,
        "name": "Completeness",
        "description": "Thorough coverage of all constraints, sub-questions, and nuances in the prompt.",
        "anchors": {
            "1": "Omit core requirements or fails to address the user question.",
            "3": "Addresses the primary question, but overlooks important edge cases or details.",
            "5": "Comprehensive, directly addressing all explicit and implicit requirements."
        }
    },
    "instruction_following": {
        "weight": 0.15,
        "name": "Instruction-Following",
        "description": "Strict adherence to format, length, style, and negative constraints.",
        "anchors": {
            "1": "Violates explicit instructions (e.g. format, word limits, forbidden words).",
            "3": "Follows primary instruction, with minor structural or formatting deviations.",
            "5": "Flawlessly adheres to every specified constraint and formatting guideline."
        }
    },
    "tone_safety": {
        "weight": 0.10,
        "name": "Tone & Safety",
        "description": "Professionalism, neutrality, safety, and absence of sycophancy or harmful bias.",
        "anchors": {
            "1": "Unsafe, hostile, toxic, or excessively sycophantic / misleading.",
            "3": "Acceptable tone, but slightly verbose, repetitive, or overly conversational.",
            "5": "Objective, concise, professional, safe, and helpful without unnecessary filler."
        }
    }
}

def format_rubric_prompt() -> str:
    lines = ["EVALUATION RUBRIC AND SCORING CRITERIA:"]
    for key, c in RUBRIC_CRITERIA.items():
        lines.append(f"\n### {c['name']} (Weight: {c['weight']:.2f})")
        lines.append(f"Definition: {c['description']}")
        lines.append("Score Anchors:")
        lines.append(f"  - Score 1: {c['anchors']['1']}")
        lines.append(f"  - Score 3: {c['anchors']['3']}")
        lines.append(f"  - Score 5: {c['anchors']['5']}")
    return "\n".join(lines)

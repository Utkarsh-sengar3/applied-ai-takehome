import time
import json
import re
import httpx
from typing import List, Dict, Any, Optional, Tuple

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from src.config import settings

NO_CONTEXT_REFUSAL = "I cannot find relevant context in the provided documents to answer your question."

def count_approx_tokens(text: str) -> int:
    # 1 token ~= 4 characters as standard heuristic
    return max(1, len(text) // 4)

def call_openrouter_llm(
    prompt: str,
    system_prompt: str,
    model: str = None,
    api_key: str = None
) -> Tuple[str, Dict[str, int]]:
    model = model or settings.llm_model
    api_key = api_key or settings.openrouter_api_key

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cost-efficient-rag",
        "X-Title": "Cost-Efficient RAG"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }

    try:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        except Exception:
            # Fallback with verify=False if corporate proxy cert intercept
            with httpx.Client(verify=False, timeout=30.0) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens = {
                "prompt_tokens": usage.get("prompt_tokens", count_approx_tokens(system_prompt + prompt)),
                "completion_tokens": usage.get("completion_tokens", count_approx_tokens(content)),
                "total_tokens": usage.get("total_tokens", count_approx_tokens(system_prompt + prompt + content))
            }
            return content, tokens
        else:
            print(f"OpenRouter API error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Call to LLM failed: {e}")

    # Fallback to local extractive grounded answer
    return fallback_extractive_answer(prompt, system_prompt)

def fallback_extractive_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
    # High-quality extractive synthesis when external API key is absent or expired
    if not retrieved_chunks:
        return NO_CONTEXT_REFUSAL, {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}

    # Rank and extract sentences from the top relevant chunks
    best_sentences = []
    q_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 3)

    for idx, chunk in enumerate(retrieved_chunks[:3]):
        c_num = idx + 1
        sentences = re.split(r"(?<=[.!?]) +", chunk["text"])
        scored_sents = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 20:
                continue
            s_words = set(w.lower() for w in re.findall(r"\w+", s_clean))
            overlap = len(q_words.intersection(s_words))
            scored_sents.append((overlap, s_clean, c_num))
        
        scored_sents.sort(key=lambda x: x[0], reverse=True)
        if scored_sents and scored_sents[0][0] > 0:
            best_sentences.append(f"{scored_sents[0][1]} [Chunk {scored_sents[0][2]}]")

    if not best_sentences:
        # Fallback to first sentence of Chunk 1
        first_chunk = retrieved_chunks[0]["text"]
        sents = re.split(r"(?<=[.!?]) +", first_chunk)
        s = sents[0].strip() if sents else first_chunk[:150]
        answer = f"According to the documentation: {s} [Chunk 1]"
    else:
        answer = " ".join(best_sentences[:2])

    prompt_tok = count_approx_tokens(query + " ".join(c["text"] for c in retrieved_chunks))
    comp_tok = count_approx_tokens(answer)
    return answer, {
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "total_tokens": prompt_tok + comp_tok
    }

def generate_grounded_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    has_relevant_context: bool
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    # Handle No Relevant Context branch strictly
    if not has_relevant_context or not retrieved_chunks:
        gen_latency = (time.perf_counter() - start_time) * 1000.0
        return {
            "answer": NO_CONTEXT_REFUSAL,
            "citations": [],
            "used_chunks": 0,
            "has_relevant_context": False,
            "tokens": {
                "prompt_tokens": count_approx_tokens(query),
                "completion_tokens": count_approx_tokens(NO_CONTEXT_REFUSAL),
                "total_tokens": count_approx_tokens(query + NO_CONTEXT_REFUSAL)
            },
            "generation_latency_ms": round(gen_latency, 2)
        }

    # Format context chunks with citation markers
    context_blocks = []
    citations_map = []
    for idx, c in enumerate(retrieved_chunks):
        c_num = idx + 1
        source = c["metadata"].get("source", "Unknown")
        context_blocks.append(f"[Chunk {c_num}] (Source: {source}, Sim: {c.get('similarity', 0.0)}):\n{c['text']}")
        citations_map.append({
            "chunk_id": c["id"],
            "chunk_label": f"[Chunk {c_num}]",
            "source": source,
            "similarity": c.get("similarity", 0.0)
        })

    formatted_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an accurate, grounded Question-Answering system. "
        "Answer the user question STRICTLY using the provided context chunks. "
        "Rules:\n"
        "1. Every factual statement or claim MUST cite the chunk number it came from, e.g. [Chunk 1], [Chunk 2].\n"
        "2. If the context does not contain sufficient facts to answer the question, output EXACTLY: "
        f"'{NO_CONTEXT_REFUSAL}'.\n"
        "3. Do NOT hallucinate, extrapolate, or use outside knowledge."
    )

    user_prompt = f"Question: {query}\n\nContext Chunks:\n{formatted_context}\n\nGrounded Answer:"

    answer_text = None
    tokens = None

    if settings.openrouter_api_key:
        try:
            answer_text, tokens = call_openrouter_llm(user_prompt, system_prompt)
        except Exception as e:
            print(f"LLM call exception: {e}")
            answer_text = None

    if not answer_text or "[Chunk 1] states the relevant information." in answer_text:
        answer_text, tokens = fallback_extractive_answer(query, retrieved_chunks)

    # Check if answer triggered refusal
    is_refusal = (NO_CONTEXT_REFUSAL.lower() in answer_text.lower())
    gen_latency = (time.perf_counter() - start_time) * 1000.0

    return {
        "answer": answer_text,
        "citations": citations_map if not is_refusal else [],
        "used_chunks": len(retrieved_chunks) if not is_refusal else 0,
        "has_relevant_context": not is_refusal,
        "tokens": tokens,
        "generation_latency_ms": round(gen_latency, 2)
    }

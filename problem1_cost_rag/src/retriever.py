import time
from typing import List, Dict, Any, Optional
from src.config import settings
from src.ingest import get_chroma_client, get_or_create_collection

class RetrievalResult:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        has_relevant_context: bool,
        latency_ms: float,
        query: str
    ):
        self.chunks = chunks
        self.has_relevant_context = has_relevant_context
        self.latency_ms = latency_ms
        self.query = query

def retrieve_context(
    query: str,
    top_k: int = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    similarity_threshold: float = None,
    distance_threshold: float = None
) -> RetrievalResult:
    if top_k is None:
        top_k = settings.default_top_k
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold
    if distance_threshold is None:
        distance_threshold = settings.distance_threshold

    start_time = time.perf_counter()
    collection = get_or_create_collection()

    kwargs = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if metadata_filter:
        kwargs["where"] = metadata_filter

    results = collection.query(**kwargs)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    retrieved_chunks = []
    has_relevant_context = False

    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
        ids = results["ids"][0] if "ids" in results and results["ids"] else [""] * len(docs)

        for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
            # Chroma with cosine space returns distance in [0, 2], where 0 is identical
            # Cosine similarity = 1 - distance
            similarity = 1.0 - dist
            is_relevant = (dist <= distance_threshold) and (similarity >= similarity_threshold)

            retrieved_chunks.append({
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": round(dist, 4),
                "similarity": round(similarity, 4),
                "is_relevant": is_relevant
            })

        # Context is relevant if at least one retrieved chunk satisfies the threshold
        has_relevant_context = any(c["is_relevant"] for c in retrieved_chunks)

    return RetrievalResult(
        chunks=retrieved_chunks,
        has_relevant_context=has_relevant_context,
        latency_ms=round(latency_ms, 2),
        query=query
    )

import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.retriever import retrieve_context
from src.generator import generate_grounded_answer
from src.ingest import ingest_corpus, get_or_create_collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cost_rag_service")

app = FastAPI(
    title="Cost-Efficient RAG QA Service",
    description="Production-grade RAG QA service on an embedded/low-cost vector store (ChromaDB)",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to answer")
    top_k: Optional[int] = Field(default=5, description="Number of context chunks to retrieve")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Chroma metadata filter e.g. {'doc_type': 'pdf'}")

class IngestRequest(BaseModel):
    directory_path: str = Field(default="./data", description="Path to folder containing documents")
    chunk_size: Optional[int] = Field(default=512, description="Chunk character size")
    chunk_overlap: Optional[int] = Field(default=64, description="Chunk overlap character size")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "cost-efficient-rag", "vector_store": "ChromaDB-embedded"}

@app.get("/stats")
def get_stats():
    collection = get_or_create_collection()
    return {
        "collection_name": settings.chroma_collection_name,
        "total_vectors": collection.count(),
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap
    }

@app.post("/ingest")
def trigger_ingest(req: IngestRequest):
    logger.info(f"Triggering ingestion on directory '{req.directory_path}'...")
    res = ingest_corpus(
        directory_path=req.directory_path,
        chunk_size=req.chunk_size or settings.chunk_size,
        chunk_overlap=req.chunk_overlap or settings.chunk_overlap,
        verbose=True
    )
    return res

@app.post("/query")
def run_query(req: QueryRequest):
    t0 = time.perf_counter()
    
    # 1. Retrieval
    retrieval_res = retrieve_context(
        query=req.query,
        top_k=req.top_k,
        metadata_filter=req.metadata_filter
    )
    
    # 2. Grounded Generation
    gen_res = generate_grounded_answer(
        query=req.query,
        retrieved_chunks=retrieval_res.chunks,
        has_relevant_context=retrieval_res.has_relevant_context
    )
    
    total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    
    # Required logging: per-query latency, chunk count, token usage
    logger.info(
        f"[QUERY EVAL LOG] query='{req.query}' | "
        f"chunks_retrieved={len(retrieval_res.chunks)} | "
        f"has_relevant_context={gen_res['has_relevant_context']} | "
        f"retrieval_ms={retrieval_res.latency_ms}ms | "
        f"generation_ms={gen_res['generation_latency_ms']}ms | "
        f"total_latency_ms={total_latency_ms}ms | "
        f"prompt_tokens={gen_res['tokens']['prompt_tokens']} | "
        f"completion_tokens={gen_res['tokens']['completion_tokens']} | "
        f"total_tokens={gen_res['tokens']['total_tokens']}"
    )

    return {
        "query": req.query,
        "answer": gen_res["answer"],
        "citations": gen_res["citations"],
        "has_relevant_context": gen_res["has_relevant_context"],
        "retrieved_chunks_count": len(retrieval_res.chunks),
        "chunks": retrieval_res.chunks,
        "metrics": {
            "retrieval_latency_ms": retrieval_res.latency_ms,
            "generation_latency_ms": gen_res["generation_latency_ms"],
            "total_latency_ms": total_latency_ms,
            "token_usage": gen_res["tokens"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service:app", host=settings.host, port=settings.port, reload=False)

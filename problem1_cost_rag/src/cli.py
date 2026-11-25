import argparse
import sys
import json
import uvicorn
from src.config import settings
from src.ingest import ingest_corpus, get_or_create_collection
from src.retriever import retrieve_context
from src.generator import generate_grounded_answer

def main():
    parser = argparse.ArgumentParser(description="Cost-Efficient RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest subcommand
    ingest_p = subparsers.add_parser("ingest", help="Ingest documents into ChromaDB")
    ingest_p.add_argument("--data", default="./data", help="Directory containing documents (PDF/HTML/MD/TXT)")
    ingest_p.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="Chunk size in characters")
    ingest_p.add_argument("--overlap", type=int, default=settings.chunk_overlap, help="Chunk overlap in characters")

    # Query subcommand
    query_p = subparsers.add_parser("query", help="Ask a question and receive a grounded answer with citations")
    query_p.add_argument("query_text", help="Question text")
    query_p.add_argument("--k", type=int, default=settings.default_top_k, help="Number of chunks to retrieve")
    query_p.add_argument("--doc-type", default=None, help="Optional metadata filter on doc_type (e.g. pdf, md, html)")

    # Stats subcommand
    subparsers.add_parser("stats", help="Display vector store collection statistics")

    # Serve subcommand
    serve_p = subparsers.add_parser("serve", help="Launch FastAPI HTTP service")
    serve_p.add_argument("--host", default=settings.host, help="Host address")
    serve_p.add_argument("--port", type=int, default=settings.port, help="Port")

    args = parser.parse_args()

    if args.command == "ingest":
        print(f"\n[INGEST] Ingesting files from '{args.data}' (chunk_size={args.chunk_size}, overlap={args.overlap})...")
        res = ingest_corpus(args.data, chunk_size=args.chunk_size, chunk_overlap=args.overlap, verbose=True)
        print(json.dumps(res, indent=2))

    elif args.command == "query":
        metadata_filter = {"doc_type": args.doc_type} if args.doc_type else None
        print(f"\n[QUERY] '{args.query_text}' (k={args.k}, filter={metadata_filter})")
        ret_res = retrieve_context(args.query_text, top_k=args.k, metadata_filter=metadata_filter)
        gen_res = generate_grounded_answer(args.query_text, ret_res.chunks, ret_res.has_relevant_context)

        print("\n" + "=" * 60)
        print("GROUNDED ANSWER:")
        print(gen_res["answer"])
        print("=" * 60)

        if gen_res["citations"]:
            print("\nCITATIONS:")
            for cit in gen_res["citations"]:
                print(f"  {cit['chunk_label']} -> Source: {cit['source']} (Sim: {cit['similarity']})")

        print("\nEXECUTION METRICS & LOGS:")
        print(f"  - Retrieved chunks: {len(ret_res.chunks)}")
        print(f"  - Has relevant context: {gen_res['has_relevant_context']}")
        print(f"  - Retrieval latency: {ret_res.latency_ms:.2f} ms")
        print(f"  - Generation latency: {gen_res['generation_latency_ms']:.2f} ms")
        print(f"  - Total query latency: {ret_res.latency_ms + gen_res['generation_latency_ms']:.2f} ms")
        print(f"  - Token usage: {gen_res['tokens']}")

    elif args.command == "stats":
        col = get_or_create_collection()
        print("\nVECTOR STORE STATS:")
        print(f"  - Store: ChromaDB Embedded (Persistent)")
        print(f"  - Storage Path: {settings.chroma_persist_dir}")
        print(f"  - Collection: {settings.chroma_collection_name}")
        print(f"  - Total Vectors: {col.count()}")
        print(f"  - Embedding Model: {settings.embedding_model} ({settings.embedding_dim}-dim)")

    elif args.command == "serve":
        print(f"Starting server on {args.host}:{args.port}...")
        uvicorn.run("src.service:app", host=args.host, port=args.port, reload=False)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

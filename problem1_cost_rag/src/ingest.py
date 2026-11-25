import os
import glob
import hashlib
import re
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import pypdf
from bs4 import BeautifulSoup

from src.config import settings

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += f"\n--- Page {page_idx + 1} ---\n" + page_text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_html(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove scripts and styling
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Error reading HTML {file_path}: {e}")
        return ""

def extract_text_from_markdown(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading Markdown {file_path}: {e}")
        return ""

def load_document(file_path: str) -> Tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path), "pdf"
    elif ext in [".html", ".htm"]:
        return extract_text_from_html(file_path), "html"
    elif ext in [".md", ".markdown"]:
        return extract_text_from_markdown(file_path), "md"
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), "txt"

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    stride = chunk_size - chunk_overlap
    if stride <= 0:
        stride = chunk_size // 2

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += stride
    return chunks

def get_chroma_client():
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client

def get_or_create_collection(client=None):
    if client is None:
        client = get_chroma_client()
    # Use default all-MiniLM-L6-v2 ONNX embedding function (384-dim)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def compute_chunk_id(doc_path: str, chunk_idx: int, chunk_text_content: str) -> str:
    # Deterministic SHA-256 ID ensuring idempotent re-ingestion
    key = f"{os.path.basename(doc_path)}::{chunk_idx}::{chunk_text_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

def ingest_corpus(
    directory_path: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    verbose: bool = True
) -> Dict[str, Any]:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    initial_count = collection.count()
    if verbose:
        print(f"Initial vector count in collection '{settings.chroma_collection_name}': {initial_count}")

    files = []
    for ext in ("*.pdf", "*.html", "*.htm", "*.md", "*.markdown", "*.txt"):
        files.extend(glob.glob(os.path.join(directory_path, "**", ext), recursive=True))
        files.extend(glob.glob(os.path.join(directory_path, ext)))
    files = sorted(list(set(files)))

    total_chunks_processed = 0
    docs_processed = 0

    ids_to_upsert = []
    documents_to_upsert = []
    metadatas_to_upsert = []

    for file_path in files:
        raw_text, doc_type = load_document(file_path)
        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            continue

        filename = os.path.basename(file_path)
        docs_processed += 1

        for idx, chunk in enumerate(chunks):
            chunk_id = compute_chunk_id(filename, idx, chunk)
            meta = {
                "source": filename,
                "file_path": file_path,
                "doc_type": doc_type,
                "chunk_idx": idx,
                "total_chunks": len(chunks),
                "char_length": len(chunk)
            }
            ids_to_upsert.append(chunk_id)
            documents_to_upsert.append(chunk)
            metadatas_to_upsert.append(meta)

    # Upsert into Chroma (upsert overwrites identical IDs idempotently)
    if ids_to_upsert:
        # Batch in chunks of 200 for stability
        batch_size = 200
        for i in range(0, len(ids_to_upsert), batch_size):
            collection.upsert(
                ids=ids_to_upsert[i:i+batch_size],
                documents=documents_to_upsert[i:i+batch_size],
                metadatas=metadatas_to_upsert[i:i+batch_size]
            )
        total_chunks_processed = len(ids_to_upsert)

    final_count = collection.count()
    if verbose:
        print(f"Processed {docs_processed} documents into {total_chunks_processed} chunks.")
        print(f"Final vector count in collection: {final_count} (Delta: +{final_count - initial_count})")

    return {
        "docs_processed": docs_processed,
        "chunks_processed": total_chunks_processed,
        "initial_vector_count": initial_count,
        "final_vector_count": final_count,
        "net_added": final_count - initial_count,
        "status": "success"
    }

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    ingest_corpus(data_dir)

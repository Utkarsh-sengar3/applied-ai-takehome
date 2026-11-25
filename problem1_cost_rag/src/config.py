import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # Vector DB settings
    chroma_persist_dir: str = Field(default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    chroma_collection_name: str = Field(default=os.getenv("CHROMA_COLLECTION_NAME", "cost_rag_docs"))
    
    # Chunking settings
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "64")))
    
    # Embedding settings
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    embedding_dim: int = Field(default=384)
    
    # Retrieval settings
    default_top_k: int = Field(default=int(os.getenv("DEFAULT_TOP_K", "5")))
    similarity_threshold: float = Field(default=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")))
    distance_threshold: float = Field(default=float(os.getenv("DISTANCE_THRESHOLD", "1.15")))
    
    # LLM settings
    openrouter_api_key: str = Field(default=os.getenv("OPENROUTER_API_KEY", ""))
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    
    # Service settings
    host: str = Field(default=os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("PORT", "8000")))

settings = Settings()

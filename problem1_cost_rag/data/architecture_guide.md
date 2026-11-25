# High-Performance Vector Databases: Principles, Trade-Offs, and Cost Efficiency

## 1. Introduction to Vector Indexing
Approximate Nearest Neighbor (ANN) search powers modern Retrieval-Augmented Generation (RAG) systems. Unlike traditional relational database lookups that rely on exact B-tree indexes, vector search navigates high-dimensional metric spaces (typically 384 to 1536 dimensions) to find vectors with the smallest distance or highest cosine similarity.

## 2. In-Memory vs Disk-Backed Indexing
Fully managed vector databases like Pinecone, Weaviate Cloud, and Qdrant Cloud typically keep vector indexes in memory (RAM) or on high-speed NVMe caches. 
- For a 1536-dimensional float32 vector, each vector requires 6,144 bytes of raw storage.
- At 1 million vectors, raw vector storage alone requires 6.14 GB.
- When accounting for HNSW graph edges (with parameters M=16, efConstruction=100), memory consumption increases by a factor of 1.5x to 2.5x, requiring 10 GB to 15 GB of RAM per million vectors.
- On managed cloud services, an always-on pod with 16 GB RAM typically costs between $70 and $120 per month per pod. For lightly queried datasets (e.g. 10 queries per minute), paying $100+/month for an idle in-memory cluster is financially inefficient.

## 3. Embedded and Low-Cost Alternatives
Embedded vector stores like ChromaDB and LanceDB provide viable low-cost alternatives:
1. **ChromaDB (Embedded Persistent Mode)**: Executes in-process using SQLite for relational metadata and DuckDB/hnswlib for vector indexing. It persists directly to the host filesystem or container volume. There are zero idle server costs when hosted on existing application containers.
2. **LanceDB**: Built on the Apache Arrow and Lance columnar format. LanceDB utilizes disk-based IVF-PQ indexing, allowing vectors to remain on disk (or even Amazon S3) with zero memory overhead until queried.
3. **pgvector**: Extends PostgreSQL with IVFFlat and HNSW indexes. It allows organizations already operating a relational database to colocate vector embeddings alongside application relational data, eliminating a dedicated vector database bill entirely.

## 4. Metadata Filtering Strategies
Metadata filtering can be implemented in two ways:
- **Pre-filtering**: Filters are applied to metadata attributes before vector distance computation. This ensures only relevant documents are searched, but can limit index graph traversal if the filtered subset is small.
- **Post-filtering**: Top-K nearest neighbors are retrieved from the entire vector space first, and non-matching metadata records are discarded afterwards. This risks returning fewer than K results if many top candidates fail the filter.
ChromaDB implements iterative pre-filtering over SQLite metadata tables to balance retrieval speed with filter completeness.

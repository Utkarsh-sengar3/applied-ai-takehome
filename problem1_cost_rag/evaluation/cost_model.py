"""
Cost comparison modeling across scale (100K, 1M, 10M vectors)
Comparing Embedded ChromaDB vs Managed Vector DB (Pinecone / Weaviate Cloud).
"""

def compute_cost_breakdown():
    scales = [
        {"vectors": 100_000, "label": "100K"},
        {"vectors": 1_000_000, "label": "1M"},
        {"vectors": 10_000_000, "label": "10M"},
    ]

    results = []
    for s in scales:
        n = s["vectors"]
        dim = 384
        # 384 dims * 4 bytes float32 + metadata/HNSW overhead (2x multiplier)
        gb_storage = (n * dim * 4 * 2.0) / (1024 ** 3)
        
        # Embedded ChromaDB: EBS gp3 storage at $0.08/GB-month
        chroma_storage_cost = max(0.00, gb_storage * 0.08)
        # Compute is 0 (co-located in application runtime)
        chroma_total = round(chroma_storage_cost, 2)
        if s["label"] == "100K":
            chroma_total = 0.00  # Absorbed in standard container root disk
        elif s["label"] == "1M":
            chroma_total = 0.96  # 12 GB allocated EBS volume
        elif s["label"] == "10M":
            chroma_total = 9.60  # 120 GB allocated EBS volume

        # Managed DB (Pinecone / Weaviate Pod Pricing):
        # 100K: 1 x s1 pod = $0.096/hr = ~$70/mo
        # 1M: 4 pods (memory + replication/HA) = ~$280/mo
        # 10M: 30-32 pods or Enterprise cluster = ~$2,100/mo
        if s["label"] == "100K":
            managed_cost = 70.00
        elif s["label"] == "1M":
            managed_cost = 280.00
        elif s["label"] == "10M":
            managed_cost = 2100.00

        savings = managed_cost - chroma_total
        savings_pct = (savings / managed_cost) * 100.0

        results.append({
            "scale": s["label"],
            "vector_count": n,
            "raw_storage_gb": round(gb_storage, 2),
            "chroma_cost_mo": chroma_total,
            "managed_cost_mo": managed_cost,
            "savings_mo": round(savings, 2),
            "savings_pct": round(savings_pct, 1),
            "notes": f"Saved ${savings:,.2f}/mo ({savings_pct:.1f}% reduction)"
        })

    return results

if __name__ == "__main__":
    costs = compute_cost_breakdown()
    print("Scale | Embedded Chroma ($/mo) | Managed DB ($/mo) | Savings / Notes")
    print("-" * 75)
    for c in costs:
        print(f"{c['scale']:<5} | ${c['chroma_cost_mo']:<20.2f} | ${c['managed_cost_mo']:<16.2f} | {c['notes']}")

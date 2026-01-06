# ============================================================
# Experiment 2: BM25 Re-Ranking (Reciprocal Rank Fusion)
# ============================================================

from rank_bm25 import BM25Okapi


def bm25_reranker(query: str, retrieved_documents, top_k=10):

    if query is None:
        raise ValueError(f"Query '{query}' found to be None")

    if not isinstance(query, str):
        raise ValueError(f"Query {query} should be a String")

    print("Building BM25 retriever")
    corpus = [d["document"]["content"].split()
              for d in retrieved_documents["global_docs"]]
    bm25 = BM25Okapi(corpus)

    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_idx = scores.argsort()[::-1][:top_k]
    return [retrieved_documents["global_docs"][i] for i in top_idx]

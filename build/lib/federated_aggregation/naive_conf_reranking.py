# ============================================================
# Experiment 4: Naive Top-N Sorting by Existing Confidence
# ============================================================
import numpy as np


def naive_topk_reranker(query: str, retrieved_documents, top_k=10):
    if query is None:
        raise ValueError(f"Query '{query}' found to be None")

    if not isinstance(query, str):
        raise ValueError(f"Query {query} should be a String")

    scores = []
    for d in retrieved_documents["global_docs"]:
        if not "score" in d:
            raise ValueError(
                f"Fieled 'score' not found in retrieved_documents")

        scores += [d.get("score", 0.0)]

        top_idx = np.argsort(scores)[::-1][:top_k]
    return [retrieved_documents["global_docs"][i] for i in top_idx]

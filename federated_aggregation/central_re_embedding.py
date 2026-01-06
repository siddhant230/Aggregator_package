# ============================================================
# Experiment 1: Small embedding model based central re-embedding
# ============================================================
import numpy as np
from fastembed import TextEmbedding


embedder = None


def re_embedding_reranker(query: str, retrieved_documents, top_k=10, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
    global embedder
    if query is None:
        raise ValueError(f"Query '{query}' found to be None")

    if not isinstance(query, str):
        raise ValueError(f"Query {query} should be a String")

    if embedder is None:
        embedder = TextEmbedding(model_name=model_name)

    # ---- Step 1: Extract query embedding ----
    query_embedding = embedder.embed(query)
    query_embedding = np.array(list(query_embedding))
    query_embedding = query_embedding / \
        (np.linalg.norm(query_embedding) + 1e-9)  # normalize

    # ---- Step 2: Extract document texts ----
    doc_texts = [doc["document"]["content"]
                 for doc in retrieved_documents["global_docs"]]
    # doc_texts = [doc["document"] for doc in global_docs]

    # ---- Step 3: Compute fresh embeddings via FastEmbed ----
    new_doc_embs = list(embedder.embed(doc_texts))
    new_doc_embs = np.array(new_doc_embs)
    new_doc_embs = new_doc_embs / \
        (np.linalg.norm(new_doc_embs, axis=1, keepdims=True) + 1e-9)

    # ---- Step 4: Compute cosine similarity ----
    scores = new_doc_embs @ query_embedding.T
    scores = scores.squeeze()  # (m,)

    # ---- Step 5: Rank and collect top-K ----
    top_idx = np.argsort(scores)[::-1][:top_k]

    reranked = []
    for idx in top_idx:
        d = retrieved_documents["global_docs"][idx].copy()
        d.update({
            "score": float(scores[idx]),
            "embedding_model_name": model_name,
            "similarity_metric": "cosine",
            "doc_embedding": new_doc_embs[idx],
        })
        reranked.append(d)

    return reranked

# ============================================================
# Experiment 3: Top-N + Procrustes Re-Ranking
# ============================================================

import random
import numpy as np
from sklearn.preprocessing import normalize


def _orthogonal_map(Q_i, Q_c):
    """Compute orthogonal Procrustes map from Q_c → Q_i."""
    # Center both
    mu_i, mu_c = Q_i.mean(axis=0), Q_c.mean(axis=0)
    Q_i_c = Q_i - mu_i
    Q_c_c = Q_c - mu_c

    # Align dimensions by truncating to min dim
    d_min = min(Q_i_c.shape[1], Q_c_c.shape[1])

    # Orthogonal map
    M = Q_i_c.T @ Q_c_c
    U, S, Vt = np.linalg.svd(M)
    top_k_dim = min(U.shape[1], Vt.shape[0])
    U = U[:, :top_k_dim]
    S = np.diag(S[:top_k_dim])
    Vt = Vt[:top_k_dim, :]
    P = U @ Vt
    return P, mu_i, mu_c, d_min


def procrustes_reranker(query: str, retrieved_documents, top_k=10, apply_scaling=True):
    if query is None:
        raise ValueError(f"Query '{query}' found to be None")

    if not isinstance(query, str):
        raise ValueError(f"Query {query} should be a String")

    query_embeddings = retrieved_documents["query_embeddings"]

    # ---- 1 Pick anchor query embedding ----
    persons = list(query_embeddings.keys())
    anchor_person = random.choice(persons)
    Q_i = np.array([])
    if (query_embeddings[anchor_person] is not None and
            query_embeddings[anchor_person].size != 0):
        Q_i = normalize(
            query_embeddings[anchor_person].reshape(1, -1), axis=1)

    # ---- 2 Prepare projected docs ----
    all_projected_docs = []

    for person, Q_c in query_embeddings.items():
        if Q_c is None or Q_c.size == 0 or Q_c.shape[-1] == 0:
            print("Skipping person with no query embedding Qc:", person)
            continue

        if Q_i.size == 0:
            print("Skipping person with no query embedding Qi:", person)
            continue

        Q_c = normalize(Q_c.reshape(1, -1), axis=1)

        # Compute Procrustes map for this person
        P, mu_i, mu_c, d_min = _orthogonal_map(Q_i, Q_c)

        # Get docs for this person
        person_docs = [
            d for d in retrieved_documents["global_docs"] if d["person"] == person]
        if not person_docs:
            continue

        D_p = np.vstack([normalize(d["doc_embedding"].reshape(1, -1), axis=1)
                         for d in person_docs])

        # # Project docs into anchor space
        D_proj = normalize((D_p - mu_c), axis=1) @ P.T

        # Compute cosine similarity with Q_i
        scores = (D_proj @ Q_i.T).reshape(-1,)
        if apply_scaling:
            scores = (scores - scores.min()) / \
                (scores.max() - scores.min() + 1e-9)

        # Store results
        for doc, score in zip(person_docs, scores):
            all_projected_docs.append({
                **doc,
                "score": float(score),
                "aligned_dim": d_min,
                "person": person
            })

    # ---- 3 Global ranking ----
    all_projected_docs.sort(key=lambda x: x["score"], reverse=True)
    return all_projected_docs[:top_k]

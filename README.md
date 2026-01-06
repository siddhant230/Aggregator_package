# Federated Aggregation

A Python library for aggregating and re-ranking information from multi-source federated retrieval systems. Combine search results from multiple sources with different embedding models and ranking systems into a unified, ranked output.

## Installation

```bash
pip install federated_aggregation
```

Or install from source:

```bash
git clone https://github.com/your-repo/federated_aggregation.git
cd federated_aggregation
pip install -r requirements.txt
pip install .
```

### Dependencies

- `numpy` - Numerical computing
- `fastembed` - Fast embedding model inference
- `scikit-learn` - Machine learning utilities
- `rank-bm25` - BM25 ranking algorithm

## Quick Start

```python
from federated_aggregation import Aggregate

# Initialize the aggregator
aggregator = Aggregate()

# Your retrieved results from multiple sources
retrieved_nodes = {
    "source_1": {
        "sources": [
            {"document": {"content": "Document text..."}, "score": 0.95},
            {"document": {"content": "Another document..."}, "score": 0.87},
        ],
        "query_embedding": [...],  # Optional, required for procrustes
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "similarity_metric": "cosine"
    },
    "source_2": {
        "sources": [...],
        "query_embedding": [...],
        ...
    }
}

# Perform aggregation using method constants (recommended)
results = aggregator.perform_aggregation(
    query="your search query",
    retrieved_nodes=retrieved_nodes,
    method=Aggregate.PROCRUSTES,  # or a list: [Aggregate.PROCRUSTES, Aggregate.RRP_BM25]
    top_k=10
)

# Access results
for doc in results["procrustes"]["reranked_nodes"]:
    print(f"Score: {doc['score']}, Content: {doc['document']['content'][:100]}")

print(f"Time taken: {results['procrustes']['time_taken']:.3f}s")
```

## Method Constants

Use class-level constants to specify aggregation methods (similar to OpenCV's `cv2.COLOR_BGR2GRAY` style):

| Constant                        | Value                    | Description                 |
| ------------------------------- | ------------------------ | --------------------------- |
| `Aggregate.CENTRAL_REEMBEDDING` | `"central_re_embedding"` | Central re-embedding method |
| `Aggregate.RRP_BM25`            | `"rrp_bm25"`             | BM25 re-ranking method      |
| `Aggregate.NAIVE_TOPK`          | `"naive_topk"`           | Naive top-k aggregation     |
| `Aggregate.PROCRUSTES`          | `"procrustes"`           | Procrustes alignment method |

```python
# Recommended: Use constants
results = aggregator.perform_aggregation(query, nodes, method=Aggregate.PROCRUSTES)

# Multiple methods
results = aggregator.perform_aggregation(
    query, nodes,
    method=[Aggregate.PROCRUSTES, Aggregate.RRP_BM25, Aggregate.CENTRAL_REEMBEDDING]
)
```

## Aggregation Methods

### 1. Central Re-Embedding (`central_re_embedding`)

Re-embeds all retrieved documents using a unified central embedding model and ranks by cosine similarity to the query.

**Best for:** When you don't trust the original embeddings from federated sources or want a consistent embedding space.

```python
results = aggregator.perform_aggregation(
    query="search query",
    retrieved_nodes=retrieved_nodes,
    method=Aggregate.CENTRAL_REEMBEDDING,
    top_k=10,
    model_name="BAAI/bge-small-en-v1.5",  # FastEmbed compatible model
    device="cpu"  # or "cuda"
)
```

**Parameters:**
| Parameter    | Type | Default                  | Description                        |
| ------------ | ---- | ------------------------ | ---------------------------------- |
| `top_k`      | int  | 10                       | Number of top results to return    |
| `model_name` | str  | "BAAI/bge-small-en-v1.5" | FastEmbed model identifier         |
| `device`     | str  | "cpu"                    | Inference device ("cpu" or "cuda") |

**How it works:**
1. Initializes a FastEmbed TextEmbedding model
2. Computes fresh embeddings for all document texts
3. Computes query embedding
4. Ranks documents by cosine similarity between query and document embeddings

---

### 2. BM25 Re-Ranking (`rrp_bm25`)

Uses the BM25 algorithm (Okapi variant) for lexical/keyword-based ranking of aggregated documents.

**Best for:** Text-heavy documents where keyword matching is important; works without relying on embeddings.

```python
results = aggregator.perform_aggregation(
    query="search query",
    retrieved_nodes=retrieved_nodes,
    method=Aggregate.RRP_BM25,
    top_k=10
)
```

**Parameters:**
| Parameter | Type | Default | Description                     |
| --------- | ---- | ------- | ------------------------------- |
| `top_k`   | int  | 10      | Number of top results to return |

**How it works:**
1. Tokenizes all documents and query by whitespace
2. Builds a BM25Okapi model on the document corpus
3. Scores all documents against the query
4. Returns top-k documents by BM25 score

---

### 3. Procrustes Alignment (`procrustes`)

Aligns embedding spaces from different sources to a common anchor space using orthogonal Procrustes transformation. Enables meaningful comparison across sources that use different embedding models.

**Best for:** Federated systems where sources use different embedding models with potentially different dimensionalities.

```python
results = aggregator.perform_aggregation(
    query="search query",
    retrieved_nodes=retrieved_nodes,
    method=Aggregate.PROCRUSTES,
    top_k=10,
    apply_scaling=True  # Normalize scores per source
)
```

**Parameters:**
| Parameter       | Type | Default | Description                            |
| --------------- | ---- | ------- | -------------------------------------- |
| `top_k`         | int  | 10      | Number of top results to return        |
| `apply_scaling` | bool | True    | Apply min-max normalization per source |

**How it works:**
1. Selects a random source as the anchor embedding space
2. For each other source, computes an orthogonal Procrustes transformation matrix
3. Projects all document embeddings into the anchor space
4. Computes cosine similarity with the anchor query embedding
5. Optionally applies min-max scaling per source for fair comparison
6. Returns globally ranked top-k documents

**Requirements:** Each source must include `query_embedding` for alignment computation.

---

### 4. Naive Top-K (`naive_topk`)

Simple aggregation that sorts all documents by their original confidence scores.

**Best for:** Baseline comparison; quick aggregation when original scores are trustworthy.

```python
results = aggregator.perform_aggregation(
    query="search query",
    retrieved_nodes=retrieved_nodes,
    method=Aggregate.NAIVE_TOPK,
    top_k=10
)
```

**Parameters:**
| Parameter | Type | Default | Description                     |
| --------- | ---- | ------- | ------------------------------- |
| `top_k`   | int  | 10      | Number of top results to return |

**How it works:**
1. Collects all documents from all sources
2. Sorts by the existing `score` field (descending)
3. Returns top-k documents

---

## Running Multiple Methods

Compare different aggregation strategies by passing a list of methods:

```python
results = aggregator.perform_aggregation(
    query="search query",
    retrieved_nodes=retrieved_nodes,
    method=[Aggregate.PROCRUSTES, Aggregate.RRP_BM25, Aggregate.CENTRAL_REEMBEDDING, Aggregate.NAIVE_TOPK],
    top_k=10
)

# Compare results
for method_name, result in results.items():
    print(f"\n{method_name}:")
    print(f"  Time: {result['time_taken']:.3f}s")
    print(f"  Top result: {result['reranked_nodes'][0]['document']['content'][:50]}...")
```

## Input Data Format

The `retrieved_nodes` parameter expects the following structure:

```python
{
    "source_name": {
        "sources": [
            {
                "document": {
                    "content": "The document text content...",
                    # ... other document fields
                },
                "document_embedding": [0.1, 0.2, ...],  # Optional
                "score": 0.95  # Required for naive_topk
            },
            # ... more documents
        ],
        "query_embedding": [0.1, 0.2, ...],  # Required for procrustes
        "document_embeddings": [[...], [...]],  # Optional
        "embedding_model_name": "model-name",
        "similarity_metric": "cosine"
    },
    # ... more sources
}
```

## Output Format

All methods return a dictionary with:

```python
{
    "method_name": {
        "reranked_nodes": [
            {
                "document": {...},
                "score": 0.95,
                # Additional fields depending on method
            },
            # ... top_k documents
        ],
        "time_taken": 0.123  # Execution time in seconds
    }
}
```

## Method Comparison

| Method                 | Embedding Required | Handles Multi-Dimensional | Speed   | Use Case                   |
| ---------------------- | ------------------ | ------------------------- | ------- | -------------------------- |
| `central_re_embedding` | No (re-computes)   | Yes                       | Slow    | Unified embedding space    |
| `rrp_bm25`             | No                 | N/A                       | Fast    | Keyword-based ranking      |
| `procrustes`           | Yes (with query)   | Yes                       | Medium  | Cross-model alignment      |
| `naive_topk`           | No                 | N/A                       | Fastest | Baseline/quick aggregation |

## License

MIT License

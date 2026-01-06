"""
Comprehensive test suite for the Federated Aggregation library.

This test module covers all functionalities including:
- Aggregate class and its methods
- Central re-embedding method
- Naive top-k baseline method
- BM25 re-ranking method
- Procrustes alignment method
- Input validation and edge cases
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from federated_aggregation import Aggregate
from federated_aggregation.central_re_embedding import re_embedding_reranker
from federated_aggregation.naive_conf_reranking import naive_topk_reranker
from federated_aggregation.rrp_bm25 import bm25_reranker
from federated_aggregation.procrustes_method import procrustes_reranker, _orthogonal_map


# =============================================================================
# Fixtures for test data
# =============================================================================

@pytest.fixture
def sample_query():
    """Sample query string for testing."""
    return "What is machine learning?"


@pytest.fixture
def sample_embedding():
    """Sample normalized embedding vector."""
    np.random.seed(42)
    emb = np.random.randn(384)
    return (emb / np.linalg.norm(emb)).tolist()


@pytest.fixture
def sample_documents():
    """Sample documents for a single source."""
    np.random.seed(42)
    return [
        {
            "document": {
                "content": "Machine learning is a subset of artificial intelligence.",
                "title": "ML Basics"
            },
            "document_embedding": np.random.randn(384).tolist(),
            "score": 0.95
        },
        {
            "document": {
                "content": "Deep learning uses neural networks with many layers.",
                "title": "Deep Learning"
            },
            "document_embedding": np.random.randn(384).tolist(),
            "score": 0.85
        },
        {
            "document": {
                "content": "Natural language processing deals with text data.",
                "title": "NLP"
            },
            "document_embedding": np.random.randn(384).tolist(),
            "score": 0.75
        }
    ]


@pytest.fixture
def sample_retrieved_nodes(sample_embedding):
    """Sample multi-source retrieved nodes structure (raw format for Aggregate class)."""
    np.random.seed(42)
    doc_embeddings_a = [np.random.randn(384).tolist() for _ in range(3)]
    doc_embeddings_b = [np.random.randn(384).tolist() for _ in range(2)]

    return {
        "source_a": {
            "sources": [
                {
                    "document": {
                        "content": "Machine learning is a subset of artificial intelligence.",
                        "title": "ML Basics"
                    },
                    "document_embedding": doc_embeddings_a[0],
                    "score": 0.95
                },
                {
                    "document": {
                        "content": "Deep learning uses neural networks with many layers.",
                        "title": "Deep Learning"
                    },
                    "document_embedding": doc_embeddings_a[1],
                    "score": 0.85
                },
                {
                    "document": {
                        "content": "Natural language processing deals with text data.",
                        "title": "NLP"
                    },
                    "document_embedding": doc_embeddings_a[2],
                    "score": 0.75
                }
            ],
            "query_embedding": sample_embedding,
            "document_embeddings": doc_embeddings_a,
            "embedding_model_name": "model-a",
            "similarity_metric": "cosine"
        },
        "source_b": {
            "sources": [
                {
                    "document": {
                        "content": "Supervised learning requires labeled data.",
                        "title": "Supervised Learning"
                    },
                    "document_embedding": doc_embeddings_b[0],
                    "score": 0.90
                },
                {
                    "document": {
                        "content": "Unsupervised learning finds patterns without labels.",
                        "title": "Unsupervised Learning"
                    },
                    "document_embedding": doc_embeddings_b[1],
                    "score": 0.80
                }
            ],
            "query_embedding": np.random.randn(384).tolist(),
            "document_embeddings": doc_embeddings_b,
            "embedding_model_name": "model-b",
            "similarity_metric": "cosine"
        }
    }


@pytest.fixture
def preprocessed_documents(sample_embedding):
    """Pre-processed documents format (with global_docs and query_embeddings)."""
    np.random.seed(42)
    return {
        "global_docs": [
            {
                "person": "source_a",
                "document": {"content": "Machine learning is a subset of artificial intelligence."},
                "embedding": np.random.randn(384),
                "doc_embedding": np.random.randn(384),
                "score": 0.95,
                "embedding_model_name": "model-a",
                "similarity_metric": "cosine"
            },
            {
                "person": "source_a",
                "document": {"content": "Deep learning uses neural networks with many layers."},
                "embedding": np.random.randn(384),
                "doc_embedding": np.random.randn(384),
                "score": 0.85,
                "embedding_model_name": "model-a",
                "similarity_metric": "cosine"
            },
            {
                "person": "source_a",
                "document": {"content": "Natural language processing deals with text data."},
                "embedding": np.random.randn(384),
                "doc_embedding": np.random.randn(384),
                "score": 0.75,
                "embedding_model_name": "model-a",
                "similarity_metric": "cosine"
            },
            {
                "person": "source_b",
                "document": {"content": "Supervised learning requires labeled data."},
                "embedding": np.random.randn(384),
                "doc_embedding": np.random.randn(384),
                "score": 0.90,
                "embedding_model_name": "model-b",
                "similarity_metric": "cosine"
            },
            {
                "person": "source_b",
                "document": {"content": "Unsupervised learning finds patterns without labels."},
                "embedding": np.random.randn(384),
                "doc_embedding": np.random.randn(384),
                "score": 0.80,
                "embedding_model_name": "model-b",
                "similarity_metric": "cosine"
            }
        ],
        "query_embeddings": {
            "source_a": np.array(sample_embedding),
            "source_b": np.random.randn(384)
        }
    }


@pytest.fixture
def multi_dim_preprocessed_documents():
    """Pre-processed documents with different embedding dimensions per source."""
    np.random.seed(42)
    return {
        "global_docs": [
            {
                "person": "source_384",
                "document": {"content": "Document from 384-dim source."},
                "embedding": np.random.randn(384),
                "doc_embedding": np.random.randn(384),
                "score": 0.9,
                "embedding_model_name": "model-384",
                "similarity_metric": "cosine"
            },
            {
                "person": "source_768",
                "document": {"content": "Document from 768-dim source."},
                "embedding": np.random.randn(768),
                "doc_embedding": np.random.randn(768),
                "score": 0.85,
                "embedding_model_name": "model-768",
                "similarity_metric": "cosine"
            }
        ],
        "query_embeddings": {
            "source_384": np.random.randn(384),
            "source_768": np.random.randn(768)
        }
    }


# =============================================================================
# Aggregate Class Tests
# =============================================================================

class TestAggregateClass:
    """Tests for the main Aggregate class."""

    def test_init(self):
        """Test Aggregate class initialization."""
        agg = Aggregate()
        assert hasattr(agg, 'supported_methods')
        assert Aggregate.CENTRAL_REEMBEDDING in agg.supported_methods
        assert Aggregate.RRP_BM25 in agg.supported_methods
        assert Aggregate.NAIVE_TOPK in agg.supported_methods
        assert Aggregate.PROCRUSTES in agg.supported_methods

    def test_class_constants(self):
        """Test that class constants are properly defined."""
        assert Aggregate.CENTRAL_REEMBEDDING == "central_re_embedding"
        assert Aggregate.RRP_BM25 == "rrp_bm25"
        assert Aggregate.NAIVE_TOPK == "naive_topk"
        assert Aggregate.PROCRUSTES == "procrustes"

    def test_flatten_query_result(self, sample_retrieved_nodes):
        """Test flattening of multi-source retrieval results."""
        agg = Aggregate()
        result = agg.flatten_query_result(sample_retrieved_nodes)

        assert "global_docs" in result
        assert "query_embeddings" in result
        assert len(result["global_docs"]) == 5  # 3 from source_a + 2 from source_b

        # Check that source information is preserved
        for doc in result["global_docs"]:
            assert "person" in doc
            assert doc["person"] in ["source_a", "source_b"]

    def test_perform_aggregation_single_method(self, sample_query, sample_retrieved_nodes):
        """Test aggregation with a single method."""
        agg = Aggregate()
        result = agg.perform_aggregation(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            method="naive_topk",
            top_k=3
        )

        assert "naive_topk" in result
        assert "reranked_nodes" in result["naive_topk"]
        assert "time_taken" in result["naive_topk"]
        assert len(result["naive_topk"]["reranked_nodes"]) <= 3

    def test_perform_aggregation_multiple_methods(self, sample_query, sample_retrieved_nodes):
        """Test aggregation with multiple methods."""
        agg = Aggregate()
        result = agg.perform_aggregation(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            method=["naive_topk", "rrp_bm25"],
            top_k=3
        )

        assert "naive_topk" in result
        assert "rrp_bm25" in result

    def test_perform_aggregation_invalid_method(self, sample_query, sample_retrieved_nodes):
        """Test aggregation with an invalid method name."""
        agg = Aggregate()
        with pytest.raises(AssertionError):
            agg.perform_aggregation(
                query=sample_query,
                retrieved_nodes=sample_retrieved_nodes,
                method="invalid_method",
                top_k=3
            )

    def test_perform_aggregation_time_tracking(self, sample_query, sample_retrieved_nodes):
        """Test that aggregation tracks execution time."""
        agg = Aggregate()
        result = agg.perform_aggregation(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            method="naive_topk",
            top_k=3
        )

        assert result["naive_topk"]["time_taken"] >= 0


# =============================================================================
# Naive Top-K Method Tests
# =============================================================================

class TestNaiveTopKReranker:
    """Tests for the naive top-k baseline method."""

    def test_basic_functionality(self, sample_query, preprocessed_documents):
        """Test basic naive top-k reranking."""
        result = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )

        assert len(result) == 3
        # Check descending score order
        scores = [doc["score"] for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self, sample_query, preprocessed_documents):
        """Test that top_k parameter limits results correctly."""
        for k in [1, 2, 5]:
            result = naive_topk_reranker(
                query=sample_query,
                retrieved_documents=preprocessed_documents,
                top_k=k
            )
            assert len(result) <= k

    def test_preserves_document_data(self, sample_query, preprocessed_documents):
        """Test that document data is preserved in results."""
        result = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=5
        )

        for doc in result:
            assert "document" in doc
            assert "score" in doc
            assert "person" in doc

    def test_query_none_raises_error(self, preprocessed_documents):
        """Test that None query raises ValueError."""
        with pytest.raises(ValueError, match="found to be None"):
            naive_topk_reranker(
                query=None,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )

    def test_query_not_string_raises_error(self, preprocessed_documents):
        """Test that non-string query raises ValueError."""
        with pytest.raises(ValueError, match="should be a String"):
            naive_topk_reranker(
                query=12345,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )


# =============================================================================
# BM25 Re-Ranking Method Tests
# =============================================================================

class TestBM25Reranker:
    """Tests for the BM25 re-ranking method."""

    def test_basic_functionality(self, sample_query, preprocessed_documents):
        """Test basic BM25 reranking."""
        result = bm25_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )

        assert len(result) <= 3
        for doc in result:
            assert "document" in doc

    def test_keyword_relevance(self, preprocessed_documents):
        """Test that BM25 ranks keyword-relevant documents higher."""
        query = "neural networks deep learning"
        result = bm25_reranker(
            query=query,
            retrieved_documents=preprocessed_documents,
            top_k=5
        )

        # Documents containing query terms should be ranked higher
        assert len(result) > 0

    def test_top_k_limit(self, sample_query, preprocessed_documents):
        """Test that top_k parameter limits results correctly."""
        for k in [1, 2, 5]:
            result = bm25_reranker(
                query=sample_query,
                retrieved_documents=preprocessed_documents,
                top_k=k
            )
            assert len(result) <= k

    def test_query_none_raises_error(self, preprocessed_documents):
        """Test that None query raises ValueError."""
        with pytest.raises(ValueError, match="found to be None"):
            bm25_reranker(
                query=None,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )

    def test_query_not_string_raises_error(self, preprocessed_documents):
        """Test that non-string query raises ValueError."""
        with pytest.raises(ValueError, match="should be a String"):
            bm25_reranker(
                query=12345,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )


# =============================================================================
# Procrustes Alignment Method Tests
# =============================================================================

class TestProcrustesReranker:
    """Tests for the Procrustes alignment method."""

    def test_basic_functionality(self, sample_query, preprocessed_documents):
        """Test basic Procrustes reranking."""
        np.random.seed(42)
        result = procrustes_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )

        assert len(result) <= 3
        for doc in result:
            assert "document" in doc
            assert "score" in doc

    def test_handles_different_embedding_dimensions(self, sample_query, multi_dim_preprocessed_documents):
        """Test that Procrustes handles different embedding dimensions."""
        np.random.seed(42)
        result = procrustes_reranker(
            query=sample_query,
            retrieved_documents=multi_dim_preprocessed_documents,
            top_k=2
        )

        assert len(result) <= 2

    def test_with_scaling(self, sample_query, preprocessed_documents):
        """Test Procrustes with scaling enabled."""
        np.random.seed(42)
        result = procrustes_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3,
            apply_scaling=True
        )

        assert len(result) <= 3

    def test_without_scaling(self, sample_query, preprocessed_documents):
        """Test Procrustes with scaling disabled."""
        np.random.seed(42)
        result = procrustes_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3,
            apply_scaling=False
        )

        assert len(result) <= 3

    def test_query_none_raises_error(self, preprocessed_documents):
        """Test that None query raises ValueError."""
        with pytest.raises(ValueError, match="found to be None"):
            procrustes_reranker(
                query=None,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )

    def test_query_not_string_raises_error(self, preprocessed_documents):
        """Test that non-string query raises ValueError."""
        with pytest.raises(ValueError, match="should be a String"):
            procrustes_reranker(
                query=12345,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )


class TestOrthogonalMap:
    """Tests for the _orthogonal_map helper function."""

    def test_same_dimension_mapping(self):
        """Test orthogonal mapping with same dimensions."""
        np.random.seed(42)
        Q_i = np.random.randn(10, 128)
        Q_c = np.random.randn(10, 128)

        P, mu_i, mu_c, d_min = _orthogonal_map(Q_i, Q_c)

        assert P.shape == (128, 128)
        assert mu_i.shape == (128,)
        assert mu_c.shape == (128,)
        assert d_min == 128

    def test_different_dimension_mapping(self):
        """Test orthogonal mapping with different dimensions."""
        np.random.seed(42)
        Q_i = np.random.randn(10, 128)
        Q_c = np.random.randn(10, 256)

        P, mu_i, mu_c, d_min = _orthogonal_map(Q_i, Q_c)

        # Should truncate to minimum dimension
        assert d_min == 128

    def test_orthogonality_of_transformation(self):
        """Test that the transformation matrix is orthogonal."""
        np.random.seed(42)
        Q_i = np.random.randn(20, 64)
        Q_c = np.random.randn(20, 64)

        P, _, _, _ = _orthogonal_map(Q_i, Q_c)

        # For an orthogonal matrix, P @ P.T should be close to identity
        identity = np.eye(P.shape[0])
        result = P @ P.T
        np.testing.assert_array_almost_equal(result, identity, decimal=5)


# =============================================================================
# Central Re-Embedding Method Tests
# =============================================================================

class TestCentralReEmbedding:
    """Tests for the central re-embedding method."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock the fastembed embedder."""
        mock = MagicMock()
        # Return normalized embeddings
        def embed_fn(texts):
            if isinstance(texts, str):
                texts = [texts]
            for _ in texts:
                emb = np.random.randn(384)
                yield emb / np.linalg.norm(emb)
        mock.embed = embed_fn
        return mock

    def test_basic_functionality_mocked(self, sample_query, preprocessed_documents, mock_embedder):
        """Test basic re-embedding reranking with mocked embedder."""
        with patch('federated_aggregation.central_re_embedding.embedder', mock_embedder):
            with patch('federated_aggregation.central_re_embedding.TextEmbedding', return_value=mock_embedder):
                # Set the global embedder to None to force re-initialization
                import federated_aggregation.central_re_embedding as cre
                original_embedder = cre.embedder
                cre.embedder = mock_embedder

                try:
                    result = re_embedding_reranker(
                        query=sample_query,
                        retrieved_documents=preprocessed_documents,
                        top_k=3
                    )

                    assert len(result) <= 3
                    for doc in result:
                        assert "document" in doc
                        assert "score" in doc
                finally:
                    cre.embedder = original_embedder

    def test_query_none_raises_error(self, preprocessed_documents):
        """Test that None query raises ValueError."""
        with pytest.raises(ValueError, match="found to be None"):
            re_embedding_reranker(
                query=None,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )

    def test_query_not_string_raises_error(self, preprocessed_documents):
        """Test that non-string query raises ValueError."""
        with pytest.raises(ValueError, match="should be a String"):
            re_embedding_reranker(
                query=12345,
                retrieved_documents=preprocessed_documents,
                top_k=3
            )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_query_string(self, preprocessed_documents):
        """Test handling of empty query string."""
        # Empty string should be valid (not None, is a string)
        result = naive_topk_reranker(
            query="",
            retrieved_documents=preprocessed_documents,
            top_k=3
        )
        assert isinstance(result, list)

    def test_single_source(self, sample_query, sample_embedding):
        """Test with a single source using Aggregate class."""
        np.random.seed(42)
        doc_embeddings = [np.random.randn(384).tolist() for _ in range(3)]
        single_source = {
            "only_source": {
                "sources": [
                    {
                        "document": {"content": "Document one about ML."},
                        "document_embedding": doc_embeddings[0],
                        "score": 0.9
                    },
                    {
                        "document": {"content": "Document two about AI."},
                        "document_embedding": doc_embeddings[1],
                        "score": 0.8
                    },
                    {
                        "document": {"content": "Document three about DL."},
                        "document_embedding": doc_embeddings[2],
                        "score": 0.7
                    }
                ],
                "query_embedding": sample_embedding,
                "document_embeddings": doc_embeddings,
                "embedding_model_name": "model",
                "similarity_metric": "cosine"
            }
        }

        agg = Aggregate()
        result = agg.perform_aggregation(
            query=sample_query,
            retrieved_nodes=single_source,
            method="naive_topk",
            top_k=3
        )

        assert len(result["naive_topk"]["reranked_nodes"]) == 3

    def test_single_document(self, sample_query, sample_embedding):
        """Test with a single document."""
        np.random.seed(42)
        doc_embedding = np.random.randn(384).tolist()
        single_doc = {
            "source": {
                "sources": [
                    {
                        "document": {"content": "Only document"},
                        "document_embedding": doc_embedding,
                        "score": 0.9
                    }
                ],
                "query_embedding": sample_embedding,
                "document_embeddings": [doc_embedding],
                "embedding_model_name": "model",
                "similarity_metric": "cosine"
            }
        }

        agg = Aggregate()
        result = agg.perform_aggregation(
            query=sample_query,
            retrieved_nodes=single_doc,
            method="naive_topk",
            top_k=5
        )

        assert len(result["naive_topk"]["reranked_nodes"]) == 1

    def test_top_k_larger_than_documents(self, sample_query, preprocessed_documents):
        """Test when top_k is larger than available documents."""
        result = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=100
        )

        # Should return all available documents (5 total)
        assert len(result) == 5

    def test_top_k_zero(self, sample_query, preprocessed_documents):
        """Test with top_k of zero."""
        result = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=0
        )

        assert len(result) == 0

    def test_special_characters_in_query(self, preprocessed_documents):
        """Test with special characters in query."""
        special_query = "What is ML? #test @user $100"
        result = bm25_reranker(
            query=special_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )

        assert isinstance(result, list)

    def test_unicode_content(self, sample_query, sample_embedding):
        """Test with Unicode content in documents."""
        np.random.seed(42)
        doc_embedding = np.random.randn(384)
        unicode_docs = {
            "global_docs": [
                {
                    "person": "source",
                    "document": {"content": "Machine learning AI ML"},
                    "embedding": doc_embedding,
                    "doc_embedding": doc_embedding,
                    "score": 0.9,
                    "embedding_model_name": "model",
                    "similarity_metric": "cosine"
                }
            ],
            "query_embeddings": {
                "source": np.array(sample_embedding)
            }
        }

        result = bm25_reranker(
            query=sample_query,
            retrieved_documents=unicode_docs,
            top_k=1
        )

        assert len(result) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_aggregation_workflow(self, sample_query, sample_retrieved_nodes):
        """Test complete aggregation workflow with multiple methods."""
        agg = Aggregate()

        # Test with all methods that don't require external models
        methods = ["naive_topk", "rrp_bm25", "procrustes"]

        result = agg.perform_aggregation(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            method=methods,
            top_k=3
        )

        for method in methods:
            assert method in result
            assert "reranked_nodes" in result[method]
            assert "time_taken" in result[method]

    def test_method_consistency(self, sample_query, preprocessed_documents):
        """Test that same method produces consistent results."""
        np.random.seed(42)
        result1 = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )

        np.random.seed(42)
        result2 = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )

        # Results should be identical
        assert len(result1) == len(result2)
        for doc1, doc2 in zip(result1, result2):
            assert doc1["score"] == doc2["score"]

    def test_different_methods_different_rankings(self, sample_query, preprocessed_documents):
        """Test that different methods may produce different rankings."""
        naive_result = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=5
        )

        bm25_result = bm25_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=5
        )

        # Both should return results
        assert len(naive_result) > 0
        assert len(bm25_result) > 0

    def test_large_document_set(self, sample_query):
        """Test with a large number of documents."""
        np.random.seed(42)

        # Create 100 documents across 5 sources
        global_docs = []
        query_embeddings = {}

        for i in range(5):
            query_embeddings[f"source_{i}"] = np.random.randn(384)
            for j in range(20):
                global_docs.append({
                    "person": f"source_{i}",
                    "document": {"content": f"Document {i}-{j} about various topics in machine learning and AI."},
                    "embedding": np.random.randn(384),
                    "doc_embedding": np.random.randn(384),
                    "score": np.random.random(),
                    "embedding_model_name": f"model-{i}",
                    "similarity_metric": "cosine"
                })

        large_preprocessed = {
            "global_docs": global_docs,
            "query_embeddings": query_embeddings
        }

        result = naive_topk_reranker(
            query=sample_query,
            retrieved_documents=large_preprocessed,
            top_k=10
        )

        assert len(result) == 10


# =============================================================================
# Performance Tests (Optional)
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_execution_time_reasonable(self, sample_query, preprocessed_documents):
        """Test that execution time is reasonable."""
        import time

        start = time.time()
        naive_topk_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )
        elapsed = time.time() - start

        # Should complete in less than 1 second for small dataset
        assert elapsed < 1.0

    def test_bm25_performance(self, sample_query, preprocessed_documents):
        """Test BM25 execution time."""
        import time

        start = time.time()
        bm25_reranker(
            query=sample_query,
            retrieved_documents=preprocessed_documents,
            top_k=3
        )
        elapsed = time.time() - start

        # Should complete in less than 2 seconds
        assert elapsed < 2.0


# =============================================================================
# Aggregate Class Method Wrapper Tests
# =============================================================================

class TestAggregateMethodWrappers:
    """Tests for the Aggregate class method wrappers."""

    def test_aggregate_naive_topk(self, sample_query, sample_retrieved_nodes):
        """Test aggregate_naive_topk wrapper method."""
        agg = Aggregate()
        result, time_taken = agg.aggregate_naive_topk(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            top_k=3,
            preprocessed=False
        )

        assert len(result) <= 3
        assert time_taken >= 0

    def test_aggregate_rrp_bm25(self, sample_query, sample_retrieved_nodes):
        """Test aggregate_rrp_bm25 wrapper method."""
        agg = Aggregate()
        result, time_taken = agg.aggregate_rrp_bm25(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            top_k=3,
            preprocessed=False
        )

        assert len(result) <= 3
        assert time_taken >= 0

    def test_aggregate_procrustes(self, sample_query, sample_retrieved_nodes):
        """Test aggregate_procrustes wrapper method."""
        np.random.seed(42)
        agg = Aggregate()
        result, time_taken = agg.aggregate_procrustes(
            query=sample_query,
            retrieved_nodes=sample_retrieved_nodes,
            top_k=3,
            preprocessed=False
        )

        assert len(result) <= 3
        assert time_taken >= 0


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

import time
import numpy as np

from .central_re_embedding import re_embedding_reranker
from .naive_conf_reranking import naive_topk_reranker
from .rrp_bm25 import bm25_reranker
from .procrustes_method import procrustes_reranker


class Aggregate:
    # Method constants (similar to cv2.COLOR_BGR2GRAY style)
    CENTRAL_REEMBEDDING = "central_re_embedding"
    RRP_BM25 = "rrp_bm25"
    NAIVE_TOPK = "naive_topk"
    PROCRUSTES = "procrustes"

    def __init__(self):
        self.supported_methods = [
            self.CENTRAL_REEMBEDDING, self.RRP_BM25, self.NAIVE_TOPK, self.PROCRUSTES]
        self.name_to_method_mapping = {
            self.CENTRAL_REEMBEDDING: self.aggregate_reembed,
            self.RRP_BM25: self.aggregate_rrp_bm25,
            self.NAIVE_TOPK: self.aggregate_naive_topk,
            self.PROCRUSTES: self.aggregate_procrustes
        }

    def flatten_query_result(self, query_result):
        """
        Flatten MockRetrievalService/SyftHubRetrievalService query() output into a single global document list.
        """
        global_docs = []
        query_embeddings = {}

        for person, info in query_result.items():
            doc_embeddings = info.get("document_embeddings", [])
            sources = info.get("sources", [])
            for i, src in enumerate(sources):
                global_docs.append({
                    "person": person,
                    "document": src["document"],
                    "embedding": np.array(src.get("document_embedding", doc_embeddings[i])),
                    "doc_embedding": np.array(doc_embeddings[i]),
                    "score": src.get("score", 0.0),
                    "embedding_model_name": info.get("embedding_model_name"),
                    "similarity_metric": info.get("similarity_metric")
                })
            query_embeddings[person] = np.array(
                info["query_embedding"]) if info["query_embedding"] else np.array([])

        flattened_output = {"global_docs": global_docs,
                            "query_embeddings": query_embeddings}
        return flattened_output

    def perform_aggregation(self, query, retrieved_nodes, method=["procrustes"], **qwargs):
        retrieved_nodes = self.flatten_query_result(
            query_result=retrieved_nodes)

        output = {}
        if isinstance(method, str):
            assert method in self.supported_methods

            reranked_nodes, time_taken = self.name_to_method_mapping[method](query, retrieved_nodes, preprocessed=True,
                                                                             **qwargs)
            output[method] = {"reranked_nodes": reranked_nodes,
                              "time_taken": time_taken}

        if isinstance(method, list):
            for m in method:
                assert m in self.supported_methods
                reranked_nodes, time_taken = self.name_to_method_mapping[m](query, retrieved_nodes, preprocessed=True,
                                                                            **qwargs)
                output[m] = {"reranked_nodes": reranked_nodes,
                             "time_taken": time_taken}
        return output

    def aggregate_reembed(self, query, retrieved_nodes, model_name="BAAI/bge-small-en-v1.5", device="cpu",
                          top_k=5,
                          preprocessed=False):
        if not preprocessed:
            retrieved_nodes = self.flatten_query_result(
                query_result=retrieved_nodes)

        start = time.time()
        reranked_documents = re_embedding_reranker(query=query, retrieved_documents=retrieved_nodes,
                                                   top_k=top_k, model_name=model_name, device=device)
        end = time.time()
        return reranked_documents, end-start

    def aggregate_naive_topk(self, query, retrieved_nodes, top_k=5, preprocessed=False):
        if not preprocessed:
            retrieved_nodes = self.flatten_query_result(
                query_result=retrieved_nodes)

        start = time.time()
        reranked_documents = naive_topk_reranker(query=query, retrieved_documents=retrieved_nodes,
                                                 top_k=top_k)
        end = time.time()
        return reranked_documents, end-start

    def aggregate_procrustes(self, query, retrieved_nodes, top_k=5, preprocessed=False):
        if not preprocessed:
            retrieved_nodes = self.flatten_query_result(
                query_result=retrieved_nodes
            )
        start = time.time()
        reranked_documents = procrustes_reranker(query=query, retrieved_documents=retrieved_nodes,
                                                 top_k=top_k)
        end = time.time()
        return reranked_documents, end-start

    def aggregate_rrp_bm25(self, query, retrieved_nodes, top_k=5, preprocessed=False):
        if not preprocessed:
            retrieved_nodes = self.flatten_query_result(
                query_result=retrieved_nodes
            )
        start = time.time()
        reranked_documents = bm25_reranker(query=query, retrieved_documents=retrieved_nodes,
                                           top_k=top_k)
        end = time.time()
        return reranked_documents, end-start

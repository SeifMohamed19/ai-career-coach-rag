import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import RRF_K, DEFAULT_TOP_K
from src.rag.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("HybridRetriever")

class HybridRetriever:
    """
    Hybrid Search Retriever combining:
    1. Dense Vector Retrieval (Semantic Cosine Similarity via ChromaDB)
    2. Sparse Lexical Retrieval (BM25 Okapi for exact keyword matches)
    3. Reciprocal Rank Fusion (RRF) for balanced multi-modal rank merging.
    """
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: List[Dict[str, Any]] = []
        self._build_bm25_index()

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace/alphanumeric tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def _build_bm25_index(self) -> None:
        """Fetches all documents from ChromaDB and compiles the in-memory BM25 index."""
        self.corpus_docs = self.vector_store.get_all_documents()
        if not self.corpus_docs:
            logger.warning("No documents in vector store to build BM25 index.")
            return

        tokenized_corpus = [self.tokenize(doc["content"]) for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 index with {len(self.corpus_docs)} documents.")

    def reload(self) -> None:
        """Reloads the BM25 index after new documents are ingested."""
        self._build_bm25_index()

    def search_sparse(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executes BM25 keyword retrieval."""
        if not self.bm25 or not self.corpus_docs:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.corpus_docs[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": round(float(scores[idx]), 4)
                })
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Search combining Dense and Sparse scores via
        Reciprocal Rank Fusion (RRF).

        RRF Score Formula:
            RRF(d) = sum( weight_i / (k + rank_i(d)) )
        """
        # 1. Dense retrieval
        dense_results = self.vector_store.search(query=query, n_results=top_k * 2, where_filter=where_filter)

        # 2. Sparse retrieval
        sparse_results = self.search_sparse(query=query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Rank dense results
        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            doc_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (dense_weight / (RRF_K + rank))

        # Rank sparse results
        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (sparse_weight / (RRF_K + rank))

        # Sort combined results by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        final_results = []
        for doc_id in sorted_ids:
            doc_info = dict(doc_map[doc_id])
            doc_info["rrf_score"] = round(rrf_scores[doc_id], 5)
            final_results.append(doc_info)

        return final_results

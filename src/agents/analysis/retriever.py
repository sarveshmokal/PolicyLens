"""
Retriever Agent for PolicyLens.

Core analysis agent that implements hybrid retrieval: combining
BM25 sparse keyword search with SBERT dense semantic search using
an alpha-weighted fusion formula:

    S_final = alpha * S_dense + (1 - alpha) * S_sparse

Alpha controls the balance: 0.0 = pure BM25, 1.0 = pure dense,
0.5 = equal weight. The optimal alpha is discovered through
experiment tracking and can be auto-tuned by the MemoryAgent.

This agent is the foundation of retrieval quality. Every metric
improvement in the experiment log traces back to changes here.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi

from src.core.base_agent import BaseAgent
from src.core.config import settings, PROJECT_ROOT


class RetrieverAgent(BaseAgent):
    """Hybrid retriever combining BM25 sparse and SBERT dense search.

    Maintains a BM25 index built from chunk texts alongside the
    dense vectors in ChromaDB. At query time, both systems score
    the corpus independently, and results are fused using alpha
    weighting. This captures both exact keyword matches (BM25)
    and semantic similarity (dense).
    """

    def __init__(self, name: str = "retriever", description: str = ""):
        super().__init__(name=name, description=description)
        self.alpha = settings.retrieval.get("alpha", 0.5)
        self.top_k = settings.retrieval.get("top_k", 10)
        self.similarity_threshold = settings.retrieval.get("similarity_threshold", 0.3)

        # BM25 index state
        self._bm25 = None
        self._bm25_corpus = []
        self._bm25_chunk_map = []

        # Embedding agent reference (lazy loaded)
        self._embedding_agent = None

    @property
    def embedding_agent(self):
        """Lazy-load the EmbeddingAgent to access ChromaDB."""
        if self._embedding_agent is None:
            from src.agents.ingestion.embedding_agent import EmbeddingAgent
            self._embedding_agent = EmbeddingAgent()
        return self._embedding_agent

    def build_bm25_index(self, chunks: list[dict]) -> None:
        """Build the BM25 sparse index from a list of text chunks.

        This must be called after ingestion and before querying.
        The index lives in memory — it's fast to rebuild from the
        chunk texts stored in ChromaDB.

        Args:
            chunks: List of chunk dicts with 'content' and metadata.
        """
        self.logger.info("Building BM25 index from %d chunks", len(chunks))

        self._bm25_corpus = []
        self._bm25_chunk_map = []

        for chunk in chunks:
            # Tokenize: lowercase and split on whitespace
            tokens = chunk["content"].lower().split()
            self._bm25_corpus.append(tokens)
            self._bm25_chunk_map.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("doc_id", chunk.get("metadata", {}).get("doc_id", "")),
                "content": chunk["content"],
                "page_number": chunk.get("page_number", chunk.get("metadata", {}).get("page_number", 0)),
                "source_file": chunk.get("source_file", chunk.get("metadata", {}).get("source_file", "")),
            })

        self._bm25 = BM25Okapi(self._bm25_corpus)
        self.logger.info("BM25 index built: %d documents indexed", len(self._bm25_corpus))

    def build_bm25_from_chromadb(self) -> None:
        """Build BM25 index from chunks already stored in ChromaDB.

        Convenience method that pulls all chunks from the vector store
        and builds the sparse index. Use this when the ingestion pipeline
        has already run and chunks are persisted.
        """
        collection = self.embedding_agent.collection
        total = collection.count()

        if total == 0:
            self.logger.warning("ChromaDB collection is empty. Run ingestion first.")
            return

        # Fetch all chunks from ChromaDB
        all_data = collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )

        chunks = []
        for i in range(len(all_data["ids"])):
            chunks.append({
                "chunk_id": all_data["ids"][i],
                "content": all_data["documents"][i],
                **all_data["metadatas"][i],
            })

        self.build_bm25_index(chunks)

    def _search_bm25(self, query: str, top_k: int) -> list[dict]:
        """Search using BM25 sparse keyword matching.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk info and BM25 scores,
            normalized to [0, 1] range.
        """
        if self._bm25 is None:
            self.logger.warning("BM25 index not built. Call build_bm25_index() first.")
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Normalize scores to [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized_scores = scores / max_score

        # Get top-k indices
        top_indices = normalized_scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(normalized_scores[idx])
            if score < self.similarity_threshold:
                continue
            chunk = self._bm25_chunk_map[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "content": chunk["content"],
                "score": round(score, 4),
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "method": "bm25",
            })

        return results

    def _search_dense(self, query: str, top_k: int) -> list[dict]:
        """Search using SBERT dense semantic similarity.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk info and cosine similarity scores.
        """
        raw_results = self.embedding_agent.search(query, top_k=top_k)

        results = []
        for r in raw_results:
            if r["score"] < self.similarity_threshold:
                continue
            results.append({
                "chunk_id": r["chunk_id"],
                "doc_id": r["metadata"].get("doc_id", ""),
                "content": r["content"],
                "score": round(r["score"], 4),
                "source_file": r["metadata"].get("source_file", ""),
                "page_number": r["metadata"].get("page_number", 0),
                "method": "dense",
            })

        return results

    def _hybrid_fusion(
        self, bm25_results: list[dict], dense_results: list[dict], alpha: float
    ) -> list[dict]:
        """Fuse BM25 and dense results using alpha-weighted scoring.

        S_final = alpha * S_dense + (1 - alpha) * S_sparse

        When a chunk appears in only one result set, the missing
        score defaults to 0. Results are sorted by fused score.

        Args:
            bm25_results: Results from BM25 search.
            dense_results: Results from dense search.
            alpha: Weight for dense scores (0 to 1).

        Returns:
            Fused results sorted by combined score.
        """
        # Build lookup by chunk_id
        score_map = {}

        for r in bm25_results:
            cid = r["chunk_id"]
            score_map[cid] = {
                **r,
                "bm25_score": r["score"],
                "dense_score": 0.0,
            }

        for r in dense_results:
            cid = r["chunk_id"]
            if cid in score_map:
                score_map[cid]["dense_score"] = r["score"]
            else:
                score_map[cid] = {
                    **r,
                    "bm25_score": 0.0,
                    "dense_score": r["score"],
                }

        # Calculate fused scores
        fused = []
        for cid, data in score_map.items():
            fused_score = alpha * data["dense_score"] + (1 - alpha) * data["bm25_score"]
            fused.append({
                "chunk_id": cid,
                "doc_id": data["doc_id"],
                "content": data["content"],
                "score": round(fused_score, 4),
                "bm25_score": data["bm25_score"],
                "dense_score": data["dense_score"],
                "source_file": data["source_file"],
                "page_number": data["page_number"],
                "method": "hybrid",
            })

        # Sort by fused score descending
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    def process(self, input_data: dict) -> dict:
        """Run hybrid retrieval for a query.

        Args:
            input_data: Must contain 'query' string.
                        Optional: 'top_k', 'alpha', 'method'.

        Returns:
            Dictionary with ranked passages, query info, and method used.
        """
        query = input_data["query"]
        top_k = input_data.get("top_k", self.top_k)
        alpha = input_data.get("alpha", self.alpha)
        method = input_data.get("method", "hybrid")

        self.logger.info(
            "Retrieving for query: '%s' (method=%s, alpha=%.2f, top_k=%d)",
            query[:80],
            method,
            alpha,
            top_k,
        )

        # Ensure BM25 index is built
        if self._bm25 is None:
            self.build_bm25_from_chromadb()

        if method == "bm25":
            passages = self._search_bm25(query, top_k)
        elif method == "dense":
            passages = self._search_dense(query, top_k)
        else:
            # Hybrid: fetch more from each, then fuse and trim
            bm25_results = self._search_bm25(query, top_k * 2)
            dense_results = self._search_dense(query, top_k * 2)
            passages = self._hybrid_fusion(bm25_results, dense_results, alpha)
            passages = passages[:top_k]

        self.logger.info(
            "Retrieved %d passages (method=%s)",
            len(passages),
            method,
        )

        return {
            "query": query,
            "passages": passages,
            "retrieval_method": method,
            "alpha": alpha,
            "top_k": top_k,
            "total_results": len(passages),
        }
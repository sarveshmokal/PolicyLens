"""
Reranker Agent for PolicyLens.

Takes retrieval results and re-scores them using a cross-encoder
model (ms-marco-MiniLM-L-6-v2). Cross-encoders are more accurate
than bi-encoders for ranking because they process the query and
passage together, capturing fine-grained interactions.

The tradeoff: cross-encoders are slower (can't pre-compute embeddings),
so they're used to re-rank a small set (top-k from retrieval),
not to search the entire corpus.
"""

from sentence_transformers import CrossEncoder

from src.core.base_agent import BaseAgent
from src.core.config import settings


class RerankerAgent(BaseAgent):
    """Re-scores retrieved passages using a cross-encoder model.

    Takes the top-k results from the RetrieverAgent and re-ranks
    them for higher precision. The cross-encoder sees query and
    passage together, unlike the bi-encoder which embeds them
    separately.
    """

    def __init__(self, name: str = "reranker", description: str = ""):
        super().__init__(name=name, description=description)
        self._model = None
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.rerank_top_k = settings.retrieval.get("rerank_top_k", 5)

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            self.logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            self.logger.info("Cross-encoder loaded")
        return self._model

    def process(self, input_data: dict) -> dict:
        """Re-rank retrieved passages using cross-encoder scoring.

        Args:
            input_data: Must contain 'query' (str) and 'passages' (list).
                        Optional: 'rerank_top_k' (int).

        Returns:
            Dictionary with re-ranked passages and score comparisons.
        """
        query = input_data["query"]
        passages = input_data.get("passages", [])
        rerank_top_k = input_data.get("rerank_top_k", self.rerank_top_k)

        if not passages:
            return {"query": query, "passages": [], "reranked": False}

        self.logger.info(
            "Re-ranking %d passages for: %s",
            len(passages),
            query[:60],
        )

        # Build query-passage pairs for cross-encoder
        pairs = [[query, p.get("content", "")] for p in passages]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach cross-encoder scores and sort
        reranked = []
        for i, passage in enumerate(passages):
            reranked.append({
                **passage,
                "original_score": passage.get("score", 0),
                "rerank_score": round(float(scores[i]), 4),
            })

        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Trim to rerank_top_k
        reranked = reranked[:rerank_top_k]

        # Update the main score to be the rerank score
        for p in reranked:
            p["score"] = p["rerank_score"]

        self.logger.info(
            "Re-ranking complete: %d -> %d passages",
            len(passages),
            len(reranked),
        )

        return {
            "query": query,
            "passages": reranked,
            "reranked": True,
            "model": self.model_name,
        }
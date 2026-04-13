"""
Evaluator Agent for PolicyLens.

Scores the quality of generated answers on multiple dimensions:
relevance, completeness, clarity, and citation quality. Uses
a combination of heuristics and embedding similarity to produce
scores without requiring an additional LLM call.
"""

from sentence_transformers import SentenceTransformer, util

from src.core.base_agent import BaseAgent


class EvaluatorAgent(BaseAgent):
    """Evaluates response quality across multiple dimensions.

    Scores answers on relevance (semantic similarity to query),
    completeness (coverage of retrieved passages), clarity
    (readability heuristics), and citation quality (are sources
    actually referenced?).
    """

    def __init__(self, name: str = "evaluator", description: str = ""):
        super().__init__(name=name, description=description)
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load SBERT for similarity scoring."""
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def process(self, input_data: dict) -> dict:
        """Evaluate answer quality on multiple dimensions.

        Args:
            input_data: Must contain 'query', 'answer', 'passages'.

        Returns:
            Dictionary with dimension scores and overall quality score.
        """
        query = input_data["query"]
        answer = input_data["answer"]
        passages = input_data.get("passages", [])
        citations = input_data.get("citations", [])

        self.logger.info("Evaluating answer quality for: %s", query[:60])

        relevance = self._score_relevance(query, answer)
        completeness = self._score_completeness(answer, passages)
        clarity = self._score_clarity(answer)
        citation_quality = self._score_citations(answer, citations, passages)

        # Weighted overall score
        overall = (
            0.35 * relevance
            + 0.25 * completeness
            + 0.20 * clarity
            + 0.20 * citation_quality
        )

        return {
            "relevance": round(relevance, 4),
            "completeness": round(completeness, 4),
            "clarity": round(clarity, 4),
            "citation_quality": round(citation_quality, 4),
            "overall": round(overall, 4),
        }

    def _score_relevance(self, query: str, answer: str) -> float:
        """Score how relevant the answer is to the query using cosine similarity."""
        embeddings = self.model.encode([query, answer], normalize_embeddings=True)
        similarity = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
        return max(0.0, min(similarity, 1.0))

    def _score_completeness(self, answer: str, passages: list[dict]) -> float:
        """Score how well the answer covers the retrieved passages."""
        if not passages:
            return 0.0

        answer_embedding = self.model.encode([answer], normalize_embeddings=True)
        passage_texts = [p.get("content", "")[:500] for p in passages[:5]]
        passage_embeddings = self.model.encode(passage_texts, normalize_embeddings=True)

        similarities = util.cos_sim(answer_embedding, passage_embeddings)[0]
        coverage = sum(1 for s in similarities if float(s) > 0.3) / len(passages[:5])
        return min(coverage, 1.0)

    def _score_clarity(self, answer: str) -> float:
        """Score answer clarity using readability heuristics."""
        words = answer.split()
        sentences = answer.count(".") + answer.count("!") + answer.count("?")
        sentences = max(sentences, 1)

        avg_sentence_length = len(words) / sentences

        # Ideal: 15-25 words per sentence
        if 15 <= avg_sentence_length <= 25:
            length_score = 1.0
        elif 10 <= avg_sentence_length <= 35:
            length_score = 0.7
        else:
            length_score = 0.4

        # Penalize very short answers
        length_penalty = min(len(words) / 50, 1.0)

        # Reward structured answers (bullet points, numbered lists)
        structure_bonus = 0.1 if any(c in answer for c in ["1.", "2.", "- ", "* "]) else 0.0

        return min(length_score * length_penalty + structure_bonus, 1.0)

    def _score_citations(self, answer: str, citations: list[dict], passages: list[dict]) -> float:
        """Score citation quality: are sources actually referenced?"""
        if not passages:
            return 0.0

        # Check if answer contains source references
        source_mentions = answer.lower().count("[source")
        source_mentions += answer.lower().count("source:")

        if source_mentions == 0:
            return 0.2  # No citations at all

        # More citations (up to passage count) is better
        expected = min(len(passages), 5)
        citation_ratio = min(source_mentions / expected, 1.0)

        return 0.3 + 0.7 * citation_ratio
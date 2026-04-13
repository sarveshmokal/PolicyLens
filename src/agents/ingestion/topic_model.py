"""
Topic Model Agent for PolicyLens.

Fourth agent in the ingestion pipeline. Discovers latent topics
across document chunks using classical NLP techniques (LDA or NMF
on TF-IDF vectors). Assigns topic distributions to each chunk
and produces document-level topic summaries.

This provides the "classical NLP" layer that complements the
transformer-based components, showing breadth across ML techniques.
"""

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.base_agent import BaseAgent


class TopicModelAgent(BaseAgent):
    """Discovers topics in document chunks using LDA or NMF.

    Builds a TF-IDF matrix from chunk texts, then applies matrix
    factorization to discover latent topics. Each chunk gets a
    topic distribution vector showing its relevance to each topic.
    """

    def __init__(self, name: str = "topic_model", description: str = ""):
        super().__init__(name=name, description=description)
        self.n_topics = 10
        self.method = "nmf"  # "lda" or "nmf"
        self.max_features = 5000
        self.top_words_per_topic = 10
        self._vectorizer = None
        self._model = None

    def process(self, input_data: dict) -> dict:
        """Discover topics from document chunks.

        Args:
            input_data: Must contain 'chunks' list with 'content' field.
                        Optionally 'document' metadata.

        Returns:
            Dictionary with topic definitions, per-chunk topic assignments,
            and document-level topic summary.
        """
        chunks = input_data["chunks"]
        document = input_data.get("document", {})

        self.logger.info(
            "Running %s topic modeling on %d chunks (n_topics=%d)",
            self.method.upper(),
            len(chunks),
            self.n_topics,
        )

        texts = [chunk["content"] for chunk in chunks]

        # Build TF-IDF matrix
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        # Fit topic model
        if self.method == "lda":
            self._model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
            )
        else:
            self._model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=300,
            )

        topic_distributions = self._model.fit_transform(tfidf_matrix)

        # Extract topic definitions (top words per topic)
        feature_names = self._vectorizer.get_feature_names_out()
        topics = self._extract_topics(feature_names)

        # Assign dominant topic to each chunk
        enriched_chunks = self._assign_topics(chunks, topic_distributions)

        # Create document-level summary
        doc_summary = self._summarize_document_topics(topic_distributions, topics)

        self.logger.info(
            "Topic modeling complete. %d topics discovered for %s",
            self.n_topics,
            document.get("filename", "unknown"),
        )

        return {
            "document": document,
            "chunks": enriched_chunks,
            "topics": topics,
            "document_topic_summary": doc_summary,
        }

    def _extract_topics(self, feature_names) -> list[dict]:
        """Extract the top words for each discovered topic.

        Args:
            feature_names: Array of vocabulary words from TF-IDF.

        Returns:
            List of topic dicts with id, top words, and their weights.
        """
        topics = []
        for topic_idx, topic_weights in enumerate(self._model.components_):
            # Get indices of top words sorted by weight
            top_indices = topic_weights.argsort()[-self.top_words_per_topic:][::-1]
            top_words = [
                {
                    "word": feature_names[i],
                    "weight": round(float(topic_weights[i]), 4),
                }
                for i in top_indices
            ]
            topics.append({
                "topic_id": topic_idx,
                "label": f"Topic {topic_idx}",
                "top_words": top_words,
                "word_summary": ", ".join([w["word"] for w in top_words[:5]]),
            })
        return topics

    def _assign_topics(self, chunks: list[dict], distributions) -> list[dict]:
        """Assign topic distribution to each chunk.

        Args:
            chunks: List of chunk dicts from preprocessor.
            distributions: Topic distribution matrix from the model.

        Returns:
            Chunks enriched with topic_distribution and dominant_topic.
        """
        enriched = []
        for i, chunk in enumerate(chunks):
            dist = distributions[i]
            dominant_topic = int(dist.argmax())
            dominant_score = float(dist[dominant_topic])

            enriched_chunk = {
                **chunk,
                "dominant_topic": dominant_topic,
                "dominant_topic_score": round(dominant_score, 4),
                "topic_distribution": [round(float(s), 4) for s in dist],
            }
            enriched.append(enriched_chunk)
        return enriched

    def _summarize_document_topics(self, distributions, topics: list[dict]) -> dict:
        """Create a document-level topic summary.

        Averages topic distributions across all chunks to show
        what the overall document is about.

        Args:
            distributions: Topic distribution matrix (chunks x topics).
            topics: Topic definitions with top words.

        Returns:
            Dict with average topic weights and dominant topics.
        """
        avg_distribution = distributions.mean(axis=0)
        ranked_indices = avg_distribution.argsort()[::-1]

        ranked_topics = []
        for idx in ranked_indices:
            ranked_topics.append({
                "topic_id": int(idx),
                "average_weight": round(float(avg_distribution[idx]), 4),
                "top_words": topics[idx]["word_summary"],
            })

        return {
            "total_topics": self.n_topics,
            "method": self.method.upper(),
            "topics_ranked": ranked_topics,
            "dominant_topic": ranked_topics[0] if ranked_topics else None,
        }
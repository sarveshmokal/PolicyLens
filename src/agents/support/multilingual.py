"""
MultiLingual Agent for PolicyLens.

Detects the language of input queries and provides basic
language routing. Supports English and German detection.
For a portfolio project, this demonstrates awareness of
multilingual NLP challenges without requiring full translation
infrastructure.
"""

import re

from src.core.base_agent import BaseAgent


class MultiLingualAgent(BaseAgent):
    """Language detection and routing for queries.

    Detects whether a query is in English or German and routes
    accordingly. Can be extended with translation capabilities
    using libraries like deep-translator or Helsinki-NLP models.
    """

    def __init__(self, name: str = "multilingual", description: str = ""):
        super().__init__(name=name, description=description)

        self.german_words = {
            "und", "oder", "nicht", "ist", "sind", "das", "die", "der",
            "ein", "eine", "mit", "von", "auf", "fur", "wie", "was",
            "kann", "werden", "haben", "sein", "nach", "uber", "auch",
        }

    def process(self, input_data: dict) -> dict:
        """Detect query language and provide routing info.

        Args:
            input_data: Must contain 'query' (str).

        Returns:
            Dictionary with detected language, confidence,
            and routing recommendation.
        """
        query = input_data.get("query", "")

        language = self._detect_language(query)
        confidence = self._detection_confidence(query, language)

        self.logger.info(
            "Language detected: %s (confidence: %.2f) for: %s",
            language,
            confidence,
            query[:60],
        )

        return {
            "query": query,
            "detected_language": language,
            "confidence": round(confidence, 4),
            "needs_translation": language != "en",
            "routing": "default" if language == "en" else "translate_first",
        }

    def _detect_language(self, text: str) -> str:
        """Detect language using word frequency analysis."""
        words = set(re.findall(r"\b\w+\b", text.lower()))

        # Check for German-specific characters
        if re.search(r"[äöüß]", text.lower()):
            return "de"

        # Check overlap with German common words
        german_overlap = len(words & self.german_words)
        if german_overlap >= 2:
            return "de"

        return "en"

    def _detection_confidence(self, text: str, detected: str) -> float:
        """Estimate confidence of language detection."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        if detected == "de":
            german_count = sum(1 for w in words if w in self.german_words)
            return min(german_count / max(len(words) * 0.3, 1), 1.0)

        # English is the default, high confidence unless very short
        return min(len(words) / 5, 1.0)
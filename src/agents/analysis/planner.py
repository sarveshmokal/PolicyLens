"""
Planner Agent for PolicyLens.

First agent in the query pipeline. Analyzes incoming questions
and creates an execution plan: decomposing complex queries into
sub-questions, detecting query language, identifying likely
source documents, and selecting retrieval strategy.

For simple factual queries, passes them through unchanged.
For complex cross-document queries, breaks them into focused
sub-questions that the retriever can handle individually.
"""

import re

from src.core.base_agent import BaseAgent
from src.core.llm_provider import llm_chain


class PlannerAgent(BaseAgent):
    """Analyzes queries and creates retrieval execution plans.

    Determines query complexity, decomposes multi-part questions,
    and selects optimal retrieval parameters. Simple queries pass
    through with minimal processing. Complex queries get broken
    into sub-questions for better retrieval coverage.
    """

    def __init__(self, name: str = "planner", description: str = ""):
        super().__init__(name=name, description=description)
        self.complexity_threshold = 15  # words above this = complex

    def process(self, input_data: dict) -> dict:
        """Analyze a query and produce an execution plan.

        Args:
            input_data: Must contain 'query' (str).

        Returns:
            Dictionary with query analysis, sub-questions if needed,
            and recommended retrieval parameters.
        """
        query = input_data["query"]

        self.logger.info("Planning execution for: %s", query[:80])

        # Analyze query characteristics
        complexity = self._assess_complexity(query)
        query_type = self._detect_query_type(query)
        language = self._detect_language(query)

        # Decompose complex queries into sub-questions
        sub_questions = []
        if complexity == "complex":
            sub_questions = self._decompose_query(query)

        # Select retrieval strategy based on query type
        strategy = self._select_strategy(query_type, complexity)

        self.logger.info(
            "Plan: complexity=%s, type=%s, sub_questions=%d, strategy=%s",
            complexity,
            query_type,
            len(sub_questions),
            strategy["method"],
        )

        return {
            "original_query": query,
            "complexity": complexity,
            "query_type": query_type,
            "language": language,
            "sub_questions": sub_questions if sub_questions else [query],
            "strategy": strategy,
        }

    def _assess_complexity(self, query: str) -> str:
        """Classify query as simple or complex.

        Heuristic based on length, presence of comparison words,
        and multi-part indicators.

        Args:
            query: The raw query string.

        Returns:
            'simple' or 'complex'.
        """
        word_count = len(query.split())

        # Multi-part indicators
        complex_markers = [
            "compare", "contrast", "difference", "versus", " vs ",
            " and ", "both", "multiple", "how does.*relate",
            "what are the.*and.*",
        ]

        has_complex_marker = any(
            re.search(marker, query.lower()) for marker in complex_markers
        )

        if word_count > self.complexity_threshold or has_complex_marker:
            return "complex"
        return "simple"

    def _detect_query_type(self, query: str) -> str:
        """Classify the type of query for strategy selection.

        Args:
            query: The raw query string.

        Returns:
            Query type: 'factual', 'comparison', 'summary', or 'analytical'.
        """
        query_lower = query.lower()

        if any(w in query_lower for w in ["compare", "contrast", "difference", "versus", " vs "]):
            return "comparison"
        elif any(w in query_lower for w in ["summarize", "summary", "overview", "outline"]):
            return "summary"
        elif any(w in query_lower for w in ["why", "how does", "what impact", "analyze", "implications"]):
            return "analytical"
        else:
            return "factual"

    def _detect_language(self, query: str) -> str:
        """Simple language detection based on character patterns.

        Args:
            query: The raw query string.

        Returns:
            Language code ('en', 'de', or 'unknown').
        """
        german_patterns = [
            r"\b(und|oder|nicht|ist|sind|das|die|der|ein|eine)\b",
            r"[äöüß]",
        ]
        for pattern in german_patterns:
            if re.search(pattern, query.lower()):
                return "de"
        return "en"

    def _decompose_query(self, query: str) -> list[str]:
        """Break a complex query into focused sub-questions.

        Uses the LLM to intelligently decompose the query.
        Falls back to simple splitting if LLM fails.

        Args:
            query: Complex query to decompose.

        Returns:
            List of simpler sub-questions.
        """
        prompt = f"""Break this complex policy question into 2-4 simpler, focused sub-questions. Each sub-question should be answerable from a single document.

Original question: {query}

Return ONLY the sub-questions, one per line, numbered 1-4. No other text."""

        try:
            response = llm_chain.generate(prompt=prompt, max_tokens=256, temperature=0.2)
            lines = response.text.strip().split("\n")

            sub_questions = []
            for line in lines:
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if len(cleaned) > 10:
                    sub_questions.append(cleaned)

            if sub_questions:
                return sub_questions[:4]
        except Exception as e:
            self.logger.warning("LLM decomposition failed: %s. Using original query.", str(e))

        return [query]

    def _select_strategy(self, query_type: str, complexity: str) -> dict:
        """Select retrieval strategy based on query analysis.

        Args:
            query_type: Type classification from _detect_query_type.
            complexity: Complexity classification.

        Returns:
            Strategy dict with method, top_k, and alpha recommendations.
        """
        strategies = {
            "factual": {"method": "hybrid", "top_k": 5, "alpha": 0.5},
            "comparison": {"method": "hybrid", "top_k": 10, "alpha": 0.4},
            "summary": {"method": "dense", "top_k": 8, "alpha": 0.6},
            "analytical": {"method": "hybrid", "top_k": 10, "alpha": 0.5},
        }

        strategy = strategies.get(query_type, strategies["factual"])

        # Complex queries benefit from more results
        if complexity == "complex":
            strategy["top_k"] = min(strategy["top_k"] + 5, 20)

        return strategy
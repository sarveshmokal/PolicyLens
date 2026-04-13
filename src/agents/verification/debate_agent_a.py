"""
Debate Agent A (Advocate) for PolicyLens.

Argues in favor of the generated answer by identifying supporting
evidence, logical consistency, and strengths. Works with Debate
Agent B in an adversarial pair to stress-test answer quality.
"""

from src.core.base_agent import BaseAgent
from src.core.llm_provider import llm_chain


class DebateAgentA(BaseAgent):
    """Advocates for the generated answer with supporting arguments.

    Reviews the answer against source evidence and constructs
    the strongest possible case for why the answer is correct,
    complete, and well-supported.
    """

    def __init__(self, name: str = "debate_a", description: str = ""):
        super().__init__(name=name, description=description)

    def process(self, input_data: dict) -> dict:
        """Generate supporting arguments for the answer.

        Args:
            input_data: Must contain 'query', 'answer', and 'passages'.

        Returns:
            Dictionary with supporting arguments, evidence strength,
            and confidence assessment.
        """
        query = input_data["query"]
        answer = input_data["answer"]
        passages = input_data.get("passages", [])

        context = self._format_evidence(passages)

        prompt = f"""You are an advocate tasked with defending an answer to a policy question. Your job is to find the strongest evidence supporting this answer.

QUESTION: {query}

PROPOSED ANSWER:
{answer}

SOURCE EVIDENCE:
{context}

Analyze the answer and provide:
1. SUPPORTING EVIDENCE: Specific quotes or data from the sources that support the answer.
2. LOGICAL CONSISTENCY: Is the answer internally consistent and logically sound?
3. COMPLETENESS: Does the answer address all aspects of the question?
4. EVIDENCE STRENGTH: Rate the overall evidence strength as STRONG, MODERATE, or WEAK.
5. CONFIDENCE: On a scale of 0.0 to 1.0, how confident are you that this answer is correct based on the evidence?

Be specific and reference the source passages directly."""

        self.logger.info("Generating advocacy arguments for query: %s", query[:60])

        llm_response = llm_chain.generate(prompt=prompt, max_tokens=1024, temperature=0.3)

        confidence = self._extract_confidence(llm_response.text)

        return {
            "role": "advocate",
            "arguments": llm_response.text,
            "confidence": confidence,
            "llm_provider": llm_response.provider,
        }

    def _format_evidence(self, passages: list[dict]) -> str:
        """Format passages into labeled evidence block."""
        parts = []
        for i, p in enumerate(passages[:5], 1):
            source = p.get("source_file", "Unknown")
            page = p.get("page_number", 0)
            content = p.get("content", "")[:500]
            parts.append(f"[Source {i}: {source}, p.{page}]\n{content}")
        return "\n\n".join(parts)

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from LLM response.

        Looks for patterns like '0.8' or '0.85' near the word
        'confidence'. Falls back to 0.5 if not found.
        """
        import re
        patterns = re.findall(r"(?:confidence|confident)[:\s]*(\d\.\d+)", text.lower())
        if patterns:
            try:
                return min(float(patterns[-1]), 1.0)
            except ValueError:
                pass
        # Heuristic: look for any decimal between 0 and 1 near the end
        all_decimals = re.findall(r"\b(0\.\d+)\b", text)
        if all_decimals:
            return min(float(all_decimals[-1]), 1.0)
        return 0.5
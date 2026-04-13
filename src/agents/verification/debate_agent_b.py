"""
Debate Agent B (Challenger) for PolicyLens.

Challenges the generated answer by identifying weaknesses,
unsupported claims, missing context, and potential biases.
Works with Debate Agent A in an adversarial pair.
"""

from src.core.base_agent import BaseAgent
from src.core.llm_provider import llm_chain


class DebateAgentB(BaseAgent):
    """Challenges the generated answer with critical arguments.

    Reviews the answer against source evidence and constructs
    the strongest possible case for why the answer might be
    wrong, incomplete, biased, or poorly supported.
    """

    def __init__(self, name: str = "debate_b", description: str = ""):
        super().__init__(name=name, description=description)

    def process(self, input_data: dict) -> dict:
        """Generate challenging arguments against the answer.

        Args:
            input_data: Must contain 'query', 'answer', and 'passages'.

        Returns:
            Dictionary with challenging arguments, identified weaknesses,
            and counter-confidence assessment.
        """
        query = input_data["query"]
        answer = input_data["answer"]
        passages = input_data.get("passages", [])

        context = self._format_evidence(passages)

        prompt = f"""You are a critical challenger tasked with finding weaknesses in an answer to a policy question. Your job is to identify flaws, gaps, and unsupported claims.

QUESTION: {query}

PROPOSED ANSWER:
{answer}

SOURCE EVIDENCE:
{context}

Analyze the answer critically and provide:
1. UNSUPPORTED CLAIMS: Identify any claims in the answer that are NOT directly supported by the source evidence.
2. MISSING CONTEXT: What important information from the sources was left out of the answer?
3. POTENTIAL BIASES: Does the answer present a one-sided view when the sources show multiple perspectives?
4. FACTUAL CONCERNS: Are there any numbers, dates, or facts that could be wrong or misrepresented?
5. WEAKNESS SEVERITY: Rate the overall weakness level as CRITICAL, MODERATE, or MINOR.
6. COUNTER-CONFIDENCE: On a scale of 0.0 to 1.0, how confident are you that this answer has significant problems? (1.0 = definitely problematic)

Be specific and explain exactly what is wrong and where."""

        self.logger.info("Generating challenge arguments for query: %s", query[:60])

        llm_response = llm_chain.generate(prompt=prompt, max_tokens=1024, temperature=0.4)

        counter_confidence = self._extract_confidence(llm_response.text)

        return {
            "role": "challenger",
            "arguments": llm_response.text,
            "counter_confidence": counter_confidence,
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
        """Extract counter-confidence from LLM response."""
        import re
        patterns = re.findall(r"(?:confidence|confident)[:\s]*(\d\.\d+)", text.lower())
        if patterns:
            try:
                return min(float(patterns[-1]), 1.0)
            except ValueError:
                pass
        all_decimals = re.findall(r"\b(0\.\d+)\b", text)
        if all_decimals:
            return min(float(all_decimals[-1]), 1.0)
        return 0.5
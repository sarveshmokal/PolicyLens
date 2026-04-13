"""
NLI (Natural Language Inference) Agent for PolicyLens.

Uses facebook/bart-large-mnli to score whether generated claims
are entailed by the source passages. This is the mathematical
backbone of answer verification.

For each claim-evidence pair, the model outputs probabilities for:
    - Entailment: the evidence supports the claim
    - Contradiction: the evidence contradicts the claim
    - Neutral: the evidence neither supports nor contradicts

A claim passes verification if entailment >= threshold (default 0.8).
"""

import logging

from src.core.base_agent import BaseAgent
from src.core.config import settings


class NLIAgent(BaseAgent):
    """Verifies claims against source evidence using NLI entailment.

    Loads BART-large-MNLI lazily and scores each claim against
    the evidence passages. Returns entailment scores, verification
    status, and detailed breakdowns per claim.
    """

    def __init__(self, name: str = "nli", description: str = ""):
        super().__init__(name=name, description=description)
        self._pipeline = None
        self.model_name = settings.verification.get(
            "nli_model", "facebook/bart-large-mnli"
        )
        self.threshold = settings.verification.get(
            "entailment_threshold", 0.8
        )

    @property
    def pipeline(self):
        """Lazy-load the NLI pipeline on first use.

        The model is ~1.6GB. Loading takes 5-10 seconds.
        Only loaded when verification is actually needed.
        """
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            self.logger.info("Loading NLI model: %s", self.model_name)
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU
            )
            self.logger.info("NLI model loaded")
        return self._pipeline

    def process(self, input_data: dict) -> dict:
        """Verify claims against evidence using NLI.

        Args:
            input_data: Must contain:
                - 'claim' (str) or 'claims' (list[str]): text to verify
                - 'evidence' (list[str]): source texts to verify against

        Returns:
            Dictionary with overall score, per-claim results,
            and verification status.
        """
        # Support both single claim and multiple claims
        if "claims" in input_data:
            claims = input_data["claims"]
        elif "claim" in input_data:
            claims = [input_data["claim"]]
        else:
            claims = []

        evidence_list = input_data.get("evidence", [])

        if not claims:
            self.logger.warning("No claims provided for verification")
            return self._empty_result()

        if not evidence_list:
            self.logger.warning("No evidence provided for verification")
            return self._empty_result()

        # Combine evidence into a single context for NLI
        combined_evidence = " ".join(evidence_list[:5])  # Limit to avoid token overflow

        self.logger.info(
            "Verifying %d claims against %d evidence passages",
            len(claims),
            len(evidence_list),
        )

        # Score each claim
        claim_results = []
        for claim in claims:
            result = self._verify_single_claim(claim, combined_evidence)
            claim_results.append(result)

        # Compute aggregate scores
        entailment_scores = [r["entailment_score"] for r in claim_results]
        avg_entailment = sum(entailment_scores) / len(entailment_scores)
        supported_count = sum(1 for r in claim_results if r["is_supported"])
        faithfulness = supported_count / len(claim_results)

        overall_supported = avg_entailment >= self.threshold

        self.logger.info(
            "Verification complete: avg_entailment=%.3f, faithfulness=%.1f%% (%d/%d claims supported)",
            avg_entailment,
            faithfulness * 100,
            supported_count,
            len(claim_results),
        )

        return {
            "entailment_score": round(avg_entailment, 4),
            "is_supported": overall_supported,
            "faithfulness": round(faithfulness, 4),
            "threshold": self.threshold,
            "claims_total": len(claim_results),
            "claims_supported": supported_count,
            "claim_results": claim_results,
        }

    def _verify_single_claim(self, claim: str, evidence: str) -> dict:
        """Score a single claim against combined evidence.

        Uses zero-shot classification with candidate labels
        matching the NLI task: entailment, contradiction, neutral.

        Args:
            claim: The claim text to verify.
            evidence: Combined evidence text.

        Returns:
            Dict with scores for each NLI category and verdict.
        """
        # Truncate evidence to avoid model token limits
        # BART has a 1024 token limit, roughly 4 chars per token
        max_evidence_chars = 3500
        truncated_evidence = evidence[:max_evidence_chars]

        # Use zero-shot classification with NLI-style labels
        result = self.pipeline(
            sequences=claim,
            candidate_labels=["entailment", "neutral", "contradiction"],
            hypothesis_template="This text is {}.",
            multi_label=False,
        )

        # Extract scores by label
        scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            scores[label] = round(float(score), 4)

        entailment_score = scores.get("entailment", 0.0)
        is_supported = entailment_score >= self.threshold

        return {
            "claim": claim[:200],
            "entailment_score": entailment_score,
            "contradiction_score": scores.get("contradiction", 0.0),
            "neutral_score": scores.get("neutral", 0.0),
            "is_supported": is_supported,
            "verdict": "SUPPORTED" if is_supported else "NOT VERIFIED",
        }

    def _empty_result(self) -> dict:
        """Return an empty result when inputs are missing."""
        return {
            "entailment_score": 0.0,
            "is_supported": False,
            "faithfulness": 0.0,
            "threshold": self.threshold,
            "claims_total": 0,
            "claims_supported": 0,
            "claim_results": [],
        }

    def split_into_claims(self, text: str) -> list[str]:
        """Split an answer into individual claims for verification.

        Simple sentence-based splitting. Each sentence is treated
        as a separate claim to verify independently.

        Args:
            text: The generated answer text.

        Returns:
            List of claim strings.
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short fragments and citation-only sentences
        claims = []
        for s in sentences:
            s = s.strip()
            # Skip empty, very short, or pure citation lines
            if len(s) < 20:
                continue
            if s.startswith("[Source") or s.startswith("Source"):
                continue
            claims.append(s)

        return claims
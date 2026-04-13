"""
Verifier Agent for PolicyLens.

Orchestrates the full verification pipeline: NLI entailment check
plus adversarial debate. Produces a final verification verdict
that combines mathematical entailment scoring with LLM-based
critical analysis.

This agent consumes the output of the SynthesizerAgent and
produces a verified, scored result ready for the user.
"""

from src.core.base_agent import BaseAgent
from src.agents.verification.nli_agent import NLIAgent
from src.agents.verification.debate_agent_a import DebateAgentA
from src.agents.verification.debate_agent_b import DebateAgentB


class VerifierAgent(BaseAgent):
    """Orchestrates NLI verification and adversarial debate.

    Combines the NLI entailment score (mathematical) with debate
    agent assessments (reasoning-based) to produce a comprehensive
    verification verdict.
    """

    def __init__(self, name: str = "verifier", description: str = ""):
        super().__init__(name=name, description=description)
        self._nli_agent = None
        self._debate_a = None
        self._debate_b = None

    @property
    def nli_agent(self) -> NLIAgent:
        if self._nli_agent is None:
            self._nli_agent = NLIAgent()
        return self._nli_agent

    @property
    def debate_a(self) -> DebateAgentA:
        if self._debate_a is None:
            self._debate_a = DebateAgentA()
        return self._debate_a

    @property
    def debate_b(self) -> DebateAgentB:
        if self._debate_b is None:
            self._debate_b = DebateAgentB()
        return self._debate_b

    def process(self, input_data: dict) -> dict:
        """Run full verification: NLI + Debate.

        Args:
            input_data: Must contain:
                - 'query' (str): original question
                - 'answer' (str): generated answer to verify
                - 'passages' (list): source passages used

        Returns:
            Dictionary with NLI scores, debate summaries,
            and final verification verdict.
        """
        query = input_data["query"]
        answer = input_data["answer"]
        passages = input_data.get("passages", [])
        enable_debate = input_data.get("enable_debate", True)

        self.logger.info("Starting verification for: %s", query[:60])

        # Step 1: NLI entailment check
        evidence_texts = [p.get("content", "") for p in passages[:5]]
        claims = self.nli_agent.split_into_claims(answer)

        nli_result = self.nli_agent.execute({
            "claims": claims,
            "evidence": evidence_texts,
        })

        # Step 2: Adversarial debate (optional)
        debate_result = {}
        if enable_debate:
            debate_input = {
                "query": query,
                "answer": answer,
                "passages": passages,
            }

            advocate_result = self.debate_a.execute(debate_input)
            challenger_result = self.debate_b.execute(debate_input)

            advocate_confidence = advocate_result.get("confidence", 0.5)
            challenger_confidence = challenger_result.get("counter_confidence", 0.5)

            # Consensus score: high advocate + low challenger = good
            consensus = advocate_confidence * (1 - challenger_confidence)

            debate_result = {
                "advocate": {
                    "arguments": advocate_result.get("arguments", ""),
                    "confidence": advocate_confidence,
                },
                "challenger": {
                    "arguments": challenger_result.get("arguments", ""),
                    "counter_confidence": challenger_confidence,
                },
                "consensus_score": round(consensus, 4),
            }

        # Step 3: Final verdict
        entailment = nli_result.get("entailment_score", 0.0)
        faithfulness = nli_result.get("faithfulness", 0.0)

        # Combined score: weighted average of NLI and debate consensus
        if debate_result:
            consensus = debate_result.get("consensus_score", 0.5)
            combined_score = 0.6 * entailment + 0.4 * consensus
        else:
            combined_score = entailment

        # Determine verdict
        if combined_score >= 0.7 and faithfulness >= 0.7:
            verdict = "VERIFIED"
        elif combined_score >= 0.5:
            verdict = "PARTIALLY VERIFIED"
        else:
            verdict = "NOT VERIFIED"

        self.logger.info(
            "Verification complete: verdict=%s, entailment=%.3f, faithfulness=%.1f%%",
            verdict,
            entailment,
            faithfulness * 100,
        )

        return {
            "query": query,
            "verdict": verdict,
            "combined_score": round(combined_score, 4),
            "nli": {
                "entailment_score": entailment,
                "faithfulness": faithfulness,
                "is_supported": nli_result.get("is_supported", False),
                "claims_total": nli_result.get("claims_total", 0),
                "claims_supported": nli_result.get("claims_supported", 0),
            },
            "debate": debate_result,
        }
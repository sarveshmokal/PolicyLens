"""
LangGraph Orchestrator for PolicyLens.

Wires all 15 agents into a state machine with conditional routing.
The graph defines the complete query pipeline:

    Query -> Plan -> Retrieve -> Rerank -> Synthesize -> Verify -> Output

Conditional edges handle:
    - Simple vs complex queries (skip planning for simple ones)
    - Verification pass/fail (re-retrieve on failure)
    - Guardrails blocking (stop pipeline on injection detection)

This is the single entry point for the entire system. The FastAPI
backend calls orchestrator.run(query) and gets back a complete,
verified, cited answer.
"""

import logging
import time
from datetime import datetime, timezone
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.agents.analysis.planner import PlannerAgent
from src.agents.analysis.retriever import RetrieverAgent
from src.agents.analysis.reranker import RerankerAgent
from src.agents.analysis.synthesizer import SynthesizerAgent
from src.agents.verification.verifier import VerifierAgent
from src.agents.support.evaluator import EvaluatorAgent
from src.agents.support.guardrails import GuardrailsAgent

logger = logging.getLogger("policylens.orchestrator")


class PipelineState(TypedDict, total=False):
    """State that flows through the entire pipeline.

    Each node reads what it needs and writes its results.
    The state accumulates as it passes through the graph.
    """
    # Input
    query: str
    top_k: int
    alpha: float
    enable_debate: bool
    enable_verification: bool

    # Planning
    plan: dict
    sub_questions: list[str]
    complexity: str

    # Retrieval
    passages: list[dict]
    retrieval_method: str

    # Synthesis
    answer: str
    citations: list[dict]
    llm_provider: str
    llm_latency: float

    # Verification
    verification: dict
    verdict: str
    entailment_score: float
    faithfulness: float

    # Evaluation
    quality_scores: dict

    # Guardrails
    safety: dict
    is_blocked: bool

    # Metadata
    error: Optional[str]
    total_time: float
    timestamp: str


class Orchestrator:
    """LangGraph-based pipeline orchestrator.

    Builds and manages the state machine that connects all agents.
    Provides a simple run() method that takes a query string and
    returns a complete result with answer, citations, verification,
    and quality scores.
    """

    def __init__(self):
        self._planner = PlannerAgent()
        self._retriever = RetrieverAgent()
        self._reranker = RerankerAgent()
        self._synthesizer = SynthesizerAgent()
        self._verifier = VerifierAgent()
        self._evaluator = EvaluatorAgent()
        self._guardrails = GuardrailsAgent()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine.

        Node execution order:
            guardrails_check -> plan -> retrieve -> rerank ->
            synthesize -> verify -> evaluate

        Conditional edges:
            - After guardrails: BLOCKED -> END, CLEAN -> plan
            - After plan: uses strategy from planner
        """
        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("guardrails_check", self._node_guardrails)
        graph.add_node("plan", self._node_plan)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("rerank", self._node_rerank)
        graph.add_node("synthesize", self._node_synthesize)
        graph.add_node("verify", self._node_verify)
        graph.add_node("evaluate", self._node_evaluate)

        # Set entry point
        graph.set_entry_point("guardrails_check")

        # Conditional: after guardrails, either block or continue
        graph.add_conditional_edges(
            "guardrails_check",
            self._route_after_guardrails,
            {
                "blocked": END,
                "continue": "plan",
            },
        )

        # Linear flow after planning
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "synthesize")

        # Conditional: after synthesis, verify or skip
        graph.add_conditional_edges(
            "synthesize",
            self._route_after_synthesis,
            {
                "verify": "verify",
                "skip_verify": "evaluate",
            },
        )

        graph.add_edge("verify", "evaluate")
        graph.add_edge("evaluate", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node functions — each wraps an agent call
    # ------------------------------------------------------------------

    def _node_guardrails(self, state: PipelineState) -> dict:
        """Check input query for safety issues."""
        result = self._guardrails.execute({
            "query": state["query"],
        })

        is_blocked = result.get("has_injection", False)

        if is_blocked:
            logger.warning("Query blocked by guardrails: %s", state["query"][:60])

        return {
            "safety": result,
            "is_blocked": is_blocked,
        }

    def _node_plan(self, state: PipelineState) -> dict:
        """Analyze query and create execution plan."""
        result = self._planner.execute({"query": state["query"]})

        return {
            "plan": result,
            "sub_questions": result.get("sub_questions", [state["query"]]),
            "complexity": result.get("complexity", "simple"),
            "alpha": result.get("strategy", {}).get("alpha", state.get("alpha", 0.5)),
            "top_k": result.get("strategy", {}).get("top_k", state.get("top_k", 10)),
        }

    def _node_retrieve(self, state: PipelineState) -> dict:
        """Retrieve relevant passages for the query."""
        all_passages = []

        for sub_q in state.get("sub_questions", [state["query"]]):
            result = self._retriever.execute({
                "query": sub_q,
                "top_k": state.get("top_k", 10),
                "alpha": state.get("alpha", 0.5),
            })
            all_passages.extend(result.get("passages", []))

        # Deduplicate by chunk_id, keeping highest score
        seen = {}
        for p in all_passages:
            cid = p.get("chunk_id", "")
            if cid not in seen or p.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = p

        passages = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)

        return {
            "passages": passages[:state.get("top_k", 10)],
            "retrieval_method": "hybrid",
        }

    def _node_rerank(self, state: PipelineState) -> dict:
        """Re-rank retrieved passages with cross-encoder."""
        result = self._reranker.execute({
            "query": state["query"],
            "passages": state.get("passages", []),
        })

        return {
            "passages": result.get("passages", state.get("passages", [])),
        }

    def _node_synthesize(self, state: PipelineState) -> dict:
        """Generate cited answer from passages."""
        result = self._synthesizer.execute({
            "query": state["query"],
            "passages": state.get("passages", []),
        })

        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "llm_provider": result.get("llm_provider", ""),
            "llm_latency": result.get("llm_latency", 0),
        }

    def _node_verify(self, state: PipelineState) -> dict:
        """Run NLI verification and adversarial debate."""
        result = self._verifier.execute({
            "query": state["query"],
            "answer": state.get("answer", ""),
            "passages": state.get("passages", []),
            "enable_debate": state.get("enable_debate", True),
        })

        return {
            "verification": result,
            "verdict": result.get("verdict", "NOT VERIFIED"),
            "entailment_score": result.get("nli", {}).get("entailment_score", 0),
            "faithfulness": result.get("nli", {}).get("faithfulness", 0),
        }

    def _node_evaluate(self, state: PipelineState) -> dict:
        """Score answer quality."""
        result = self._evaluator.execute({
            "query": state["query"],
            "answer": state.get("answer", ""),
            "passages": state.get("passages", []),
            "citations": state.get("citations", []),
        })

        return {
            "quality_scores": result,
        }

    # ------------------------------------------------------------------
    # Routing functions — conditional edges
    # ------------------------------------------------------------------

    def _route_after_guardrails(self, state: PipelineState) -> str:
        """Decide whether to continue or block after guardrails."""
        if state.get("is_blocked", False):
            return "blocked"
        return "continue"

    def _route_after_synthesis(self, state: PipelineState) -> str:
        """Decide whether to run verification after synthesis."""
        if state.get("enable_verification", True):
            return "verify"
        return "skip_verify"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        enable_debate: bool = True,
        enable_verification: bool = True,
    ) -> dict:
        """Run the complete pipeline for a query.

        This is the single entry point for the entire system.
        Takes a natural language question and returns a verified,
        cited answer with quality scores.

        Args:
            query: Natural language question.
            top_k: Number of passages to retrieve.
            alpha: Dense vs sparse retrieval weight.
            enable_debate: Whether to run adversarial debate.
            enable_verification: Whether to run NLI verification.

        Returns:
            Complete result dict with answer, citations, verification,
            quality scores, and execution metadata.
        """
        logger.info("Pipeline starting for: %s", query[:80])
        start_time = time.perf_counter()

        initial_state = {
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
            "enable_debate": enable_debate,
            "enable_verification": enable_verification,
            "error": None,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error("Pipeline failed: %s", str(e))
            final_state = {
                **initial_state,
                "answer": "An error occurred while processing your query.",
                "error": str(e),
                "verdict": "ERROR",
            }

        elapsed = time.perf_counter() - start_time

        # Build final response
        result = {
            "query": query,
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "verdict": final_state.get("verdict", ""),
            "entailment_score": final_state.get("entailment_score", 0),
            "faithfulness": final_state.get("faithfulness", 0),
            "quality_scores": final_state.get("quality_scores", {}),
            "llm_provider": final_state.get("llm_provider", ""),
            "complexity": final_state.get("complexity", ""),
            "is_blocked": final_state.get("is_blocked", False),
            "safety": final_state.get("safety", {}),
            "error": final_state.get("error"),
            "total_time_seconds": round(elapsed, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Pipeline complete in %.1fs: verdict=%s, faithfulness=%.1f%%",
            elapsed,
            result["verdict"],
            result.get("faithfulness", 0) * 100,
        )

        return result


# Singleton instance
orchestrator = Orchestrator()
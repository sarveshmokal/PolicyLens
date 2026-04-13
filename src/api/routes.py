"""
API routes for PolicyLens.

Defines all REST endpoints for the policy analysis system:
    - POST /api/query — Run a full pipeline query
    - POST /api/query/fast — Quick query without verification
    - POST /api/retrieve — Retrieval only (no synthesis)
    - GET  /api/agents — List all agents and their status
    - GET  /api/agents/health — Health check all loaded agents
    - GET  /api/stats — Collection and system statistics
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.orchestrator import orchestrator
from src.core.registry import registry
from src.agents.ingestion.embedding_agent import EmbeddingAgent

logger = logging.getLogger("policylens.routes")


# ---------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for a full pipeline query."""
    question: str = Field(description="Natural language question about policy documents")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of passages to retrieve")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Dense vs sparse weight")
    enable_debate: bool = Field(default=True, description="Run adversarial debate agents")
    enable_verification: bool = Field(default=True, description="Run NLI verification")


class RetrievalRequest(BaseModel):
    """Request body for retrieval-only queries."""
    query: str = Field(description="Search query")
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    method: str = Field(default="hybrid", description="hybrid, bm25, or dense")


class QueryResponse(BaseModel):
    """Response from a full pipeline query."""
    question: str
    answer: str
    citations: list[dict] = []
    verdict: str = ""
    entailment_score: float = 0.0
    faithfulness: float = 0.0
    quality_scores: dict = {}
    llm_provider: str = ""
    complexity: str = ""
    total_time_seconds: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------
# Router
# ---------------------------------------------------------------

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def full_query(request: QueryRequest):
    """Run a full pipeline query with retrieval, synthesis, and verification.

    This is the main endpoint. Takes a natural language question and
    returns a verified, cited answer from the policy documents.
    """
    logger.info("Full query: %s", request.question[:80])

    try:
        result = orchestrator.run(
            query=request.question,
            top_k=request.top_k,
            alpha=request.alpha,
            enable_debate=request.enable_debate,
            enable_verification=request.enable_verification,
        )

        return QueryResponse(
            question=result.get("query", request.question),
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            verdict=result.get("verdict", ""),
            entailment_score=result.get("entailment_score", 0),
            faithfulness=result.get("faithfulness", 0),
            quality_scores=result.get("quality_scores", {}),
            llm_provider=result.get("llm_provider", ""),
            complexity=result.get("complexity", ""),
            total_time_seconds=result.get("total_time_seconds", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error("Query failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/fast", response_model=QueryResponse)
async def fast_query(request: QueryRequest):
    """Quick query without verification or debate.

    Same as /query but with verification and debate disabled.
    Returns faster but without entailment scores.
    """
    logger.info("Fast query: %s", request.question[:80])

    try:
        result = orchestrator.run(
            query=request.question,
            top_k=request.top_k,
            alpha=request.alpha,
            enable_debate=False,
            enable_verification=False,
        )

        return QueryResponse(
            question=result.get("query", request.question),
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            verdict=result.get("verdict", ""),
            quality_scores=result.get("quality_scores", {}),
            llm_provider=result.get("llm_provider", ""),
            complexity=result.get("complexity", ""),
            total_time_seconds=result.get("total_time_seconds", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error("Fast query failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve")
async def retrieve_only(request: RetrievalRequest):
    """Retrieval-only endpoint — returns passages without synthesis.

    Useful for debugging retrieval quality and exploring
    what the system finds for a given query.
    """
    from src.agents.analysis.retriever import RetrieverAgent

    retriever = RetrieverAgent()
    result = retriever.execute({
        "query": request.query,
        "top_k": request.top_k,
        "alpha": request.alpha,
        "method": request.method,
    })

    return result


@router.get("/agents")
async def list_agents():
    """List all registered agents with their status.

    Returns agent names, groups, enabled status, and whether
    they are currently loaded in memory.
    """
    return {
        "agents": registry.list_available(),
        "total": len(registry.list_available()),
    }


@router.get("/agents/health")
async def agents_health():
    """Run health checks on all loaded agents.

    Only checks agents that have been instantiated. Agents
    that haven't been used yet won't appear here.
    """
    return {
        "health": registry.health_check_all(),
    }


@router.get("/stats")
async def system_stats():
    """Return system statistics including collection info."""
    try:
        emb = EmbeddingAgent()
        collection_stats = emb.get_collection_stats()
    except Exception:
        collection_stats = {"error": "Could not connect to ChromaDB"}

    return {
        "collection": collection_stats,
        "agents_registered": len(registry.list_available()),
    }
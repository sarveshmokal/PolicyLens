"""
Pydantic schemas for PolicyLens agent inputs and outputs.

These models define the exact data contract between agents.
Every agent receives and returns validated, typed data instead
of raw dictionaries. This catches data shape errors at the
boundary between agents, not deep inside processing logic.

Schema naming convention:
    {Purpose}Input  - data flowing INTO an agent
    {Purpose}Output - data flowing OUT of an agent
    {Purpose}       - shared data structures used by multiple agents
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums - Constrained choices that prevent typos and invalid values
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    """Possible states for an agent health check."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ChunkingStrategy(str, Enum):
    """Supported text chunking methods."""
    RECURSIVE = "recursive"
    FIXED = "fixed"
    SEMANTIC = "semantic"


class LLMProvider(str, Enum):
    """Available LLM providers in the failover chain."""
    GROQ = "groq"
    OLLAMA = "ollama"
    FLAN_T5 = "flan-t5"


# ---------------------------------------------------------------------------
# Core data structures - Used across multiple agents
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Metadata extracted from a policy document."""
    doc_id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Document title")
    source_org: str = Field(default="", description="Publishing organization (OECD, IMF, etc.)")
    filename: str = Field(description="Original PDF filename")
    page_count: int = Field(default=0, ge=0, description="Total pages in the document")
    file_size_bytes: int = Field(default=0, ge=0, description="File size in bytes")


class TextChunk(BaseModel):
    """A single chunk of text extracted from a document."""
    chunk_id: str = Field(description="Unique identifier for this chunk")
    doc_id: str = Field(description="Parent document ID")
    content: str = Field(description="The actual text content")
    page_number: int = Field(default=0, ge=0, description="Source page in the PDF")
    chunk_index: int = Field(default=0, ge=0, description="Position within the document")
    token_count: int = Field(default=0, ge=0, description="Approximate token count")


class RetrievedPassage(BaseModel):
    """A passage returned by the retrieval system with relevance score."""
    chunk_id: str = Field(description="ID of the retrieved chunk")
    doc_id: str = Field(description="Source document ID")
    content: str = Field(description="Text content of the passage")
    score: float = Field(description="Relevance score from retrieval")
    source_file: str = Field(default="", description="Original filename for citation")
    page_number: int = Field(default=0, ge=0, description="Page number for citation")


class Citation(BaseModel):
    """A source citation attached to a generated answer."""
    source_file: str = Field(description="PDF filename")
    page_number: int = Field(default=0, description="Page number")
    chunk_id: str = Field(default="", description="Referenced chunk ID")
    relevance_score: float = Field(default=0.0, description="How relevant this source was")


class ExecutionMetadata(BaseModel):
    """Metadata attached to every agent execution result."""
    agent: str = Field(description="Name of the agent that produced this result")
    execution_time_seconds: float = Field(description="How long the agent took")
    timestamp: str = Field(description="ISO format UTC timestamp")


# ---------------------------------------------------------------------------
# Agent-specific I/O schemas
# ---------------------------------------------------------------------------

# -- Ingestion --

class IngestionInput(BaseModel):
    """Input for the document ingestion pipeline."""
    file_path: str = Field(description="Path to the PDF file")
    force_reprocess: bool = Field(default=False, description="Re-ingest even if already processed")


class IngestionOutput(BaseModel):
    """Output from the document ingestion pipeline."""
    document: DocumentMetadata
    chunks: list[TextChunk] = Field(default_factory=list)
    processing_time_seconds: float = Field(default=0.0)


# -- Retrieval --

class RetrievalInput(BaseModel):
    """Input for the retrieval agent."""
    query: str = Field(description="User's search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Dense vs sparse weight")


class RetrievalOutput(BaseModel):
    """Output from the retrieval agent."""
    query: str = Field(description="Original query")
    passages: list[RetrievedPassage] = Field(default_factory=list)
    retrieval_method: str = Field(default="hybrid", description="Method used (bm25/dense/hybrid)")


# -- Synthesis --

class SynthesisInput(BaseModel):
    """Input for the synthesizer agent."""
    query: str = Field(description="Original user question")
    passages: list[RetrievedPassage] = Field(description="Retrieved context passages")


class SynthesisOutput(BaseModel):
    """Output from the synthesizer agent."""
    query: str = Field(description="Original user question")
    answer: str = Field(description="Generated answer text")
    citations: list[Citation] = Field(default_factory=list)
    llm_provider: str = Field(default="", description="Which LLM generated this answer")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Answer confidence score")


# -- Verification --

class VerificationInput(BaseModel):
    """Input for the verification agent."""
    claim: str = Field(description="The claim or answer to verify")
    evidence: list[str] = Field(description="Source texts to verify against")


class VerificationOutput(BaseModel):
    """Output from the verification agent."""
    claim: str = Field(description="The claim that was verified")
    entailment_score: float = Field(ge=0.0, le=1.0, description="NLI entailment probability")
    is_supported: bool = Field(description="Whether claim passes the entailment threshold")
    details: str = Field(default="", description="Explanation of the verification result")


# -- Query (end-to-end pipeline) --

class QueryInput(BaseModel):
    """Input for a full end-to-end query through the pipeline."""
    question: str = Field(description="User's natural language question")
    top_k: int = Field(default=10, ge=1, le=100)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_debate: bool = Field(default=True, description="Run adversarial debate agents")
    enable_verification: bool = Field(default=True, description="Run NLI verification")


class QueryOutput(BaseModel):
    """Final output from the complete pipeline."""
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entailment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_verified: Optional[bool] = Field(default=None)
    llm_provider: str = Field(default="")
    total_time_seconds: float = Field(default=0.0)


# -- Health Check --

class AgentHealthResponse(BaseModel):
    """Health check response from a single agent."""
    agent: str
    status: AgentStatus = AgentStatus.HEALTHY
    description: str = ""
    executions: int = Field(default=0, ge=0)
    last_execution_time: Optional[float] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
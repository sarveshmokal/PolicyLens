"""
FastAPI application for PolicyLens.

Entry point for the REST API. Configures CORS, logging,
and mounts all route modules. Run with:

    uvicorn src.api.main:app --reload --port 8000
"""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import settings


def setup_logging():
    """Configure application-wide logging."""
    log_config = settings.logging_config
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get(
            "format",
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        ),
    )


setup_logging()
logger = logging.getLogger("policylens.api")

app = FastAPI(
    title="PolicyLens API",
    description="Multi-Agent RAG System for Policy Document Analysis",
    version="0.1.0",
)

# CORS middleware — allows the React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with timing."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info(
        "%s %s -> %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# Mount routes
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "PolicyLens API",
        "status": "running",
        "version": "0.1.0",
    }
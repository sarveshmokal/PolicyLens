"""
Batch ingestion script for PolicyLens.

Processes all PDFs in data/raw/ through the full ingestion pipeline:
DocProcessor -> Preprocessor -> EmbeddingAgent

Run from project root:
    python scripts/ingest_all.py
"""

import logging
import os
import sys
import time

# Add project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ingestion.doc_processor import DocProcessorAgent
from src.agents.ingestion.preprocessor import PreprocessorAgent
from src.agents.ingestion.embedding_agent import EmbeddingAgent


def setup_logging():
    """Configure logging to show progress in terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger("policylens.ingest")

    pdf_dir = os.path.join("data", "raw")
    pdf_files = sorted([
        f for f in os.listdir(pdf_dir) if f.endswith(".pdf")
    ])

    logger.info("Found %d PDFs to process in %s", len(pdf_files), pdf_dir)

    # Initialize agents once (reuse across all documents)
    doc_processor = DocProcessorAgent()
    preprocessor = PreprocessorAgent()
    embedding_agent = EmbeddingAgent()

    total_chunks = 0
    results = []
    start_time = time.perf_counter()

    for i, filename in enumerate(pdf_files, 1):
        file_path = os.path.join(pdf_dir, filename)
        logger.info("--- [%d/%d] Processing: %s ---", i, len(pdf_files), filename)

        try:
            # Stage 1: Extract text and chunk
            raw = doc_processor.execute({"file_path": file_path})

            # Stage 2: Clean and extract entities
            cleaned = preprocessor.execute(raw)

            # Stage 3: Embed and store
            embedded = embedding_agent.execute(cleaned)

            chunks_count = embedded["chunks_embedded"]
            total_chunks += chunks_count
            skipped = embedded.get("skipped", False)

            results.append({
                "filename": filename,
                "chunks": chunks_count,
                "skipped": skipped,
                "status": "skipped" if skipped else "success",
            })

            logger.info(
                "[%d/%d] %s: %d chunks %s",
                i, len(pdf_files), filename, chunks_count,
                "(skipped - already embedded)" if skipped else "(embedded)",
            )

        except Exception as e:
            logger.error("[%d/%d] Failed to process %s: %s", i, len(pdf_files), filename, str(e))
            results.append({
                "filename": filename,
                "chunks": 0,
                "skipped": False,
                "status": f"error: {str(e)}",
            })

    elapsed = time.perf_counter() - start_time

    # Print summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Documents processed: %d", len(pdf_files))
    logger.info("Total chunks embedded: %d", total_chunks)
    logger.info("Collection total: %d", embedding_agent.collection.count())
    logger.info("Total time: %.1f seconds", elapsed)
    logger.info("=" * 60)

    for r in results:
        status_icon = "SKIP" if r["skipped"] else "OK" if "error" not in r["status"] else "FAIL"
        logger.info("  [%s] %s -> %d chunks", status_icon, r["filename"], r["chunks"])


if __name__ == "__main__":
    main()
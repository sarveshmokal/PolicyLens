"""
Retrieval evaluation script for PolicyLens.

Runs all queries from data/policy_queries.json through the
RetrieverAgent in three modes (hybrid, bm25, dense) and measures
retrieval quality metrics:

    - Hit Rate@k: % of queries where at least one relevant doc is in top-k
    - MRR (Mean Reciprocal Rank): average of 1/rank of first relevant result
    - Source Precision@k: % of top-k results from the correct source document

Results are saved as a versioned experiment run in results/experiments/.

Run from project root:
    python scripts/evaluate_retrieval.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.analysis.retriever import RetrieverAgent


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_queries(filepath: str) -> list[dict]:
    """Load evaluation queries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_query(retriever: RetrieverAgent, query_data: dict, method: str, top_k: int) -> dict:
    """Run a single query and compute retrieval metrics.

    Args:
        retriever: The RetrieverAgent instance.
        query_data: Query dict with 'query' and 'expected_sources'.
        method: Retrieval method ('hybrid', 'bm25', 'dense').
        top_k: Number of results to evaluate.

    Returns:
        Dict with per-query metrics.
    """
    expected_sources = query_data.get("expected_sources", [])

    start = time.perf_counter()
    result = retriever.execute({
        "query": query_data["query"],
        "method": method,
        "top_k": top_k,
    })
    latency = time.perf_counter() - start

    passages = result.get("passages", [])

    # Hit Rate: is any relevant source in the top-k results?
    hit = False
    first_relevant_rank = None

    for rank, passage in enumerate(passages, 1):
        source_file = passage.get("source_file", "")
        if source_file in expected_sources:
            hit = True
            if first_relevant_rank is None:
                first_relevant_rank = rank

    # MRR: reciprocal of the rank of the first relevant result
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

    # Source Precision@k: fraction of top-k from correct sources
    correct_sources = sum(
        1 for p in passages if p.get("source_file", "") in expected_sources
    )
    source_precision = correct_sources / len(passages) if passages else 0.0

    return {
        "query_id": query_data.get("id", 0),
        "query": query_data["query"][:80],
        "category": query_data.get("category", ""),
        "difficulty": query_data.get("difficulty", ""),
        "method": method,
        "hit": hit,
        "mrr": round(mrr, 4),
        "source_precision": round(source_precision, 4),
        "first_relevant_rank": first_relevant_rank,
        "total_results": len(passages),
        "latency_seconds": round(latency, 4),
        "expected_sources": expected_sources,
    }


def compute_aggregate_metrics(per_query_results: list[dict]) -> dict:
    """Compute aggregate metrics across all queries.

    Args:
        per_query_results: List of per-query metric dicts.

    Returns:
        Dict with averaged metrics.
    """
    n = len(per_query_results)
    if n == 0:
        return {}

    hit_rate = sum(1 for r in per_query_results if r["hit"]) / n
    avg_mrr = sum(r["mrr"] for r in per_query_results) / n
    avg_precision = sum(r["source_precision"] for r in per_query_results) / n
    avg_latency = sum(r["latency_seconds"] for r in per_query_results) / n

    # Break down by category
    categories = set(r["category"] for r in per_query_results)
    by_category = {}
    for cat in categories:
        cat_results = [r for r in per_query_results if r["category"] == cat]
        cat_n = len(cat_results)
        by_category[cat] = {
            "count": cat_n,
            "hit_rate": round(sum(1 for r in cat_results if r["hit"]) / cat_n, 4),
            "avg_mrr": round(sum(r["mrr"] for r in cat_results) / cat_n, 4),
            "avg_source_precision": round(
                sum(r["source_precision"] for r in cat_results) / cat_n, 4
            ),
        }

    # Break down by difficulty
    difficulties = set(r["difficulty"] for r in per_query_results)
    by_difficulty = {}
    for diff in difficulties:
        diff_results = [r for r in per_query_results if r["difficulty"] == diff]
        diff_n = len(diff_results)
        by_difficulty[diff] = {
            "count": diff_n,
            "hit_rate": round(sum(1 for r in diff_results if r["hit"]) / diff_n, 4),
            "avg_mrr": round(sum(r["mrr"] for r in diff_results) / diff_n, 4),
        }

    return {
        "total_queries": n,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(avg_mrr, 4),
        "source_precision": round(avg_precision, 4),
        "avg_latency_seconds": round(avg_latency, 4),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }


def save_experiment(run_name: str, config: dict, results: dict, per_query: list[dict]):
    """Save experiment results as a versioned JSON file.

    Args:
        run_name: Identifier for this experiment run.
        config: Configuration used for this run.
        results: Aggregate metrics by method.
        per_query: Detailed per-query results.
    """
    output_dir = os.path.join("results", "experiments")
    os.makedirs(output_dir, exist_ok=True)

    experiment = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results": results,
        "per_query_details": per_query,
    }

    filepath = os.path.join(output_dir, f"{run_name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, ensure_ascii=False)

    return filepath


def main():
    setup_logging()
    logger = logging.getLogger("policylens.eval")

    queries = load_queries(os.path.join("data", "policy_queries.json"))
    logger.info("Loaded %d evaluation queries", len(queries))

    retriever = RetrieverAgent()
    top_k = 5
    alpha = 0.5
    methods = ["bm25", "dense", "hybrid"]

    all_results = {}
    all_per_query = []

    for method in methods:
        logger.info("=" * 50)
        logger.info("Evaluating method: %s (alpha=%.2f, top_k=%d)", method, alpha, top_k)
        logger.info("=" * 50)

        per_query_results = []
        for q in queries:
            result = evaluate_query(retriever, q, method, top_k)
            per_query_results.append(result)
            all_per_query.append(result)

            status = "HIT" if result["hit"] else "MISS"
            logger.info(
                "  [%s] Q%d (MRR=%.2f): %s",
                status, result["query_id"], result["mrr"], result["query"],
            )

        metrics = compute_aggregate_metrics(per_query_results)
        all_results[method] = metrics

        logger.info("-" * 50)
        logger.info(
            "%s Results: Hit Rate=%.2f%%, MRR=%.4f, Source Precision=%.4f, Avg Latency=%.3fs",
            method.upper(),
            metrics["hit_rate"] * 100,
            metrics["mrr"],
            metrics["source_precision"],
            metrics["avg_latency_seconds"],
        )

    # Print comparison table
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE RETRIEVAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("%-10s | %-10s | %-10s | %-15s | %-10s", "Method", "Hit Rate", "MRR", "Src Precision", "Latency")
    logger.info("-" * 60)
    for method in methods:
        m = all_results[method]
        logger.info(
            "%-10s | %-9.1f%% | %-10.4f | %-15.4f | %-9.3fs",
            method.upper(),
            m["hit_rate"] * 100,
            m["mrr"],
            m["source_precision"],
            m["avg_latency_seconds"],
        )
    logger.info("=" * 60)

    # Save experiment
    config = {
        "alpha": alpha,
        "top_k": top_k,
        "methods": methods,
        "total_chunks_in_collection": 7504,
        "version": "v1_baseline",
        "description": "Initial baseline evaluation with default alpha=0.5",
    }

    filepath = save_experiment("run_001_baseline", config, all_results, all_per_query)
    logger.info("Experiment saved to: %s", filepath)


if __name__ == "__main__":
    main()
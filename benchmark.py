#!/usr/bin/env python3
"""
benchmark.py — Compare token usage: file_agent vs mcp_agent on FastAPI source.

Usage:
    python benchmark.py                     # run all tasks, both agents
    python benchmark.py --tasks t1 t3       # run specific tasks
    python benchmark.py --agents file_agent # run only file agent
    python benchmark.py --no-mcp-index      # skip re-indexing (MCP already indexed)

Environment:
    ANTHROPIC_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tasks import TASKS, Task
from agents import RunResult, run_file_agent, run_mcp_agent
from report import print_comparison, save_json

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="codebase-memory-mcp benchmark")
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK_ID",
        help="Task IDs to run (default: all). E.g. --tasks t1_discovery t3_impact",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["file_agent", "mcp_agent"],
        default=["file_agent", "mcp_agent"],
        help="Which agents to run (default: both)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "latest.json",
        help="Path for JSON results output",
    )
    return parser.parse_args()


def select_tasks(task_ids: list[str] | None) -> list[Task]:
    if not task_ids:
        return TASKS
    id_set = set(task_ids)
    selected = [t for t in TASKS if t.id in id_set]
    missing = id_set - {t.id for t in selected}
    if missing:
        print(f"WARNING: unknown task IDs: {missing}", file=sys.stderr)
    return selected


def run_one(agent: str, task: Task) -> RunResult:
    print(f"\n  Running [{agent}] on [{task.id}] …", flush=True)
    t0 = time.perf_counter()

    if agent == "file_agent":
        result = run_file_agent(task.id, task.prompt)
    else:
        result = run_mcp_agent(task.id, task.prompt)

    elapsed = time.perf_counter() - t0
    status = f"ERROR: {result.error}" if result.error else "ok"
    print(
        f"  → {status}  "
        f"in={result.input_tokens:,}  out={result.output_tokens:,}  "
        f"tools={result.tool_calls}  iters={result.iterations}  "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    tasks = select_tasks(args.tasks)
    agents = args.agents

    print(f"\nBenchmark: {len(tasks)} task(s) × {len(agents)} agent(s) = {len(tasks)*len(agents)} run(s)")
    print(f"Target repo: target_repo/ (FastAPI — MIT license)")
    print(f"Model: claude-sonnet-4-6")

    results: list[RunResult] = []

    # Interleave agents per task so both face the same task in sequence.
    # This reduces ordering bias from conversation-history token growth.
    for task in tasks:
        for agent in agents:
            result = run_one(agent, task)
            results.append(result)

    print_comparison(results)
    save_json(results, args.output)


if __name__ == "__main__":
    main()

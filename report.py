"""
Formats benchmark results as a human-readable table and saves JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

from agents import RunResult


def _row(cells: list, widths: list[int]) -> str:
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def _divider(widths: list[int]) -> str:
    return "+-" + "-+-".join("-" * w for w in widths) + "-+"


def print_comparison(results: list[RunResult]) -> None:
    # Group by task
    by_task: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, {})[r.agent] = r

    cols = ["Task", "Agent", "Input tok", "Output tok", "Tool calls", "Iters", "Status"]
    widths = [max(len(c), 30) for c in cols]
    widths[0] = 20
    widths[1] = 12
    widths[2] = 10
    widths[3] = 11
    widths[4] = 11
    widths[5] = 6
    widths[6] = 8

    print()
    print("=" * 90)
    print("  BENCHMARK RESULTS — FastAPI codebase — file_agent vs mcp_agent")
    print("=" * 90)
    print(_divider(widths))
    print(_row(cols, widths))
    print(_divider(widths))

    for task_id, agents in sorted(by_task.items()):
        for agent_name in ("file_agent", "mcp_agent"):
            r = agents.get(agent_name)
            if r is None:
                continue
            status = "ERROR" if r.error else "ok"
            print(
                _row(
                    [
                        task_id,
                        agent_name,
                        f"{r.input_tokens:,}",
                        f"{r.output_tokens:,}",
                        r.tool_calls,
                        r.iterations,
                        status,
                    ],
                    widths,
                )
            )

        # Print ratio row
        fa = agents.get("file_agent")
        ma = agents.get("mcp_agent")
        if fa and ma and ma.input_tokens > 0:
            ratio = fa.input_tokens / ma.input_tokens
            savings = (1 - 1 / ratio) * 100 if ratio > 1 else 0
            print(
                _row(
                    [
                        "",
                        f"ratio {ratio:.2f}x",
                        f"(-{savings:.0f}% MCP)",
                        "",
                        "",
                        "",
                        "",
                    ],
                    widths,
                )
            )
        print(_divider(widths))

    # Summary
    total_fa_in = sum(r.input_tokens for r in results if r.agent == "file_agent")
    total_ma_in = sum(r.input_tokens for r in results if r.agent == "mcp_agent")
    if total_ma_in > 0:
        overall_ratio = total_fa_in / total_ma_in
        print()
        print(
            f"  Overall input-token ratio  (file / mcp): {overall_ratio:.2f}x"
            f"  ({total_fa_in:,} vs {total_ma_in:,})"
        )
    print()


def save_json(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: list[dict[str, Any]] = []
    for r in results:
        data.append(
            {
                "agent": r.agent,
                "task_id": r.task_id,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "tool_calls": r.tool_calls,
                "iterations": r.iterations,
                "error": r.error,
                "tool_call_log": r.tool_call_log,
                "answer_chars": len(r.answer),
            }
        )
    path.write_text(json.dumps({"timestamp": datetime.utcnow().isoformat(), "results": data}, indent=2))
    print(f"  Results saved to {path}")

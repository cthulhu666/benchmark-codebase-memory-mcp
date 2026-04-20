"""
simulate.py — Token-cost benchmark without an LLM orchestrator.

For each task we define the tool call sequence a competent agent *would* make,
then execute those calls against the real codebase / real MCP server and
measure the total bytes returned.  Bytes ÷ 4 ≈ Claude input tokens.

This captures the dominant cost driver: it's the tool *responses* (raw file
content vs. compact graph data) that determine how expensive a task is, not
the model's own output text.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_PATH = Path(__file__).parent / "target_repo"
MCP_COMMAND = "/Users/cthulhu/.local/bin/codebase-memory-mcp"
MCP_PROJECT = "Users-cthulhu-Dev-benchmark-codebase-memory-mcp-target_repo"

# Approx chars per Claude token (conservative — real ratio is ~3.5-4)
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    tool: str
    args: dict
    response_chars: int = 0
    response_preview: str = ""


@dataclass
class AgentTrace:
    agent: str
    task_id: str
    calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None

    @property
    def total_chars(self) -> int:
        return sum(c.response_chars for c in self.calls)

    @property
    def estimated_tokens(self) -> int:
        return self.total_chars // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# File-agent tool implementations (sync)
# ---------------------------------------------------------------------------


def file_list(pattern: str) -> str:
    matches = sorted(REPO_PATH.glob(pattern))
    return json.dumps([str(m.relative_to(REPO_PATH)) for m in matches if m.is_file()])


def file_read(path: str) -> str:
    p = REPO_PATH / path
    return p.read_text(errors="replace") if p.exists() else f"NOT FOUND: {path}"


def file_grep(pattern: str, path: str = "", file_glob: str = "**/*.py", max_results: int = 300) -> str:
    compiled = re.compile(pattern)
    root = REPO_PATH / path
    results = []
    for fpath in sorted(root.glob(file_glob)):
        if not fpath.is_file():
            continue
        try:
            for lineno, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                if compiled.search(line):
                    results.append({"file": str(fpath.relative_to(REPO_PATH)), "line": lineno, "content": line.rstrip()})
                    if len(results) >= max_results:
                        return json.dumps(results)
        except Exception:
            pass
    return json.dumps(results)


def exec_file_tool(name: str, args: dict) -> str:
    if name == "list_files":
        return file_list(args["pattern"])
    if name == "read_file":
        return file_read(args["path"])
    if name == "grep_code":
        return file_grep(
            args["pattern"],
            args.get("path", ""),
            args.get("file_glob", "**/*.py"),
            args.get("max_results", 300),
        )
    return f"UNKNOWN TOOL: {name}"


# ---------------------------------------------------------------------------
# MCP tool execution (async)
# ---------------------------------------------------------------------------


async def exec_mcp_tool(session: ClientSession, name: str, args: dict) -> str:
    mcp_args = {"project": MCP_PROJECT, **args}
    try:
        result = await session.call_tool(name, mcp_args)
        parts = [item.text for item in result.content if hasattr(item, "text")]
        return "\n".join(parts)
    except Exception as exc:
        return f"MCP ERROR: {exc}"


# ---------------------------------------------------------------------------
# Task traces
#
# Each task defines two lists of (tool, args) pairs:
#   file_steps  — what a file-reading agent would do
#   mcp_steps   — what an MCP agent would do
#
# The file_steps are chosen to be *realistic but not exhaustive*: a competent
# agent would list files, do a targeted grep to narrow down, then read the
# relevant files.  We don't make it artificially bad.
# ---------------------------------------------------------------------------


TASK_TRACES = {
    # -----------------------------------------------------------------------
    # T1: Discover all HTTP method decorators in fastapi/
    # -----------------------------------------------------------------------
    "t1_discovery": {
        "name": "Route discovery",
        "file_steps": [
            # List the package to know what's there
            ("list_files", {"pattern": "fastapi/**/*.py"}),
            # Grep for decorator-style route registrations
            ("grep_code", {"pattern": r"@(app|router)\.(get|post|put|patch|delete|head|options|trace)\(", "path": "fastapi", "file_glob": "**/*.py"}),
            # The grep misses programmatic registrations — read the two routing files
            ("read_file", {"path": "fastapi/routing.py"}),
            ("read_file", {"path": "fastapi/applications.py"}),
        ],
        "mcp_steps": [
            # Architecture gives the lay of the land
            ("get_architecture", {"aspects": ["routing"]}),
            # Search for Route nodes directly
            ("search_graph", {"label": "Route", "file_pattern": "fastapi/"}),
            # Find the HTTP-method decorator functions on APIRouter/FastAPI
            ("search_graph", {"name_pattern": r"^(get|post|put|patch|delete|head|options|trace)$", "label": "Function", "file_pattern": "fastapi/routing"}),
        ],
    },

    # -----------------------------------------------------------------------
    # T2: Find all APIRouter instantiations in tests/ with their kwargs
    # -----------------------------------------------------------------------
    "t2_pattern": {
        "name": "APIRouter instantiation pattern",
        "file_steps": [
            # Grep to find which test files use APIRouter
            ("grep_code", {"pattern": r"APIRouter\(", "path": "tests", "file_glob": "**/*.py"}),
            # Read the files that came back (top 8 unique files is typical)
            ("read_file", {"path": "tests/test_router_prefix.py"}),
            ("read_file", {"path": "tests/test_router_events.py"}),
            ("read_file", {"path": "tests/test_include_router.py"}),
            ("read_file", {"path": "tests/test_router_prefix_with_template.py"}),
            ("read_file", {"path": "tests/test_application.py"}),
            ("read_file", {"path": "tests/test_router_dependencies.py"}),
            ("read_file", {"path": "tests/test_router_multihost.py"}),
            ("read_file", {"path": "tests/test_callable_endpoint.py"}),
        ],
        "mcp_steps": [
            # One graph query returns all instantiation sites
            ("query_graph", {"query": "MATCH (f:Function)-[:CALLS]->(c) WHERE c.name = 'APIRouter' RETURN f.name, f.file_path, c.name LIMIT 60"}),
            # Get the APIRouter class definition to see what kwargs are valid
            ("get_code_snippet", {"qualified_name": "APIRouter.__init__"}),
        ],
    },

    # -----------------------------------------------------------------------
    # T3: Find every caller of get_dependant in fastapi/
    # -----------------------------------------------------------------------
    "t3_impact": {
        "name": "Dependency injection impact",
        "file_steps": [
            # Find the definition first
            ("grep_code", {"pattern": r"def get_dependant", "path": "fastapi", "file_glob": "**/*.py"}),
            # Find all call sites
            ("grep_code", {"pattern": r"get_dependant\(", "path": "fastapi", "file_glob": "**/*.py"}),
            # Read files that contain calls (typically 2-3 files)
            ("read_file", {"path": "fastapi/dependencies/utils.py"}),
            ("read_file", {"path": "fastapi/routing.py"}),
        ],
        "mcp_steps": [
            # Direct reverse-call query
            ("query_graph", {"query": "MATCH (caller)-[:CALLS]->(callee) WHERE callee.name = 'get_dependant' RETURN caller.name, caller.file_path, callee.name"}),
            # Get the function signature to understand why it's called
            ("get_code_snippet", {"qualified_name": "get_dependant"}),
        ],
    },

    # -----------------------------------------------------------------------
    # T4: Full call chain FastAPI.__call__ → route handler invocation
    # -----------------------------------------------------------------------
    "t4_trace": {
        "name": "Request routing call chain",
        "file_steps": [
            # Start at the ASGI entry point
            ("read_file", {"path": "fastapi/applications.py"}),
            # FastAPI inherits from Starlette — need to find __call__ there
            ("grep_code", {"pattern": r"def __call__", "path": "fastapi", "file_glob": "**/*.py"}),
            # The routing machinery lives here
            ("read_file", {"path": "fastapi/routing.py"}),
            # Low-level request handling
            ("grep_code", {"pattern": r"async def handle", "path": "fastapi", "file_glob": "**/*.py"}),
            # Dependency solver is called on every request
            ("read_file", {"path": "fastapi/dependencies/utils.py"}),
        ],
        "mcp_steps": [
            # Outbound call tree from the ASGI entry point
            ("trace_path", {"function_name": "__call__", "direction": "outbound", "depth": 6, "mode": "calls"}),
            # Outbound from solve_dependencies to capture the dep-injection leg
            ("trace_path", {"function_name": "solve_dependencies", "direction": "outbound", "depth": 5, "mode": "calls"}),
        ],
    },
}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_file_trace(task_id: str) -> AgentTrace:
    task = TASK_TRACES[task_id]
    trace = AgentTrace(agent="file_agent", task_id=task_id)
    for tool, args in task["file_steps"]:
        response = exec_file_tool(tool, args)
        call = ToolCall(
            tool=tool,
            args=args,
            response_chars=len(response),
            response_preview=response[:120].replace("\n", " "),
        )
        trace.calls.append(call)
    return trace


async def run_mcp_trace(task_id: str, session: ClientSession) -> AgentTrace:
    task = TASK_TRACES[task_id]
    trace = AgentTrace(agent="mcp_agent", task_id=task_id)
    for tool, args in task["mcp_steps"]:
        response = await exec_mcp_tool(session, tool, args)
        call = ToolCall(
            tool=tool,
            args=args,
            response_chars=len(response),
            response_preview=response[:120].replace("\n", " "),
        )
        trace.calls.append(call)
    return trace


REPO_PATH_ABS = str(REPO_PATH)

async def _ensure_indexed(session: ClientSession) -> None:
    result = await session.call_tool("list_projects", {})
    if MCP_PROJECT not in str(result.content):
        print("  [setup] Indexing repo (one-time) …", flush=True)
        await session.call_tool("index_repository", {"repo_path": REPO_PATH_ABS, "mode": "full"})
        for _ in range(120):
            st = await session.call_tool("index_status", {"project": MCP_PROJECT})
            if "completed" in str(st.content).lower() or "indexed" in str(st.content).lower():
                break
            await asyncio.sleep(5)
        print("  [setup] Done.", flush=True)
    else:
        print("  [setup] Repo already indexed.", flush=True)


async def run_all() -> list[AgentTrace]:
    traces: list[AgentTrace] = []

    server_params = StdioServerParameters(command=MCP_COMMAND, args=[])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await _ensure_indexed(session)

            for task_id in TASK_TRACES:
                print(f"\n  Task: {task_id}")

                print(f"    [file_agent] running …", flush=True)
                ft = run_file_trace(task_id)
                traces.append(ft)
                print(f"    [file_agent] {ft.total_chars:,} chars  ~{ft.estimated_tokens:,} tokens  ({len(ft.calls)} calls)")

                print(f"    [mcp_agent]  running …", flush=True)
                mt = await run_mcp_trace(task_id, session)
                traces.append(mt)
                print(f"    [mcp_agent]  {mt.total_chars:,} chars  ~{mt.estimated_tokens:,} tokens  ({len(mt.calls)} calls)")

    return traces


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(traces: list[AgentTrace]) -> None:
    by_task: dict[str, dict[str, AgentTrace]] = {}
    for t in traces:
        by_task.setdefault(t.task_id, {})[t.agent] = t

    print()
    print("=" * 100)
    print("  BENCHMARK RESULTS — FastAPI (MIT) — file_agent vs mcp_agent")
    print("  Method: measure context pulled per task; tokens estimated at 4 chars/token")
    print("=" * 100)

    hdr = f"  {'Task':<22} {'Agent':<12} {'Calls':>6} {'Chars':>12} {'~Tokens':>10} {'Ratio':>10}"
    print(hdr)
    print("  " + "-" * 76)

    total_file, total_mcp = 0, 0
    for task_id, agents in sorted(by_task.items()):
        task_name = TASK_TRACES[task_id]["name"]
        fa = agents.get("file_agent")
        ma = agents.get("mcp_agent")

        for agent_name, tr in [("file_agent", fa), ("mcp_agent", ma)]:
            if tr is None:
                continue
            print(f"  {task_id:<22} {agent_name:<12} {len(tr.calls):>6} {tr.total_chars:>12,} {tr.estimated_tokens:>10,}")

        if fa and ma and ma.estimated_tokens > 0:
            ratio = fa.estimated_tokens / ma.estimated_tokens
            savings = (1 - 1 / ratio) * 100
            print(f"  {'':22} {'↑ ratio':<12} {'':>6} {'':>12} {'':>10} {ratio:>9.1f}x  (MCP saves ~{savings:.0f}%)")
            total_file += fa.estimated_tokens
            total_mcp += ma.estimated_tokens

        print("  " + "-" * 76)

    if total_mcp > 0:
        overall = total_file / total_mcp
        print()
        print(f"  TOTAL — file_agent: ~{total_file:,} tokens   mcp_agent: ~{total_mcp:,} tokens")
        print(f"  Overall ratio: {overall:.1f}x  (MCP uses ~{100/overall:.0f}% of file-agent context)")
    print()

    # Per-call breakdown
    print("  TOOL CALL BREAKDOWN")
    print("  " + "-" * 76)
    for task_id, agents in sorted(by_task.items()):
        print(f"\n  [{task_id}]")
        for agent_name in ("file_agent", "mcp_agent"):
            tr = agents.get(agent_name)
            if not tr:
                continue
            print(f"    {agent_name}:")
            for c in tr.calls:
                key = list(c.args.values())[0] if c.args else ""
                print(f"      {c.tool:<28} {c.response_chars:>8,} chars   {str(key)[:50]}")


def save_results(traces: list[AgentTrace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for t in traces:
        out.append({
            "agent": t.agent,
            "task_id": t.task_id,
            "total_chars": t.total_chars,
            "estimated_tokens": t.estimated_tokens,
            "num_calls": len(t.calls),
            "calls": [
                {"tool": c.tool, "args": c.args, "response_chars": c.response_chars}
                for c in t.calls
            ],
        })
    path.write_text(json.dumps({"note": "token estimate = chars / 4", "traces": out}, indent=2))
    print(f"\n  Results saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASK_TRACES.keys()))
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    args = parser.parse_args()

    # Filter to requested tasks
    all_task_ids = list(TASK_TRACES.keys())
    task_ids = [t for t in all_task_ids if t in args.tasks]
    if not task_ids:
        print("No matching tasks.", file=sys.stderr)
        sys.exit(1)

    # Temporarily filter TASK_TRACES
    original = dict(TASK_TRACES)
    for k in list(TASK_TRACES.keys()):
        if k not in task_ids:
            del TASK_TRACES[k]

    traces = asyncio.run(run_all())
    print_report(traces)
    save_results(traces, args.output)

    TASK_TRACES.update(original)

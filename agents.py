"""
Two agent implementations for the benchmark:

FileAgent  — uses only read_file / list_files / grep_code (classic file I/O)
McpAgent   — uses codebase-memory-mcp tools via MCP stdio transport

Both share the same run_agent() driver which calls the Anthropic API in a
standard tool-use loop and accumulates token usage statistics.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

REPO_PATH = Path(__file__).parent / "target_repo"
MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 40  # safety cap per task run

SYSTEM_PROMPT = """\
You are a code analysis assistant. The repository under analysis is the FastAPI
source code located at {repo_path}.

Use your tools to answer the question accurately and completely.
When you have gathered enough information to give a full answer, stop calling
tools and write your final response.
""".strip()

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    agent: str
    task_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    iterations: int = 0
    error: str | None = None
    answer: str = ""
    # Per-tool-call breakdown for analysis
    tool_call_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anthropic client (shared)
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic()


async def _run_tool_loop(
    tools: list[dict],
    tool_executor,  # async callable(name, inputs) -> Any
    task_prompt: str,
    result: RunResult,
) -> None:
    """Async Anthropic tool-use loop. Mutates *result* in place."""
    messages: list[dict] = [{"role": "user", "content": task_prompt}]
    system = SYSTEM_PROMPT.format(repo_path=str(REPO_PATH))

    for _ in range(MAX_ITERATIONS):
        result.iterations += 1
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
            ),
        )
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    result.answer += block.text
            break

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result.tool_calls += 1
            tool_output = await tool_executor(block.name, block.input)
            serialised = (
                json.dumps(tool_output)
                if not isinstance(tool_output, str)
                else tool_output
            )
            result.tool_call_log.append(
                {
                    "tool": block.name,
                    "input": block.input,
                    "output_chars": len(serialised),
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": serialised,
                }
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


# ===========================================================================
# FILE AGENT
# ===========================================================================

FILE_TOOLS = [
    {
        "name": "list_files",
        "description": (
            "List files in the repository matching a glob pattern. "
            "Returns relative paths from the repo root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. 'fastapi/**/*.py' or 'tests/*.py'",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full content of a file given its path relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from repo root"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep_code",
        "description": (
            "Search file contents for a regex pattern. "
            "Returns matching lines with file path and line number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex"},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (relative to repo root). Defaults to repo root.",
                    "default": "",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Optional glob to filter files, e.g. '*.py'",
                    "default": "**/*.py",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Cap on results returned. Default 200.",
                    "default": 200,
                },
            },
            "required": ["pattern"],
        },
    },
]


def _file_tool_executor(name: str, inputs: dict) -> Any:
    if name == "list_files":
        pattern = inputs["pattern"]
        matches = sorted(REPO_PATH.glob(pattern))
        return [str(m.relative_to(REPO_PATH)) for m in matches if m.is_file()]

    if name == "read_file":
        p = REPO_PATH / inputs["path"]
        if not p.exists():
            return f"ERROR: file not found: {inputs['path']}"
        return p.read_text(errors="replace")

    if name == "grep_code":
        pattern = inputs["pattern"]
        search_root = REPO_PATH / inputs.get("path", "")
        file_glob = inputs.get("file_glob", "**/*.py")
        max_results = int(inputs.get("max_results", 200))
        compiled = re.compile(pattern)
        results = []
        for fpath in sorted(search_root.glob(file_glob)):
            if not fpath.is_file():
                continue
            try:
                for lineno, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        results.append(
                            {
                                "file": str(fpath.relative_to(REPO_PATH)),
                                "line": lineno,
                                "content": line.rstrip(),
                            }
                        )
                        if len(results) >= max_results:
                            return results
            except Exception:
                pass
        return results

    return f"ERROR: unknown tool {name}"


async def _async_file_tool_executor(name: str, inputs: dict) -> Any:
    return _file_tool_executor(name, inputs)


async def _run_file_agent(task_id: str, task_prompt: str) -> RunResult:
    result = RunResult(agent="file_agent", task_id=task_id)
    try:
        await _run_tool_loop(FILE_TOOLS, _async_file_tool_executor, task_prompt, result)
    except Exception as exc:
        result.error = str(exc)
    return result


def run_file_agent(task_id: str, task_prompt: str) -> RunResult:
    return asyncio.run(_run_file_agent(task_id, task_prompt))


# ===========================================================================
# MCP AGENT
# ===========================================================================

MCP_COMMAND = "/Users/cthulhu/.local/bin/codebase-memory-mcp"
MCP_PROJECT = str(REPO_PATH)

MCP_TOOLS = [
    {
        "name": "get_architecture",
        "description": (
            "Get a high-level architecture overview of the codebase — packages, "
            "services, dependencies, and project structure at a glance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of aspects to focus on, e.g. ['routing', 'middleware']",
                }
            },
        },
    },
    {
        "name": "search_graph",
        "description": (
            "Search the code knowledge graph for functions, classes, routes, or variables. "
            "Far more precise than grep — returns structured node data without reading whole files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name_pattern": {
                    "type": "string",
                    "description": "Regex pattern to match node names",
                },
                "label": {
                    "type": "string",
                    "description": "Node type filter: Function, Class, Route, Variable, Module",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Regex to filter by file path",
                },
                "include_connected": {
                    "type": "boolean",
                    "description": "Include directly connected nodes",
                },
                "limit": {"type": "integer", "description": "Max results"},
            },
        },
    },
    {
        "name": "query_graph",
        "description": (
            "Run a Cypher query against the knowledge graph. "
            "Use for relationship traversals: CALLS, IMPORTS, DEFINES, IMPLEMENTS. "
            "Example: MATCH (a)-[:CALLS]->(b) WHERE a.name='foo' RETURN b.name, b.file_path"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Cypher query string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_code_snippet",
        "description": (
            "Retrieve the source code of a specific function or class. "
            "Much cheaper than read_file — returns only the relevant definition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function or class name"},
                "file_path": {
                    "type": "string",
                    "description": "Optional: narrow to a specific file",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Full-text search across the repository source. "
            "Returns matching lines with context. Use when you need raw text matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "file_pattern": {
                    "type": "string",
                    "description": "Limit to files matching this pattern, e.g. 'fastapi/'",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "trace_call_path",
        "description": (
            "Trace the call chain between two functions. "
            "Returns all intermediate steps on the path from source to target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "description": "Starting function name",
                },
                "target_name": {
                    "type": "string",
                    "description": "Ending function name",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum call depth to explore. Default 10.",
                    "default": 10,
                },
            },
            "required": ["source_name", "target_name"],
        },
    },
]


async def _index_if_needed(session: ClientSession) -> None:
    """Index the repo if it has not been indexed yet (setup cost, not benchmarked)."""
    result = await session.call_tool("list_projects", {})
    projects_text = str(result.content)
    if MCP_PROJECT not in projects_text:
        print(f"  [setup] Indexing {MCP_PROJECT} …", flush=True)
        await session.call_tool("index_repository", {"repo_path": MCP_PROJECT, "mode": "full"})
        # Wait for indexing to complete
        for _ in range(60):
            status = await session.call_tool("index_status", {"project": MCP_PROJECT})
            status_text = str(status.content)
            if "completed" in status_text.lower() or "indexed" in status_text.lower():
                break
            await asyncio.sleep(5)
        print("  [setup] Indexing complete.", flush=True)


async def _run_mcp_agent(task_id: str, task_prompt: str) -> RunResult:
    result = RunResult(agent="mcp_agent", task_id=task_id)

    server_params = StdioServerParameters(command=MCP_COMMAND, args=[])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await _index_if_needed(session)

            async def tool_executor(name: str, inputs: dict) -> Any:
                return await _call_mcp(session, name, inputs)

            try:
                await _run_tool_loop(MCP_TOOLS, tool_executor, task_prompt, result)
            except Exception as exc:
                result.error = str(exc)

    return result


async def _call_mcp(session: ClientSession, name: str, inputs: dict) -> Any:
    """Map agent tool names → MCP tool calls, injecting project where needed."""
    # Every MCP tool in this server requires a 'project' argument
    mcp_inputs = {"project": MCP_PROJECT, **inputs}

    tool_map = {
        "get_architecture": "get_architecture",
        "search_graph": "search_graph",
        "query_graph": "query_graph",
        "get_code_snippet": "get_code_snippet",
        "search_code": "search_code",
        "trace_call_path": "trace_call_path",
    }
    mcp_name = tool_map.get(name, name)
    try:
        result = await session.call_tool(mcp_name, mcp_inputs)
        # MCP returns a list of content items; join text blocks
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts) if parts else str(result.content)
    except Exception as exc:
        return f"MCP ERROR: {exc}"


def run_mcp_agent(task_id: str, task_prompt: str) -> RunResult:
    """Sync wrapper — creates a new event loop for each task run."""
    return asyncio.run(_run_mcp_agent(task_id, task_prompt))

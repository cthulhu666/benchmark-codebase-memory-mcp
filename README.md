# codebase-memory-mcp vs file-reading: retrieval cost benchmark

## What this is

[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) is an MCP server that builds a knowledge graph of a codebase — indexing functions, classes, call relationships, and imports — and exposes them as structured query tools (`query_graph`, `trace_path`, `search_code`, etc.).

The question this benchmark asks: **does querying the graph actually cost fewer tokens than reading files?**

A typical coding agent answers questions about a codebase by grepping and reading source files. Every tool response flows into the LLM's context window, so the total bytes returned by tool calls is the dominant variable cost of a session. If the graph index lets an agent retrieve the same information with fewer bytes, that translates directly into cheaper, faster, longer-horizon sessions.

This repository measures that difference on a real codebase ([FastAPI](https://github.com/tiangolo/fastapi), MIT) across four representative tasks. Both agents are given optimal tool sequences and verified to produce correct answers before any costs are compared.

---

## Summary

On tasks that require **traversing code relationships** — "who calls this function?", "what does this function call?" — the MCP graph index retrieves correct answers using **2–4× fewer tokens** than a file-reading agent.

On tasks that are **pure text pattern matches** where you need the verbatim source lines (e.g. "show me all instantiation sites with their arguments"), a plain grep is ~2× cheaper because `search_code` bundles graph metadata around every hit that the task doesn't need.

**Geometric mean across four tasks: MCP uses 2.0× fewer retrieval tokens.**

| Task | file_agent | mcp_agent | Winner |
|------|-----------|-----------|--------|
| T1 — callers of `jsonable_encoder` | 579 tok | 147 tok | MCP 3.9× |
| T2 — `APIRouter(...)` instantiations with kwargs | 4,323 tok | 8,182 tok | file 1.9× |
| T3 — callers of `get_dependant` | 201 tok | 64 tok | MCP 3.1× |
| T4 — direct callees of `get_dependant` | 796 tok | 303 tok | MCP 2.6× |
| **Geometric mean** | | | **MCP 2.0×** |

Token estimate: chars ÷ 4 (standard approximation for Claude models).

---

## Methodology

### What is measured

**Tool response chars** — the raw bytes returned by each tool call an agent makes to answer a task. This isolates *retrieval cost*: the one thing that actually differs between a file-reading agent and an MCP-graph agent.

Everything else — LLM output tokens, session scaffolding (system prompt, tool definitions, task prompt), and conversation-history replay — is identical for both agents on equivalent hardware and is therefore excluded. Including those would conflate retrieval cost with session overhead, which grows quadratically in multi-turn loops and has nothing to do with the index approach.

### Target codebase

[FastAPI](https://github.com/tiangolo/fastapi) (MIT licence), shallow-cloned into `target_repo/`. It is a real, widely-known Python web framework with non-trivial internal structure: cross-file call chains, a dependency-injection subsystem, and a large test suite (~580 test files). Chosen because it is recognisable, permissively licensed, and representative of mid-size Python projects.

The index was built once with `index_repository` (full mode) before any measurements were taken and was not modified during the benchmark. The MCP project slug is derived automatically from the repo path by the server.

### Agent designs

**file_agent** — has three tools: `grep_code` (regex search returning matching lines with optional context), `read_file` (return full file text), `read_function` (extract a named function's body from a file). These are the canonical tools a code-reading agent would use.

**mcp_agent** — has the full `codebase-memory-mcp` tool set: `query_graph` (Cypher against the knowledge graph), `search_code` (text search enriched with graph metadata), `trace_path` (call-chain traversal), `get_code_snippet`, `search_graph`, `get_architecture`.

### Tool sequences: pre-defined and optimal

Both agents use **pre-defined, deterministic tool sequences** rather than live LLM orchestration. There is no Anthropic API key involved; no model inference happens during the benchmark.

This choice removes three sources of noise:
1. Non-determinism in LLM tool selection
2. Suboptimal tool choices inflating one side
3. Variance across runs

The sequences are chosen to be **optimal for each agent** — the minimum set of calls needed to produce a complete correct answer. For the file agent this means using `include_function=True` on grep (so enclosing function names appear in results, not just line numbers) and `context_after=3` where multi-line constructs span several lines. For the MCP agent this means using `query_graph` for relationship lookups rather than `search_code`, which would be unnecessarily verbose.

Both sequences were verified to produce answers of **equivalent completeness** before costs were compared — see Correctness section below.

### Correctness verification

Each task has a **gold-standard assertion**: a set of strings that must appear in the combined tool response for an agent to be marked correct. Runs that fail verification are excluded from ratio calculations and flagged in the report.

Gold items were chosen to appear in both response formats (grep JSON and query_graph JSON), so the same assertion tests both agents without format-specific parsing.

| Task | Gold items checked |
|------|--------------------|
| T1 | `serialize_response`, `request_validation_exception_handler`, `get_openapi` |
| T2 | `prefix=`, `lifespan=`, `on_startup=`, `dependencies=` |
| T3 | `get_parameterless_sub_dependant`, `solve_dependencies` |
| T4 | `get_typed_signature`, `analyze_param`, `add_param_to_fields` |

All 8 runs (4 tasks × 2 agents) passed verification in the current results.

### Why geometric mean

The arithmetic total (file: 5,899 tok vs MCP: 8,696 tok) is dominated by T2 because it has the most call sites (79 matches). A single high-volume task would swamp the signal from the other three if summed directly. The geometric mean treats each task equally and is the standard aggregation for ratios.

---

## Caveats

### 1. Pre-defined sequences, not live agents

The numbers reflect *what the retrieval costs would be if both agents made optimal tool choices*. A real LLM agent will sometimes read more than necessary (e.g. loading a whole file to find one function), make redundant calls, or pick a less efficient tool. In practice, the file agent tends to overshoot more often — reading large files to extract small pieces of information — so the real-world gap likely favours MCP more than these numbers show. But this is not demonstrated here.

### 2. Graph coverage gaps in this codebase

`codebase-memory-mcp` indexes module-level functions and their CALLS relationships. For FastAPI specifically:

- **Class methods are not indexed as separate Function nodes.** `APIRouter.get`, `APIRouter.post`, etc. do not appear in the graph.
- **Closures and nested functions are not captured as CALLS edges.** FastAPI's request-handling path runs through closures returned by `get_request_handler`, which the graph cannot trace.
- **Starlette inheritance is not resolved.** `FastAPI.__call__` is inherited from Starlette; calls originating there are invisible to the graph.

Tasks were deliberately chosen to avoid these gaps. A benchmark that included tasks requiring class-method traversal or closure tracing would show MCP failing or producing wrong answers — which is not measured here.

### 3. Four tasks on one codebase

This is not statistically significant. The four tasks cover two query types (caller lookup and callee lookup) plus one text-search task, on one Python codebase. Different languages, architectures, or codebases may produce different ratios, particularly if they rely heavily on class hierarchies or dynamic dispatch that graph indexers struggle to represent.

### 4. T2: MCP overhead is structural, not inherent to the task

`search_code`'s verbosity on T2 comes from returning graph metadata (node type, in/out degree, qualified name) alongside every hit — data the task does not need. A hypothetical MCP tool that returned only the matched lines and their context would close most of the gap. The current result reflects `codebase-memory-mcp`'s actual API, not a theoretical limit.

### 5. Single run, no variance

Because both agents use deterministic tool sequences against the same files and the same index, every run produces identical output. Variance is zero by construction. This is correct for a deterministic retrieval cost measurement but means there are no confidence intervals.

### 6. chars ÷ 4 is an approximation

The 4-chars-per-token estimate is conservative for English prose and slightly generous for dense JSON. Actual token counts depend on the tokenizer and content. The relative comparisons are robust to this approximation; the absolute token numbers should be treated as indicative.

---

## Reproducing results

```bash
# Requires: Python 3.11+, codebase-memory-mcp installed at the path in the script

# Install dependencies
pip install mcp

# Run all four tasks
python fair_benchmark.py

# Run specific tasks
python fair_benchmark.py --tasks t1_callers t3_impact

# Results saved to
results/fair.json
```

The MCP project must already be indexed before running. If it is not:

```python
# In a Python session with mcp installed
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def index():
    async with stdio_client(StdioServerParameters(
        command="codebase-memory-mcp", args=[]
    )) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            await s.call_tool("index_repository", {
                "repo_path": str(Path("target_repo").resolve()),
                "mode": "full"
            })

asyncio.run(index())
```

---

## Comparison with the upstream claims

The [codebase-memory-mcp README](https://github.com/DeusData/codebase-memory-mcp) and its accompanying paper ([arXiv:2603.27277](https://arxiv.org/abs/2603.27277)) make the following claims:

| Source | Claim |
|--------|-------|
| README | **120× fewer tokens** — "5 structural queries: ~3,400 tokens vs ~412,000 via file-by-file search" |
| Paper (Table 6) | **10× fewer tokens** — ~1,000 tok/question (MCP) vs ~10,000 tok/question (explorer) |
| This benchmark | **2–4× fewer tokens** on relationship queries; file grep **1.9× cheaper** on text-pattern search; geometric mean **2.0×** |

These numbers are not contradictory — they measure different things. Here is what accounts for the gaps.

### The README's 120× is not in the paper

The paper reports 10×, not 120×. The 3,400 vs 412,000 figures do not appear in the paper's tables. They are likely a single informal measurement on a particularly large task, not a benchmark result.

### The paper's 10× vs this benchmark's 2–4×

**The file baseline is an unoptimised LLM agent.** The paper's explorer agent is a live Claude model running free with file-reading tools, making "dozens of tool calls" through iterative exploration. Such an agent commonly loads whole files to find what it needs — `routing.py` in FastAPI is 197k chars (~49k tokens) on its own. This benchmark's file agent uses `grep_code` first and reads only what is strictly necessary to answer the question. The 10× ratio is real, but it partly reflects a comparison against an unoptimised baseline rather than a best-case file reader.

**Session overhead is included in their token count.** In a multi-turn LLM session every prior message re-enters the context on each new turn. A file agent making 20 tool calls accumulates those responses quadratically across turns. This benchmark explicitly excludes session overhead and measures only tool response chars — the retrieval cost that is directly controlled by the choice of index approach. Conflating retrieval cost with conversation replay inflates the file-agent number substantially.

**MCP has lower answer quality — the README omits this.** The paper reports 83% answer quality for the MCP agent versus **92% for the file explorer**. MCP trades some accuracy for efficiency. This benchmark only compares runs where both agents produce verified correct answers, so all ratios here are accuracy-equivalent. The upstream benchmark does not apply this control.

**Task selection favours graph-native queries.** The paper's 12 question categories include hub detection, caller ranking, and dependency manifests — tasks designed around graph traversal. This benchmark's T2 (text pattern search with visible kwargs) shows the opposite direction: file grep is 1.9× cheaper when the task requires verbatim source lines rather than relationship data. A task mix weighted toward graph-native queries will always produce larger ratios in MCP's favour.

### Summary

The 120× README headline is unsupported by the paper. The paper's 10× is real but conflates retrieval cost with session overhead and measures against an unoptimised file reader. Controlling for both — optimal file agent, retrieval cost only, verified equivalent answers — the honest figure is **2–4× on relationship queries**, with file grep winning on text-pattern search. The advantage is genuine but meaningfully smaller than the headline claims, and it comes with an answer-quality trade-off the README does not surface.

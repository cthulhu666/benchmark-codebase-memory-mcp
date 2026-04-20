"""Parse BENCHMARK_RESULT JSON blocks out of subagent response strings."""
import json, re
from dataclasses import dataclass, field
from typing import Optional

MARKER = "BENCHMARK_RESULT:"

@dataclass
class AgentResult:
    task_id: str
    agent: str
    tool_log: list[dict] = field(default_factory=list)
    answer: str = ""
    parse_error: Optional[str] = None

    @property
    def total_chars(self) -> int:
        return sum(c.get("response_chars", 0) for c in self.tool_log)

    @property
    def estimated_tokens(self) -> int:
        return self.total_chars // 4

    @property
    def num_calls(self) -> int:
        return len(self.tool_log)


def parse(text: str) -> AgentResult:
    # Find the last BENCHMARK_RESULT: {...} block
    idx = text.rfind(MARKER)
    if idx == -1:
        # Try bare JSON block at end
        m = re.search(r'\{[^{}]*"task_id"[^{}]*\}', text, re.DOTALL)
        if not m:
            return AgentResult(task_id="unknown", agent="unknown",
                               parse_error=f"No marker found. Tail: {text[-300:]}")
        raw = m.group()
    else:
        raw = text[idx + len(MARKER):].strip()
        # grab until balanced brace end
        depth, end = 0, 0
        for i, ch in enumerate(raw):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        raw = raw[:end]

    try:
        d = json.loads(raw)
        return AgentResult(
            task_id=d.get("task_id", "?"),
            agent=d.get("agent", "?"),
            tool_log=d.get("tool_log", []),
            answer=d.get("answer", ""),
        )
    except Exception as e:
        return AgentResult(task_id="unknown", agent="unknown",
                           parse_error=f"JSON parse failed: {e}\nRaw: {raw[:300]}")


def print_report(results: list[AgentResult]) -> None:
    by_task: dict[str, dict[str, AgentResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, {})[r.agent] = r

    print()
    print("=" * 100)
    print("  BENCHMARK — FastAPI (MIT) — file_agent vs mcp_agent — REAL subagent runs")
    print("  Measurement: chars returned by each tool call; ~tokens = chars ÷ 4")
    print("=" * 100)
    hdr = f"  {'Task':<22} {'Agent':<12} {'Calls':>6} {'Resp chars':>12} {'~Tokens':>10}"
    print(hdr)
    print("  " + "-" * 66)

    total_file, total_mcp = 0, 0
    for task_id in sorted(by_task):
        agents = by_task[task_id]
        for ag in ("file_agent", "mcp_agent"):
            r = agents.get(ag)
            if not r:
                print(f"  {task_id:<22} {ag:<12}  (missing)")
                continue
            if r.parse_error:
                print(f"  {task_id:<22} {ag:<12}  PARSE ERROR: {r.parse_error[:60]}")
                continue
            print(f"  {task_id:<22} {ag:<12} {r.num_calls:>6} {r.total_chars:>12,} {r.estimated_tokens:>10,}")

        fa = agents.get("file_agent")
        ma = agents.get("mcp_agent")
        if fa and ma and not fa.parse_error and not ma.parse_error and ma.estimated_tokens > 0:
            ratio = fa.estimated_tokens / ma.estimated_tokens
            saved = (1 - 1/ratio) * 100
            print(f"  {'':22} {'↑ ratio':<12} {'':>6} {'':>12} {ratio:>9.1f}x  (MCP saves ~{saved:.0f}%)")
            total_file += fa.estimated_tokens
            total_mcp  += ma.estimated_tokens
        print("  " + "-" * 66)

    if total_mcp > 0:
        overall = total_file / total_mcp
        print()
        print(f"  TOTAL — file_agent ~{total_file:,} tok   mcp_agent ~{total_mcp:,} tok   ratio {overall:.1f}x")

    print()
    print("  TOOL CALL DETAIL")
    print("  " + "-" * 66)
    for task_id in sorted(by_task):
        print(f"\n  [{task_id}]")
        for ag in ("file_agent", "mcp_agent"):
            r = by_task[task_id].get(ag)
            if not r or r.parse_error:
                continue
            print(f"    {ag}:")
            for c in r.tool_log:
                summary = str(c.get("input_summary", ""))[:45]
                print(f"      {c['tool']:<30} {c['response_chars']:>9,} chars  {summary}")
    print()

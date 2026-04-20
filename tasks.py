"""
Benchmark task definitions.

Each task is a coding/analysis question a developer might ask while working
on the FastAPI codebase. They're ordered from lightweight discovery to deep
call-chain tracing so we can see how the token gap grows with task complexity.
"""

from dataclasses import dataclass


@dataclass
class Task:
    id: str
    name: str
    prompt: str
    # Rough category — used only for reporting
    kind: str  # discovery | pattern | impact | trace


TASKS: list[Task] = [
    Task(
        id="t1_discovery",
        name="Route discovery",
        kind="discovery",
        prompt=(
            "List every HTTP method decorator used in the fastapi/ package source "
            "(e.g. @router.get, @app.post, etc.). "
            "For each one, show the file and line where it appears. "
            "Be exhaustive — do not stop until you have checked all source files."
        ),
    ),
    Task(
        id="t2_pattern",
        name="APIRouter instantiation pattern",
        kind="pattern",
        prompt=(
            "Find every place in the tests/ directory where `APIRouter` is instantiated "
            "(i.e. `APIRouter(`). "
            "For each instantiation, show: the file path, the keyword arguments passed "
            "(prefix, tags, dependencies, etc.), and the variable name it's assigned to. "
            "Summarise the common patterns you observe."
        ),
    ),
    Task(
        id="t3_impact",
        name="Dependency injection impact",
        kind="impact",
        prompt=(
            "The function `get_dependant` is defined inside the fastapi package. "
            "Find every function in the fastapi/ package that calls `get_dependant` "
            "(directly or indirectly). "
            "For each caller, show the file, function name, and a one-line description "
            "of why it calls `get_dependant`."
        ),
    ),
    Task(
        id="t4_trace",
        name="Request routing call chain",
        kind="trace",
        prompt=(
            "Trace the complete call chain that executes when a client sends an HTTP "
            "request to a FastAPI application — starting from `FastAPI.__call__` (the "
            "ASGI entry point) all the way through to where the user-defined route "
            "handler function is actually invoked. "
            "List every intermediate function call in order, with file paths."
        ),
    ),
]

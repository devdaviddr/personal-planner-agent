"""Tool registry + SQLite schema + the personal-planner tools.

Each tool is a Python function with a JSON-schema-described signature. The
LLM picks which to call; `agent.run` dispatches.
"""

import sqlite3
from datetime import date

from embeddings import cosine, embed, pack, unpack

TOOLS: dict[str, dict] = {}

# Tools the agent's reflection step can safely skip — these don't mutate
# state, so the model can already see whether the result is right.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "list_tasks", "get_today", "recall",
})


def tool(name: str, description: str, schema: dict):
    def decorator(fn):
        TOOLS[name] = {"description": description, "schema": schema, "fn": fn}
        return fn
    return decorator


def tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": t["description"],
                "parameters": t["schema"],
            },
        }
        for name, t in TOOLS.items()
    ]


# Shared connection across the REPL and the tool-dispatch thread pool.
# check_same_thread=False is safe here: there's a single writer process and
# SQLite serializes commits internally.
db = sqlite3.connect("planner.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript(
    """
    CREATE TABLE IF NOT EXISTS tasks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      due_date TEXT,
      status TEXT NOT NULL DEFAULT 'open',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT,
      tool_calls TEXT,
      tool_name TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      text TEXT NOT NULL,
      embedding BLOB NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
)


# --- task tools ---------------------------------------------------------


@tool(
    "add_task",
    "Add a task to the planner.",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "due_date": {"type": "string", "description": "ISO 8601 date, e.g. 2026-05-28"},
        },
        "required": ["title"],
    },
)
def add_task(title: str, due_date: str | None = None) -> dict:
    if due_date is not None:
        date.fromisoformat(due_date)  # reject hallucinated dates loudly
    cur = db.execute(
        "INSERT INTO tasks (title, due_date) VALUES (?, ?)", (title, due_date),
    )
    db.commit()
    return {"id": cur.lastrowid, "title": title, "due_date": due_date}


@tool(
    "list_tasks",
    "List tasks. Defaults to open; pass status='done' or status='all' for others.",
    {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["open", "done", "all"], "default": "open"},
        },
    },
)
def list_tasks(status: str = "open") -> list[dict]:
    if status == "all":
        rows = db.execute(
            "SELECT id, title, due_date, status FROM tasks "
            "ORDER BY status, due_date IS NULL, due_date",
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, title, due_date, status FROM tasks WHERE status = ? "
            "ORDER BY due_date IS NULL, due_date",
            (status,),
        ).fetchall()
    return [dict(r) for r in rows]


@tool(
    "complete_task",
    "Mark a task complete.",
    {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    },
)
def complete_task(id: int) -> dict:
    cur = db.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (id,))
    db.commit()
    if cur.rowcount == 0:
        return {"error": f"no task with id={id}"}
    return {"id": id, "status": "done"}


@tool(
    "update_task",
    "Rename a task or change its due date. Pass only the fields you want changed.",
    {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "title": {"type": "string"},
            "due_date": {"type": "string", "description": "ISO 8601 date or empty string to clear"},
        },
        "required": ["id"],
    },
)
def update_task(id: int, title: str | None = None, due_date: str | None = None) -> dict:
    sets, params = [], []
    if title is not None:
        sets.append("title = ?")
        params.append(title)
    if due_date is not None:
        if due_date != "":
            date.fromisoformat(due_date)
        sets.append("due_date = ?")
        params.append(due_date or None)
    if not sets:
        return {"error": "nothing to update"}
    params.append(id)
    cur = db.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", params)
    db.commit()
    if cur.rowcount == 0:
        return {"error": f"no task with id={id}"}
    return {"id": id, "updated": True}


@tool(
    "delete_task",
    "Delete a task permanently.",
    {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    },
)
def delete_task(id: int) -> dict:
    cur = db.execute("DELETE FROM tasks WHERE id = ?", (id,))
    db.commit()
    if cur.rowcount == 0:
        return {"error": f"no task with id={id}"}
    return {"id": id, "deleted": True}


@tool("get_today", "Get today's date in ISO 8601.", {"type": "object", "properties": {}})
def get_today() -> dict:
    return {"date": date.today().isoformat()}


# --- long-term memory tools --------------------------------------------


# Cache of (text, vec) for all memories. Loaded lazily on first recall and
# appended to on remember. Lets recall skip the SQLite scan + blob unpack on
# every call, which dominates latency once memories grow past a few dozen.
_memory_cache: list[tuple[str, list[float]]] | None = None


def _load_memory_cache() -> list[tuple[str, list[float]]]:
    global _memory_cache
    if _memory_cache is None:
        rows = db.execute("SELECT text, embedding FROM memories").fetchall()
        _memory_cache = [(r["text"], unpack(r["embedding"])) for r in rows]
    return _memory_cache


@tool(
    "remember",
    "Store a durable fact about the user.",
    {
        "type": "object",
        "properties": {"fact": {"type": "string"}},
        "required": ["fact"],
    },
)
def remember(fact: str) -> dict:
    vec = embed(fact)
    db.execute(
        "INSERT INTO memories (text, embedding) VALUES (?, ?)", (fact, pack(vec)),
    )
    db.commit()
    if _memory_cache is not None:
        _memory_cache.append((fact, vec))
    return {"ok": True, "fact": fact}


@tool(
    "recall",
    "Search long-term memory by meaning.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    },
)
def recall(query: str, k: int = 3) -> list[dict]:
    qv = embed(query)
    scored = sorted(
        ((cosine(qv, vec), text) for text, vec in _load_memory_cache()),
        reverse=True,
    )
    return [{"text": t, "score": round(s, 3)} for s, t in scored[:k]]

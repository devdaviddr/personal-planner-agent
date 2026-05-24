"""Tool registry + SQLite schema + the personal-planner tools.

Each tool is a Python function with a JSON-schema-described signature. The
LLM picks which to call; `agent.run` dispatches.
"""

import sqlite3
from datetime import date

from embeddings import cosine, embed, pack, unpack

TOOLS: dict[str, dict] = {}


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


# Shared connection — single-threaded REPL, so the default is fine.
db = sqlite3.connect("planner.db")
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
    cur = db.execute(
        "INSERT INTO tasks (title, due_date) VALUES (?, ?)", (title, due_date),
    )
    db.commit()
    return {"id": cur.lastrowid, "title": title, "due_date": due_date}


@tool("list_tasks", "List open tasks.", {"type": "object", "properties": {}})
def list_tasks() -> list[dict]:
    rows = db.execute(
        "SELECT id, title, due_date FROM tasks WHERE status = 'open' "
        "ORDER BY due_date IS NULL, due_date",  # push null-due tasks to the bottom
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
    db.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (id,))
    db.commit()
    return {"id": id, "status": "done"}


@tool("get_today", "Get today's date in ISO 8601.", {"type": "object", "properties": {}})
def get_today() -> dict:
    return {"date": date.today().isoformat()}


# --- long-term memory tools --------------------------------------------


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
    rows = db.execute("SELECT text, embedding FROM memories").fetchall()
    scored = sorted(
        ((cosine(qv, unpack(r["embedding"])), r["text"]) for r in rows),
        reverse=True,
    )
    return [{"text": t, "score": round(s, 3)} for s, t in scored[:k]]

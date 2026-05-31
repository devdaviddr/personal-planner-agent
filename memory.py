"""Short-term memory: conversation buffer persisted per session in SQLite.

`save_message` writes one row per turn. `load_history` reads them back, with
trim_to_user_boundary keeping the history valid for tool-calling APIs.
"""

import json
import sqlite3

HISTORY_LIMIT = 20  # rough turn budget — bump for longer agents


def save_message(db: sqlite3.Connection, session: str, msg: dict) -> None:
    db.execute(
        "INSERT INTO messages (session, role, content, tool_calls, tool_name) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            session,
            msg["role"],
            msg.get("content"),
            json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None,
            msg.get("tool_name"),
        ),
    )
    db.commit()


def load_history(db: sqlite3.Connection, session: str) -> list[dict]:
    rows = db.execute(
        "SELECT role, content, tool_calls, tool_name FROM messages "
        "WHERE session = ? ORDER BY id DESC LIMIT ?",
        (session, HISTORY_LIMIT),
    ).fetchall()
    msgs = []
    for r in reversed(rows):
        m = {"role": r["role"]}
        if r["content"] is not None:
            m["content"] = r["content"]
        if r["tool_calls"]:
            m["tool_calls"] = json.loads(r["tool_calls"])
        if r["tool_name"]:
            m["tool_name"] = r["tool_name"]
        msgs.append(m)
    return trim_to_user_boundary(msgs)


def trim_to_user_boundary(msgs: list[dict]) -> list[dict]:
    # Tool-calling APIs require an assistant message with tool_calls to be
    # immediately followed by role: tool messages for each call. After a
    # window slice we have to guard both ends:
    #   (1) start on a user message — drop any orphan tool/assistant prefix
    #   (2) drop trailing orphans: a role:tool with no live assistant before
    #       it, or an assistant whose tool_calls were never answered.
    start = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if start is None:
        return []
    msgs = msgs[start:]
    while msgs and (
        msgs[-1].get("role") == "tool"
        or (msgs[-1].get("role") == "assistant" and msgs[-1].get("tool_calls"))
    ):
        msgs.pop()
    return msgs

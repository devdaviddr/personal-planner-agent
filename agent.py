"""Personal-planner agent: plan -> act -> reflect.

Reads stdin, dispatches tool calls against Ollama, persists everything to
SQLite (tasks, conversation, long-term memories). Run with `python agent.py`.
"""

import json

import ollama

from memory import load_history, save_message
from tools import TOOLS, db, tool_specs

MODEL = "qwen3.5:9b"
MAX_TURNS = 8
MAX_REFLECTIONS = 2

SYSTEM = {
    "role": "system",
    "content": (
        "You are a personal planner.\n"
        "- For greetings or trivial chit-chat, reply directly in one short sentence. Do NOT plan, do NOT call tools.\n"
        "- Use get_today before reasoning about relative dates (tomorrow, next week, etc).\n"
        "- For multi-step requests only, write a short plan first (1–3 bullets), then call the tools to execute it.\n"
        "- If the user states a durable preference or fact about themselves, call remember.\n"
        "- If a question would benefit from past context, call recall before answering.\n"
        "- Call tools through the structured tool-call interface only. Never write tool calls as JSON in your reply text.\n"
        "- After executing, summarize what you did in one line."
    ),
}

REFLECT_PROMPT = (
    "You are reviewing an agent transcript. Given the user's original request "
    "and the actions taken, answer in JSON: "
    '{"done": true|false, "critique": "..."}. '
    "Set done=true if the request was fully satisfied. "
    "Set done=false and provide a concrete critique if anything is missing or wrong."
)


def to_dict(msg) -> dict:
    # Ollama returns Pydantic models. Convert to plain dicts so json.dumps
    # (and later, SQLite storage) work without a custom encoder.
    return msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg


def _act(messages: list[dict]) -> tuple[str, list[dict]]:
    """One pass of the tool-calling loop. We both *mutate* `messages` (so the
    caller and reflection see the full transcript) and *return* the slice
    of new turns this call added, so `run` can decide what to persist."""
    start = len(messages)
    for _ in range(MAX_TURNS):
        res = ollama.chat(model=MODEL, messages=messages, tools=tool_specs())
        msg = to_dict(res["message"])
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", ""), messages[start:]
        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"] or {}
            if isinstance(args, str):  # some models return arguments as JSON text
                args = json.loads(args)
            try:
                result = TOOLS[name]["fn"](**args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool", "content": json.dumps(result), "tool_name": name,
            })
    return "I hit my tool-call limit.", messages[start:]


def _reflect(original: str, reply: str, messages: list[dict]) -> dict:
    transcript = "\n".join(
        f"{m['role']}: {m.get('content', '') or m.get('tool_calls', '')}"
        for m in messages[-8:]
    )
    res = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": REFLECT_PROMPT},
            {"role": "user", "content":
                f"Original request: {original}\n\nTranscript:\n{transcript}\n\nFinal reply: {reply}"},
        ],
        format="json",
    )
    content = to_dict(res["message"]).get("content") or "{}"
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"done": True, "critique": ""}  # fail open on malformed output


def run(session: str, user_input: str) -> str:
    save_message(db, session, {"role": "user", "content": user_input})
    original = user_input
    messages = [SYSTEM, *load_history(db, session)]
    reply, added = "", []
    succeeded = False

    for attempt in range(MAX_REFLECTIONS + 1):
        reply, added = _act(messages)
        try:
            verdict = _reflect(original, reply, messages)
        except Exception:
            verdict = {"done": True, "critique": ""}  # fail open on any error
        if verdict["done"]:
            succeeded = True
            break
        if attempt == MAX_REFLECTIONS:
            break
        # Critique is appended in-memory only — never persisted.
        messages.append({
            "role": "user",
            "content": f"Your previous attempt was incomplete: {verdict['critique']}",
        })

    # Persist only the winning attempt's assistant/tool turns. On exhaustion
    # without success, the original user message remains as a dangling row;
    # next turn's load + trim handles that gracefully.
    if succeeded:
        for m in added:
            save_message(db, session, m)
    return reply


if __name__ == "__main__":
    import uuid

    session = uuid.uuid4().hex  # one session per process; pin a stable id to span runs
    print(f"session: {session}\n")
    while (text := input("you> ").strip()):
        print(f"bot> {run(session, text)}\n")

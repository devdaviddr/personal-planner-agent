"""Personal-planner agent: plan -> act -> reflect.

Reads stdin, dispatches tool calls against Ollama, persists everything to
SQLite (tasks, conversation, long-term memories). Run with `python agent.py`.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import ollama

from memory import load_history, save_message
from tools import READ_ONLY_TOOLS, TOOLS, db, tool_specs

logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request noise
log = logging.getLogger("agent")

MODEL = "qwen3.5:9b"
REFLECT_MODEL = "qwen3.5:4b"  # smaller critic — reflection is yes/no, not generation
MAX_TURNS = 4
MAX_REFLECTIONS = 2

# Keep model resident across calls so KV cache survives between turns.
KEEP_ALIVE = "24h"
CHAT_OPTIONS = {"num_ctx": 4096, "num_predict": 512}
REFLECT_OPTIONS = {"num_ctx": 2048, "num_predict": 128}

_tool_pool = ThreadPoolExecutor(max_workers=4)

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


def _dispatch(call: dict) -> dict:
    name = call["function"]["name"]
    args = call["function"]["arguments"] or {}
    if isinstance(args, str):  # some models return arguments as JSON text
        args = json.loads(args)
    try:
        result = TOOLS[name]["fn"](**args)
    except Exception as e:
        log.exception("tool %s failed with args=%r", name, args)
        result = {"error": str(e)}
    return {"role": "tool", "content": json.dumps(result), "tool_name": name}


def _act(messages: list[dict]) -> tuple[str, list[dict]]:
    """One pass of the tool-calling loop. We both *mutate* `messages` (so the
    caller and reflection see the full transcript) and *return* the slice
    of new turns this call added, so `run` can decide what to persist."""
    start = len(messages)
    for _ in range(MAX_TURNS):
        res = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=tool_specs(),
            options=CHAT_OPTIONS,
            keep_alive=KEEP_ALIVE,
        )
        msg = to_dict(res["message"])
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", ""), messages[start:]
        # Dispatch independent tool calls concurrently; map preserves order
        # so each tool message lines up with its originating tool_call.
        for tool_msg in _tool_pool.map(_dispatch, calls):
            messages.append(tool_msg)
    return "I hit my tool-call limit.", messages[start:]


def _reflect(original: str, reply: str, messages: list[dict]) -> dict:
    transcript = "\n".join(
        f"{m['role']}: {m.get('content', '') or m.get('tool_calls', '')}"
        for m in messages[-8:]
    )
    res = ollama.chat(
        model=REFLECT_MODEL,
        messages=[
            {"role": "system", "content": REFLECT_PROMPT},
            {"role": "user", "content":
                f"Original request: {original}\n\nTranscript:\n{transcript}\n\nFinal reply: {reply}"},
        ],
        format="json",
        options=REFLECT_OPTIONS,
        keep_alive=KEEP_ALIVE,
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

    for attempt in range(MAX_REFLECTIONS + 1):
        reply, added = _act(messages)
        # Skip reflection when nothing observable happened: no tools at all
        # (chit-chat), or only read-only lookups whose correctness the model
        # can already see in the transcript. Reflection only earns its cost
        # when a mutation could be wrong or incomplete.
        tool_names = {m["tool_name"] for m in added if m.get("role") == "tool"}
        if not tool_names or tool_names <= READ_ONLY_TOOLS:
            break
        try:
            verdict = _reflect(original, reply, messages)
        except Exception:
            log.exception("reflect failed")
            verdict = {"done": True, "critique": ""}  # fail open on any error
        if verdict["done"]:
            break
        if attempt == MAX_REFLECTIONS:
            break
        # Critique is appended in-memory only — never persisted.
        messages.append({
            "role": "user",
            "content": f"Your previous attempt was incomplete: {verdict['critique']}",
        })

    # Persist the final attempt's turns whether or not reflection passed —
    # the user already saw `reply`, so the DB must match. Without this, a
    # failed reflection leaves the user message dangling with no assistant
    # follow-up, producing two consecutive user messages next turn.
    for m in added:
        save_message(db, session, m)
    return reply


if __name__ == "__main__":
    import uuid

    session = uuid.uuid4().hex  # one session per process; pin a stable id to span runs
    print(f"session: {session}\n")
    while (text := input("you> ").strip()):
        print(f"bot> {run(session, text)}\n")

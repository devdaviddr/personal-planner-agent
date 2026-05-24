# personal-planner-agent

A local-only personal-planner AI agent in ~300 lines of plain Python. No agent frameworks. Backed by a local [Ollama](https://ollama.com) model and a single SQLite file.

Companion code for the tutorial **[Build an AI Agent from Scratch with Ollama and Python](https://www.danielruffolo.net)**, which walks through every line and explains the fundamentals (tools, short-term memory, long-term memory, planning, reflection) one layer at a time.

## What it does

Talk to the agent from your terminal. It can:

- Add, list, and complete tasks (stored in `tasks` table).
- Remember durable facts about you across sessions (long-term memory via embeddings + cosine similarity over SQLite).
- Carry conversation context within a session (short-term memory, with boundary-safe trim).
- Plan before acting on multi-step requests.
- Reflect on its own work and retry if a critique finds it incomplete.

Everything runs locally. Nothing leaves your machine.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    you (terminal REPL)                   │
└────────────────────────────┬─────────────────────────────┘
                             │  user text ⇅ reply
                             ▼
┌──────────────────────────────────────────────────────────┐
│                         agent.py                         │
│                                                          │
│     ┌──────┐    ┌──────┐    ┌─────────┐                  │
│     │ plan │ ──▶│ act  │ ──▶│ reflect │ ── done ──▶      │
│     └──────┘    └──┬───┘    └────┬────┘                  │
│        ▲           │             │                       │
│        └─── critique ◀───────────┘                       │
└─────┬────────────────────────────────────┬───────────────┘
      │ chat (tools advertised)            │ dispatch
      ▼                                    ▼
┌──────────────┐                  ┌────────────────────┐
│    Ollama    │                  │      tools.py      │
│  qwen3:8b    │                  │                    │
│  nomic-embed │◀── embeddings ───│  task + memory     │
│       -text  │                  │  tools             │
└──────────────┘                  └─────────┬──────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │       planner.db        │
                              │                         │
                              │  tasks · messages ·     │
                              │  memories (+ vec BLOB)  │
                              └─────────────────────────┘
```

## Files

```
agent.py         entry point · REPL · plan → act → reflect loop
tools.py         tool registry + task / memory tools + SQLite schema
memory.py        save_message, load_history, trim_to_user_boundary
embeddings.py    embed(), cosine(), pack/unpack for SQLite BLOBs
requirements.txt single dep: ollama>=0.4
planner.db       SQLite — created on first run
```

## Setup

```sh
# 1. Install Ollama and pull the two models
brew install ollama       # or follow https://ollama.com
ollama serve &
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 2. Install the Python client
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run
python agent.py
```

## Example session

```
you> i have wednesday afternoons free for meetings
bot> Noted.

you> add a task to buy groceries tomorrow
bot> Added "buy groceries" due 2026-05-25.

you> when should i schedule the dentist?
bot> Plan:
     - recall any free-time preferences
     - resolve "next Wednesday" via get_today
     - propose a date
     You mentioned Wednesday afternoons are free. Next Wednesday is
     2026-05-27 — want me to add it?

you> yes
bot> Added "dentist" due 2026-05-27.

you> what's on my list?
bot> 2 open tasks:
     1. buy groceries — 2026-05-25
     2. dentist — 2026-05-27
```

## Tunable knobs

All in `agent.py`:

| Constant | Default | What it controls |
|---|---|---|
| `MODEL` | `qwen3:8b` | Any tool-calling-capable Ollama model. `llama3.2:3b` is smaller but trips on tool schemas more often. |
| `MAX_TURNS` | `8` | Max tool-call iterations per user message. |
| `MAX_REFLECTIONS` | `2` | How many self-critique retries before giving up. |
| `HISTORY_LIMIT` | `40` (in `memory.py`) | Rough turn budget loaded per call. |

## Extending it

- **New tools** — add a `@tool(...)` decorated function to `tools.py`. The registry is auto-picked-up.
- **Different domain** — replace the task tools with calls to GitHub Issues, Linear, your calendar, or any REST API. Everything else (memory, reflection, agent loop) stays as-is.
- **Bigger memory** — past ~10k memories the in-Python cosine loop is too slow. Swap `recall()` for a proper vector index (Chroma, LanceDB, pgvector).

## License

MIT.

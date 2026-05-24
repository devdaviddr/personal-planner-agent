"""Embeddings + raw cosine similarity in pure Python.

Long-term memory uses these to do meaning-based retrieval over the `memories`
table. SQLite has no native vector type, so vectors are packed into a BLOB
column as explicit little-endian float32 (portable across machines).
"""

import struct

import ollama

EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    res = ollama.embed(model=EMBED_MODEL, input=text)
    return res["embeddings"][0]


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)  # explicit little-endian float32


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

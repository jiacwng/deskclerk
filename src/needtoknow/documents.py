# Documents and their readers: each document is split into chunks of at most 800 characters,
# the chunks are embedded, and the document, its access rows and its chunks are written in one
# transaction.

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg

from needtoknow.embed import embed_passages

CHUNK_CHARS = 800
# A paragraph too long for one chunk is split at line breaks, then sentences, then spaces, as
# LangChain's RecursiveCharacterTextSplitter does; a run with no space left is cut.
_SPLITS = ((r"\n", "\n"), (r"(?<=[.!?])\s+", " "), (r"\s+", " "))


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str
    readers: frozenset[str]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    position: int
    text: str


def split_into_chunks(document: Document) -> list[Chunk]:
    limit = max(CHUNK_CHARS - len(document.title) - 2, CHUNK_CHARS // 2)
    groups: list[list[str]] = []
    for paragraph in _paragraphs(document.body):
        for piece in _fit(paragraph, limit):
            if groups and len(_chunk_text(document.title, [*groups[-1], piece])) <= CHUNK_CHARS:
                groups[-1].append(piece)
            else:
                groups.append([piece])
    return [
        Chunk(doc_id=document.id, position=position, text=_chunk_text(document.title, group))
        for position, group in enumerate(groups)
    ]


def write_documents(owner: psycopg.Connection[Any], documents: Sequence[Document]) -> int:
    chunks = [chunk for document in documents for chunk in split_into_chunks(document)]
    # Embedding takes seconds, so it runs before the transaction opens and holds no locks.
    embeddings = embed_passages([chunk.text for chunk in chunks])
    with owner.transaction(), owner.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO documents (id, title, body) VALUES (%s, %s, %s)",
            [(document.id, document.title, document.body) for document in documents],
        )
        cursor.executemany(
            "INSERT INTO doc_access (doc_id, principal) VALUES (%s, %s)",
            [(document.id, reader) for document in documents for reader in document.readers],
        )
        cursor.executemany(
            "INSERT INTO chunks (doc_id, position, text, embedding) VALUES (%s, %s, %s, %s)",
            [
                (chunk.doc_id, chunk.position, chunk.text, embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
    return len(chunks)


def _paragraphs(body: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", body) if paragraph.strip()]


def _fit(text: str, limit: int, level: int = 0) -> list[str]:
    if len(text) <= limit:
        return [text]
    if level == len(_SPLITS):
        return [text[start : start + limit] for start in range(0, len(text), limit)]
    pattern, joiner = _SPLITS[level]
    pieces: list[str] = []
    for part in re.split(pattern, text):
        if not part.strip():
            continue
        for piece in _fit(part, limit, level + 1):
            if pieces and len(pieces[-1]) + len(joiner) + len(piece) <= limit:
                pieces[-1] += joiner + piece
            else:
                pieces.append(piece)
    return pieces


def _chunk_text(title: str, group: list[str]) -> str:
    return "\n\n".join([title, *group] if title else group)

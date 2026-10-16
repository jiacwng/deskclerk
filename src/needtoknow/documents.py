# Documents and their readers: each document is split into chunks of at most 800 characters,
# the chunks are embedded, and the document, its access rows and its chunks are written in one
# transaction.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from langchain_text_splitters import RecursiveCharacterTextSplitter

from needtoknow.embed import embed_passages

CHUNK_CHARS = 800
# A body is split at blank lines, then line breaks, then sentence ends, then spaces; a run with
# no space left is cut.
_SEPARATORS = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+", ""]


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
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS, is_separator_regex=True, chunk_size=limit, chunk_overlap=0
    )
    chunks = []
    for position, body in enumerate(splitter.split_text(document.body)):
        text = f"{document.title}\n\n{body}" if document.title else body
        chunks.append(Chunk(doc_id=document.id, position=position, text=text))
    return chunks


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

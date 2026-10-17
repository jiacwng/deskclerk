# Loads the Enron mailboxes of the CMU release, maildir/<custodian>/<folder>/<file>, as documents
# that only the custodians whose mailbox holds a copy may read. The release gives every file its
# own Message-ID, even copies of one email in several folders or mailboxes, so an email is known
# by a hash of its date, sender, recipients, subject and body.

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from email.parser import HeaderParser
from pathlib import Path
from typing import Any

import psycopg

from needtoknow.documents import CHUNK_CHARS, Document, write_documents

ID_PREFIX = "enron-"
BATCH = 200
KEY_HEADERS = ("Date", "From", "To", "Cc", "Subject")
# A file longer than a hundred chunks is skipped, so one email cannot hold up a batch.
MAX_FILE_BYTES = CHUNK_CHARS * 100
TITLE_CHARS = CHUNK_CHARS // 8

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Email:
    id: str
    subject: str
    body: str


@dataclass(frozen=True)
class Loaded:
    files: int
    skipped: int
    emails: int
    chunks: int


def parse_email(data: bytes) -> Email:
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\x00", "")
    head, _, raw_body = text.partition("\n\n")
    headers = HeaderParser().parsestr(head)
    fields = [" ".join(str(headers.get(name, "")).split()) for name in KEY_HEADERS]
    body = "\n".join(line.rstrip() for line in raw_body.strip().splitlines())
    digest = hashlib.sha256("\x1f".join([*fields, body]).encode()).hexdigest()
    return Email(id=ID_PREFIX + digest, subject=fields[-1][:TITLE_CHARS], body=body)


# Files come in path order, so --limit always reads the same ones. Links are skipped: the archive
# is untrusted and a link could point outside it.
def mail_files(maildir: Path, limit: int | None = None) -> Iterator[tuple[str, Path]]:
    files = 0
    for mailbox in sorted(maildir.iterdir()):
        if not mailbox.is_dir() or mailbox.is_symlink():
            continue
        for path in sorted(mailbox.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if files == limit:
                return
            files += 1
            yield mailbox.name, path


# Emails already in the database are not embedded again, so a load that stops can be run again
# and carries on; a copy found later only adds its custodian as a reader.
def load_enron(owner: psycopg.Connection[Any], maildir: Path, limit: int | None = None) -> Loaded:
    known = {row[0] for row in owner.execute("SELECT id FROM documents WHERE id LIKE 'enron-%'")}
    pending: dict[str, tuple[Email, set[str]]] = {}
    readers: list[tuple[str, str]] = []
    files = skipped = emails = chunks = 0
    for custodian, path in mail_files(maildir, limit):
        files += 1
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped += 1
            continue
        email = parse_email(path.read_bytes())
        reader = f"user:{custodian}"
        if email.id in pending:
            pending[email.id][1].add(reader)
        elif email.id in known:
            readers.append((email.id, reader))
        else:
            pending[email.id] = (email, {reader})
        if len(pending) == BATCH or len(readers) == BATCH:
            chunks += _write(owner, pending, readers)
            emails += len(pending)
            known.update(pending)
            pending.clear()
            readers.clear()
            _log.info("%d files read, %d emails and %d chunks written", files, emails, chunks)
    chunks += _write(owner, pending, readers)
    emails += len(pending)
    return Loaded(files=files, skipped=skipped, emails=emails, chunks=chunks)


def _write(
    owner: psycopg.Connection[Any],
    pending: dict[str, tuple[Email, set[str]]],
    readers: list[tuple[str, str]],
) -> int:
    documents = [
        Document(id=email.id, title=email.subject, body=email.body, readers=frozenset(custodians))
        for email, custodians in pending.values()
    ]
    chunks = write_documents(owner, documents) if documents else 0
    with owner.transaction(), owner.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO doc_access (doc_id, principal) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            readers,
        )
    return chunks

# One way to open a SQLite file: write-ahead log and a long wait for a locked database.

import sqlite3
from pathlib import Path

BUSY_TIMEOUT_SECONDS = 30


def connect(path: Path) -> sqlite3.Connection:
    database = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
    database.execute("PRAGMA journal_mode=WAL")
    return database

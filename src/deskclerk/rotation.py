# Re-encrypts every stored value with the newest token key: the secrets table, key_check and the
# encrypted checkpoint blobs and writes, one transaction per table.

from dataclasses import dataclass

from cryptography.fernet import MultiFernet

from deskclerk.database import Database

# table, select of the key columns then the ciphertext, update taking the ciphertext then the keys
TABLES = (
    (
        "secrets",
        "SELECT user_id, server_name, kind, value FROM secrets FOR UPDATE",
        "UPDATE secrets SET value = %s WHERE user_id = %s AND server_name = %s AND kind = %s",
    ),
    (
        "key_check",
        "SELECT only_row, value FROM key_check FOR UPDATE",
        "UPDATE key_check SET value = %s WHERE only_row = %s",
    ),
    (
        "checkpoint_blobs",
        "SELECT thread_id, checkpoint_ns, channel, version, blob FROM checkpoint_blobs "
        "WHERE type LIKE '%+fernet' FOR UPDATE",
        "UPDATE checkpoint_blobs SET blob = %s WHERE thread_id = %s AND checkpoint_ns = %s "
        "AND channel = %s AND version = %s",
    ),
    (
        "checkpoint_writes",
        "SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, blob "
        "FROM checkpoint_writes WHERE type LIKE '%+fernet' FOR UPDATE",
        "UPDATE checkpoint_writes SET blob = %s WHERE thread_id = %s AND checkpoint_ns = %s "
        "AND checkpoint_id = %s AND task_id = %s AND idx = %s",
    ),
)


@dataclass(frozen=True)
class Rotated:
    table: str
    rows: int


def rotate_key(database: Database, fernet: MultiFernet) -> list[Rotated]:
    return [
        Rotated(table, _rotate_table(database, fernet, table, select, update))
        for table, select, update in TABLES
    ]


def _rotate_table(
    database: Database, fernet: MultiFernet, table: str, select: str, update: str
) -> int:
    with database.pool.connection() as connection:
        found = connection.execute("SELECT to_regclass(%s)", (table,)).fetchone()
        if found is None or found[0] is None:
            return 0
        rows = connection.execute(select).fetchall()
        for row in rows:
            connection.execute(update, (fernet.rotate(bytes(row[-1])), *row[:-1]))
    return len(rows)

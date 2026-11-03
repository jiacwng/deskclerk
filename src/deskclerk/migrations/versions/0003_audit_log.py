# The audit log table and its hash chain. Each row holds the line as printed, the columns search
# uses, and a hash over the previous row's hash and the line. The checkpoint row holds the hash the
# oldest kept row chains from: all zeros at first, the last deleted row's hash after retention.

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

STATEMENTS = (
    "CREATE TABLE audit_log ("
    "id BIGSERIAL PRIMARY KEY, time TIMESTAMPTZ NOT NULL, line TEXT NOT NULL, "
    "user_id TEXT NOT NULL, entry TEXT NOT NULL, servers TEXT[] NOT NULL, tools TEXT[] NOT NULL, "
    "decisions TEXT[] NOT NULL, hash TEXT NOT NULL)",
    "CREATE INDEX audit_log_time ON audit_log (time)",
    "CREATE INDEX audit_log_user ON audit_log (user_id, id)",
    "CREATE INDEX audit_log_entry ON audit_log (entry, id)",
    "CREATE INDEX audit_log_servers ON audit_log USING GIN (servers)",
    "CREATE INDEX audit_log_tools ON audit_log USING GIN (tools)",
    "CREATE INDEX audit_log_decisions ON audit_log USING GIN (decisions)",
    "CREATE TABLE audit_checkpoint ("
    "only_row BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_row), hash TEXT NOT NULL)",
    "INSERT INTO audit_checkpoint (hash) VALUES (repeat('0', 64))",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)

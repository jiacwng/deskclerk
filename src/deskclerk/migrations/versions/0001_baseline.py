# The tables and indexes deskclerk keeps its own state in. The conversation checkpoints belong to
# LangGraph, which creates its tables itself. IF NOT EXISTS lets a database that an earlier
# release created by hand take this revision without error.

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES = [
    "CREATE TABLE IF NOT EXISTS oauth_flows ("
    "state TEXT PRIMARY KEY, user_id TEXT NOT NULL, server_name TEXT NOT NULL, "
    "deadline DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS oauth_callbacks ("
    "state TEXT PRIMARY KEY, code TEXT, iss TEXT, error TEXT, "
    "created_at DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS conversations ("
    "user_id TEXT NOT NULL, id UUID NOT NULL, title TEXT NOT NULL, "
    "updated_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (user_id, id))",
    "CREATE TABLE IF NOT EXISTS quota_hits ("
    "user_id TEXT NOT NULL, hit_at DOUBLE PRECISION NOT NULL)",
    "CREATE INDEX IF NOT EXISTS quota_hits_user ON quota_hits (user_id, hit_at)",
    "CREATE TABLE IF NOT EXISTS scim_users ("
    "position BIGSERIAL, id TEXT PRIMARY KEY, external_id TEXT NOT NULL UNIQUE, "
    "user_name TEXT NOT NULL, active BOOLEAN NOT NULL)",
    "CREATE UNIQUE INDEX IF NOT EXISTS scim_users_user_name ON scim_users (lower(user_name))",
    "CREATE TABLE IF NOT EXISTS sessions ("
    "id_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_id TEXT NOT NULL, "
    "name TEXT NOT NULL, sid TEXT, subject TEXT, groups TEXT NOT NULL, "
    "expires_at DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS costs ("
    "user_id TEXT NOT NULL, month TEXT NOT NULL, source TEXT NOT NULL, "
    "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
    "cache_read_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL, "
    "usd DOUBLE PRECISION NOT NULL)",
    "CREATE INDEX IF NOT EXISTS costs_user_month ON costs (user_id, month)",
    "CREATE TABLE IF NOT EXISTS secrets ("
    "user_id TEXT NOT NULL, server_name TEXT NOT NULL, kind TEXT NOT NULL, "
    "value BYTEA NOT NULL, PRIMARY KEY (user_id, server_name, kind))",
    "CREATE TABLE IF NOT EXISTS key_check ("
    "only_row BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_row), value BYTEA NOT NULL)",
]


def upgrade() -> None:
    for statement in TABLES:
        op.execute(statement)

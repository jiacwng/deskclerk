# The table of MCP servers the admin sets up, one row per server with every field of a server
# entry. Secrets are references: client_secret_env holds a variable name and headers keep their
# ${env:NAME} text.

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

SERVERS = (
    "CREATE TABLE servers ("
    "position BIGSERIAL, name TEXT PRIMARY KEY, plain_http BOOLEAN NOT NULL, "
    "url TEXT NOT NULL, client_id TEXT, client_secret_env TEXT, auth TEXT NOT NULL, "
    "tools TEXT[], headers JSONB NOT NULL, client_cert TEXT, client_key TEXT, groups TEXT[], "
    "always_allow BOOLEAN NOT NULL, confirm_tools TEXT[] NOT NULL, trusted BOOLEAN NOT NULL, "
    "title TEXT, description TEXT, examples TEXT[] NOT NULL, paused BOOLEAN NOT NULL)"
)


def upgrade() -> None:
    op.execute(SERVERS)

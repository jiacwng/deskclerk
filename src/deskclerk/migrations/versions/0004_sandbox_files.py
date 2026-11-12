# The table of files that code run in the sandbox left in /mnt/data, kept per employee and
# conversation so the next run can start with them and the employee can download them.

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE sandbox_files ("
        "id UUID PRIMARY KEY, user_id TEXT NOT NULL, conversation_id UUID NOT NULL, "
        "name TEXT NOT NULL, content BYTEA NOT NULL, size INTEGER NOT NULL, "
        "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
        "UNIQUE (user_id, conversation_id, name))"
    )

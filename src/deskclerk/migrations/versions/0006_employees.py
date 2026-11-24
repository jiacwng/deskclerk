# The employees table: each employee's display name, for the admin page.

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

STATEMENT = (
    "CREATE TABLE employees ("
    "user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, last_seen TIMESTAMPTZ NOT NULL)"
)


def upgrade() -> None:
    op.execute(STATEMENT)

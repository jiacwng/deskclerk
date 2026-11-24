# The employees table, one row per employee with the display name from their token and when they
# last signed in or first appeared in this process, so the admin page can show a name beside an id.

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

# The routines table: a request an employee saved with a weekly schedule. days holds weekdays with
# Monday 0 to Sunday 6, next_run_at is the next occurrence in the routine's time zone, and the last
# run's outcome and conversation are kept for the page. The sign-in refresh token lives in secrets.

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

STATEMENTS = (
    "CREATE TABLE routines ("
    "id UUID PRIMARY KEY, user_id TEXT NOT NULL, request TEXT NOT NULL, "
    "days SMALLINT[] NOT NULL, time_of_day TIME NOT NULL, time_zone TEXT NOT NULL, "
    "next_run_at TIMESTAMPTZ NOT NULL, paused BOOLEAN NOT NULL DEFAULT FALSE, "
    "last_run_at TIMESTAMPTZ, last_outcome TEXT, last_conversation UUID)",
    "CREATE INDEX routines_due ON routines (next_run_at) WHERE NOT paused",
    "CREATE INDEX routines_user ON routines (user_id)",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)

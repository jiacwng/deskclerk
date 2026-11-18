# A flagged column on the audit log, true when any tool result in the line was flagged by the
# injection screen. It is filled from the stored JSON line, where flagged is a count on an ask or
# confirm line and a boolean on a gateway call line.

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

STATEMENTS = (
    "ALTER TABLE audit_log ADD COLUMN flagged BOOLEAN NOT NULL DEFAULT FALSE",
    "UPDATE audit_log SET flagged = TRUE WHERE line::jsonb -> 'flagged' NOT IN "
    "('false'::jsonb, '0'::jsonb, 'null'::jsonb)",
    "CREATE INDEX audit_log_flagged ON audit_log (id) WHERE flagged",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)

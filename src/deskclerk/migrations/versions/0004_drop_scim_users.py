# Drops the table of provisioned users that nothing reads or writes any more.

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS scim_users")

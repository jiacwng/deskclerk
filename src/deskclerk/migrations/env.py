# Alembic environment: runs the migrations on the database that Database.migrate hands over.

from alembic import context
from sqlalchemy import create_engine, pool

engine = create_engine(context.config.attributes["url"], poolclass=pool.NullPool)
with engine.connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()

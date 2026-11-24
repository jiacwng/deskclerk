# The display name of each employee, kept in Postgres so the admin page can show a name beside an
# id. Browser sign-in always records the name. A bearer caller is recorded the first time this
# process sees that name for them, so most requests cost no query.

from dataclasses import dataclass

from deskclerk.auth import Identity
from deskclerk.database import Database

SELECT_ALL = "SELECT user_id, display_name FROM employees ORDER BY lower(display_name), user_id"
UPSERT = (
    "INSERT INTO employees (user_id, display_name, last_seen) VALUES (%s, %s, now()) "
    "ON CONFLICT (user_id) DO UPDATE SET display_name = EXCLUDED.display_name, last_seen = now()"
)


@dataclass(frozen=True)
class Employee:
    user_id: str
    display_name: str


class EmployeeStore:
    def __init__(self, database: Database) -> None:
        self._pool = database.pool
        self._recorded: dict[str, str] = {}

    def recorded(self, identity: Identity) -> bool:
        return self._recorded.get(identity.user_id) == identity.display_name

    def record(self, identity: Identity) -> None:
        with self._pool.connection() as connection:
            connection.execute(UPSERT, (identity.user_id, identity.display_name))
        self._recorded[identity.user_id] = identity.display_name

    def all(self) -> list[Employee]:
        with self._pool.connection() as connection:
            found = connection.execute(SELECT_ALL).fetchall()
        return [Employee(str(user), str(name)) for user, name in found]

    def names(self) -> dict[str, str]:
        return {employee.user_id: employee.display_name for employee in self.all()}

    def delete_user(self, user_id: str) -> int:
        self._recorded.pop(user_id, None)
        with self._pool.connection() as connection:
            return connection.execute(
                "DELETE FROM employees WHERE user_id = %s", (user_id,)
            ).rowcount

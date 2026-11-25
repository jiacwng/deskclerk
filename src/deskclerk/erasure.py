# Erasing one employee: their MCP tokens, browser sessions, conversations, spending rows and
# display name are deleted and one audit line records the counts. The erase command calls it.

from dataclasses import dataclass

from deskclerk.audit import Audit
from deskclerk.conversations import delete_user_threads
from deskclerk.employees import EmployeeStore
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore


@dataclass(frozen=True)
class Erased:
    secrets: int
    sessions: int
    threads: int
    costs: int


def erase_user(
    user_id: str,
    tokens: TokenStore,
    sessions: SessionStore,
    spending_store: SpendingStore,
    employees: EmployeeStore,
    audit: Audit,
) -> Erased:
    erased = Erased(
        secrets=tokens.delete_user(user_id),
        sessions=sessions.delete_user(user_id),
        threads=delete_user_threads(tokens.database, user_id),
        costs=spending_store.delete_user(user_id),
    )
    employees.delete_user(user_id)
    audit.erased(user_id, erased.secrets, erased.sessions, erased.threads, erased.costs)
    return erased

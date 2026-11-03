# The summary behind /admin: each configured server, paused or not, with how many employees have
# connected it, what each employee spent this month, and the conversations most recently active.
# Recent activity comes from the conversations table, which holds the employee, the title and the
# time of the last message. A title is the start of a request, so it is shown only when
# DESKCLERK_AUDIT_TEXT allows request text in the logs. The audit routes read the audit table.

from datetime import datetime

from pydantic import BaseModel

from deskclerk.auditstore import AuditPage, AuditRow
from deskclerk.conversations import recent_conversations
from deskclerk.database import Database
from deskclerk.servers import McpServer
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore

RECENT_LIMIT = 50


class AdminServer(BaseModel):
    name: str
    url: str
    auth: str
    paused: bool
    employees_connected: int | None
    allowlist_size: int | None


class AdminUsage(BaseModel):
    user_id: str
    calls: int
    spent_usd: float
    budget_usd: float


class AdminActivity(BaseModel):
    user_id: str
    title: str | None
    updated_at: datetime


class AdminSummary(BaseModel):
    month: str
    servers: list[AdminServer]
    usage: list[AdminUsage]
    recent: list[AdminActivity]


class AdminAuditRow(BaseModel):
    id: int
    time: datetime
    user: str
    entry: str
    servers: list[str]
    tools: list[str]
    decisions: list[str]
    hash: str
    line: str


class AdminAuditPage(BaseModel):
    rows: list[AdminAuditRow]
    next_cursor: int | None


def server_rows(servers: list[McpServer], tokens: TokenStore) -> list[AdminServer]:
    counts = tokens.connected_counts()
    return [
        AdminServer(
            name=server.name,
            url=server.url,
            auth=server.auth,
            paused=server.paused,
            employees_connected=None if server.auth == "none" else counts.get(server.name, 0),
            allowlist_size=None if server.tools is None else len(server.tools),
        )
        for server in servers
    ]


def usage_rows(spending: SpendingStore, month: str, budget_usd: float) -> list[AdminUsage]:
    return [
        AdminUsage(
            user_id=row.user_id, calls=row.calls, spent_usd=row.total_usd, budget_usd=budget_usd
        )
        for row in spending.report(month)
    ]


async def recent_rows(database: Database, show_titles: bool) -> list[AdminActivity]:
    found = await recent_conversations(database, RECENT_LIMIT)
    return [
        AdminActivity(
            user_id=c.user_id, title=c.title if show_titles else None, updated_at=c.updated_at
        )
        for c in found
    ]


def audit_row(row: AuditRow) -> AdminAuditRow:
    return AdminAuditRow(**vars(row))


def audit_page(page: AuditPage) -> AdminAuditPage:
    return AdminAuditPage(rows=[audit_row(row) for row in page.rows], next_cursor=page.next_cursor)

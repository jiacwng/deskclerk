# The code sandbox as the agent sees it: a client for the sandbox runner, the table that keeps each
# conversation's files and the one tool, run_python. The code runs in a container with no network.
# The files it leaves in /mnt/data are stored here and put back before the next run.

import base64
import binascii
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from deskclerk.connections import ToolInfo, ToolResult
from deskclerk.database import Database
from deskclerk.servers import McpServer

SANDBOX_SERVER_NAME = "sandbox"
SANDBOX_TOOL_NAME = f"{SANDBOX_SERVER_NAME}__run_python"
REQUEST_TIMEOUT = httpx.Timeout(90.0)
MAX_REASON_CHARS = 300

SANDBOX_SERVER = McpServer(
    name=SANDBOX_SERVER_NAME, url="", client_id=None, client_secret=None, auth="none", trusted=True
)
RUN_PYTHON = ToolInfo(
    name="run_python",
    description=(
        "Run Python 3.12 code in a sandbox with pandas, numpy, matplotlib and openpyxl and no "
        "network. Files the code saves in /mnt/data are kept for this conversation, offered to "
        "the employee as downloads and put back in /mnt/data on every later run. Variables are "
        "not kept between runs. Print what you need to see."
    ),
    input_schema={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The Python code to run."}},
        "required": ["code"],
    },
    read_only=True,
    destructive=False,
    title="Run Python",
)


class SandboxFailed(Exception):
    pass


class RunnerFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    content: str


class RunnerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    error: str
    files: list[RunnerFile]


@dataclass(frozen=True)
class StoredFile:
    id: UUID
    name: str
    size: int


@dataclass(frozen=True)
class FileContent:
    name: str
    content: bytes


class SandboxClient:
    def __init__(self, url: str) -> None:
        self.http = httpx.AsyncClient(base_url=url, timeout=REQUEST_TIMEOUT)

    async def run(self, code: str, files: list[FileContent]) -> RunnerResult:
        body = {
            "code": code,
            "files": [
                {"name": file.name, "content": base64.b64encode(file.content).decode()}
                for file in files
            ],
        }
        try:
            response = await self.http.post("/run", json=body)
        except httpx.HTTPError as error:
            raise SandboxFailed(f"the runner did not answer ({type(error).__name__})") from None
        if response.status_code != httpx.codes.OK:
            raise SandboxFailed(f"the runner refused the run: {response.text[:MAX_REASON_CHARS]}")
        try:
            return RunnerResult.model_validate_json(response.content)
        except ValidationError:
            raise SandboxFailed("the runner sent an answer that is not a result") from None

    async def close(self) -> None:
        await self.http.aclose()


class SandboxFiles:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def contents(self, user_id: str, conversation: UUID) -> list[FileContent]:
        async with self.database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT name, content FROM sandbox_files "
                "WHERE user_id = %s AND conversation_id = %s ORDER BY name",
                (user_id, conversation),
            )
            found = await cursor.fetchall()
        return [FileContent(row["name"], bytes(row["content"])) for row in found]

    async def listing(self, user_id: str, conversation: UUID) -> list[StoredFile]:
        async with self.database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT id, name, size FROM sandbox_files "
                "WHERE user_id = %s AND conversation_id = %s ORDER BY name",
                (user_id, conversation),
            )
            found = await cursor.fetchall()
        return [StoredFile(row["id"], row["name"], row["size"]) for row in found]

    async def read(self, user_id: str, conversation: UUID, file_id: UUID) -> FileContent | None:
        async with self.database.async_pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT name, content FROM sandbox_files "
                "WHERE user_id = %s AND conversation_id = %s AND id = %s",
                (user_id, conversation, file_id),
            )
            row = await cursor.fetchone()
        return None if row is None else FileContent(row["name"], bytes(row["content"]))

    async def replace_all(
        self, user_id: str, conversation: UUID, files: list[FileContent]
    ) -> list[StoredFile]:
        changed: list[StoredFile] = []
        async with self.database.async_pool.connection() as connection:
            async with connection.transaction():
                for file in files:
                    cursor = await connection.execute(
                        "INSERT INTO sandbox_files "
                        "(id, user_id, conversation_id, name, content, size) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (user_id, conversation_id, name) DO UPDATE "
                        "SET content = excluded.content, size = excluded.size, created_at = now() "
                        "WHERE sandbox_files.content IS DISTINCT FROM excluded.content "
                        "RETURNING id, name, size",
                        (
                            uuid4(),
                            user_id,
                            conversation,
                            file.name,
                            file.content,
                            len(file.content),
                        ),
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        changed.append(StoredFile(row["id"], row["name"], row["size"]))
                await connection.execute(
                    "DELETE FROM sandbox_files WHERE user_id = %s AND conversation_id = %s "
                    "AND NOT (name = ANY(%s))",
                    (user_id, conversation, [file.name for file in files]),
                )
        return changed


@dataclass(frozen=True)
class Sandbox:
    client: SandboxClient
    files: SandboxFiles
    user_id: str
    conversation: UUID
    created: list[StoredFile] = field(default_factory=list)

    async def run_python(self, code: str) -> ToolResult:
        kept = await self.files.contents(self.user_id, self.conversation)
        try:
            result = await self.client.run(code, kept)
        except SandboxFailed as error:
            return ToolResult(f"The sandbox is not available: {error}", True)
        if result.error:
            return ToolResult(f"The code did not finish: {result.error}", True)
        try:
            written = [
                FileContent(file.name, base64.b64decode(file.content, validate=True))
                for file in result.files
            ]
        except binascii.Error:
            return ToolResult("The sandbox sent a file that is not valid.", True)
        self.created.extend(await self.files.replace_all(self.user_id, self.conversation, written))
        return ToolResult(_describe(result, written), result.exit_code != 0)


def _describe(result: RunnerResult, written: list[FileContent]) -> str:
    listed = ", ".join(f"{file.name} ({len(file.content):,} bytes)" for file in written)
    return (
        f"exit code: {result.exit_code}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
        f"files in /mnt/data: {listed or 'none'}"
    )

# The sandbox runner: one HTTP route that runs a snippet of Python in a fresh, locked down Docker
# container and returns its output and the files it wrote. It holds the Docker socket and no
# deskclerk secret. The container flags are constants here and no request field can change them.

import asyncio
import base64
import binascii
import contextlib
from asyncio.streams import StreamReader, StreamWriter
from dataclasses import dataclass
from typing import Self
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

IMAGE = "deskclerk-sandbox:latest"
RUN_SECONDS = 60
KILL_SECONDS = 20
MAX_CONCURRENT_RUNS = 4
MAX_CODE_CHARS = 100_000
MAX_FILES = 50
MAX_FILE_NAME_BYTES = 255
MAX_FILES_BYTES = 20 * 1024 * 1024
MAX_ENCODED_FILES = MAX_FILES_BYTES * 4 // 3 + 4 * MAX_FILES
MAX_REQUEST_BYTES = MAX_ENCODED_FILES + 2 * 1024 * 1024
MAX_OUTPUT_BYTES = MAX_ENCODED_FILES + 2 * 1024 * 1024
MAX_STDERR_BYTES = 4096
READ_CHUNK_BYTES = 65536
CONTAINER_FLAGS = (
    "--rm",
    "-i",
    "--network",
    "none",
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--user",
    "65534:65534",
    "--memory",
    "512m",
    "--memory-swap",
    "512m",
    "--pids-limit",
    "128",
    "--cpus",
    "1",
    "--log-driver",
    "none",
    "--tmpfs",
    "/mnt/data:size=64m,mode=1777",
    "--tmpfs",
    "/tmp:size=64m,mode=1777",
)


class BadResult(Exception):
    pass


def check_file_name(name: str) -> str:
    try:
        encoded = name.encode()
    except UnicodeEncodeError:
        raise ValueError("a file name is not valid text") from None
    if not name or name in (".", "..") or "/" in name or "\\" in name or not name.isprintable():
        raise ValueError("a file name is empty, has a path separator or has a control character")
    if len(encoded) > MAX_FILE_NAME_BYTES:
        raise ValueError(f"a file name is over {MAX_FILE_NAME_BYTES} bytes")
    return name


class SandboxFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    content: str

    @field_validator("name")
    @classmethod
    def _name_is_a_plain_file_name(cls, name: str) -> str:
        return check_file_name(name)

    @field_validator("content")
    @classmethod
    def _content_is_base64(cls, content: str) -> str:
        try:
            base64.b64decode(content, validate=True)
        except binascii.Error:
            raise ValueError("file content is not base64") from None
        return content


def _check_files(files: list[SandboxFile]) -> None:
    if sum(len(file.content) for file in files) > MAX_ENCODED_FILES:
        raise ValueError(f"the files are over {MAX_FILES_BYTES} bytes")
    if len({file.name for file in files}) != len(files):
        raise ValueError("two files have the same name")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(max_length=MAX_CODE_CHARS)
    files: list[SandboxFile] = Field(max_length=MAX_FILES)

    @model_validator(mode="after")
    def _files_fit(self) -> Self:
        _check_files(self.files)
        return self


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    error: str
    files: list[SandboxFile] = Field(max_length=MAX_FILES)

    @model_validator(mode="after")
    def _files_fit(self) -> Self:
        _check_files(self.files)
        return self


def parse_result(output: bytes) -> RunResult:
    try:
        return RunResult.model_validate_json(output)
    except ValidationError as error:
        reasons = "; ".join(str(problem["msg"]) for problem in error.errors())
        raise BadResult(
            f"the sandbox returned a result that was refused: {reasons[:500]}"
        ) from None


@dataclass(frozen=True)
class Finished:
    output: bytes
    errors: bytes
    exit_code: int
    failure: str


async def _read_up_to(stream: StreamReader, limit: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    while chunk := await stream.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            return b"", True
        chunks.append(chunk)
    return b"".join(chunks), False


async def _read_head(stream: StreamReader, limit: int) -> bytes:
    head = b""
    while chunk := await stream.read(READ_CHUNK_BYTES):
        head = (head + chunk)[:limit]
    return head


async def _send(stdin: StreamWriter, job: bytes) -> None:
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        stdin.write(job)
        await stdin.drain()
        stdin.close()


async def _kill(name: str, process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    await process.wait()
    removal = await asyncio.create_subprocess_exec(
        "docker",
        "rm",
        "--force",
        name,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(KILL_SECONDS):
            await removal.wait()


def _pipes(process: asyncio.subprocess.Process) -> tuple[StreamWriter, StreamReader, StreamReader]:
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise RuntimeError("the docker process has no pipes")
    return process.stdin, process.stdout, process.stderr


async def run_in_container(job: bytes) -> Finished:
    name = f"deskclerk-sandbox-{uuid4().hex}"
    process = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        *CONTAINER_FLAGS,
        "--name",
        name,
        IMAGE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin, stdout, stderr = _pipes(process)
    sending = asyncio.create_task(_send(stdin, job))
    reading_errors = asyncio.create_task(_read_head(stderr, MAX_STDERR_BYTES))
    try:
        async with asyncio.timeout(RUN_SECONDS):
            output, too_long = await _read_up_to(stdout, MAX_OUTPUT_BYTES)
            if too_long:
                await _kill(name, process)
                return Finished(b"", b"", -1, "the output was over the size limit")
            await process.wait()
            errors = await reading_errors
    except TimeoutError:
        await _kill(name, process)
        return Finished(b"", b"", -1, f"the code ran longer than {RUN_SECONDS} seconds")
    except BaseException:
        await _kill(name, process)
        raise
    finally:
        sending.cancel()
        reading_errors.cancel()
    return Finished(output, errors, process.returncode or 0, "")


async def _body_up_to(request: Request, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, "the request is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _problems(error: ValidationError) -> list[str]:
    return [f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}" for problem in error.errors()]


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    turns = asyncio.Semaphore(MAX_CONCURRENT_RUNS)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run")
    async def run(request: Request) -> RunResult:
        body = await _body_up_to(request, MAX_REQUEST_BYTES)
        try:
            job = RunRequest.model_validate_json(body)
        except ValidationError as error:
            raise HTTPException(422, _problems(error)) from None
        async with turns:
            finished = await run_in_container(job.model_dump_json().encode())
        if finished.failure:
            return RunResult(exit_code=-1, stdout="", stderr="", error=finished.failure, files=[])
        try:
            return parse_result(finished.output)
        except BadResult as error:
            detail = f"{error} (docker exit {finished.exit_code})"
            stderr = finished.errors[:500].decode(errors="replace")
            raise HTTPException(502, f"{detail}: {stderr}") from None

    return app

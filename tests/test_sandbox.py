# The code sandbox against the real runner (DESKCLERK_SANDBOX_URL) and Postgres: what a run returns,
# what the container cannot do, the limits that end a run, the files kept per conversation, who may
# download them and when they are deleted.

import asyncio
import base64
import importlib.util
import io
import json
import os
import subprocess
import time
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
import pytest
from cryptography.fernet import Fernet
from psycopg.conninfo import conninfo_to_dict

from conftest import DATABASE_URL
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.conversations import delete_user_threads, open_checkpointer
from deskclerk.database import Database
from deskclerk.retention import delete_old_threads
from deskclerk.sandbox import FileContent, Sandbox, SandboxClient, SandboxFiles
from test_agent import answer, calls, script
from test_api import Sign, as_user, ask, make_settings, web
from test_erase import KEY, rows, save_thread
from test_retention import checkpoint_id_aged

RUNNER_SOURCE = Path(__file__).parent.parent / "sandbox" / "runner" / "runner.py"
RUN_SECONDS = 60
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sandbox_runner", RUNNER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner_url() -> str:
    url = os.environ.get("DESKCLERK_SANDBOX_URL")
    if url is None:
        pytest.skip("DESKCLERK_SANDBOX_URL is not set")
    return url


def run_code(
    runner_url: str, code: str, files: dict[str, bytes] | None = None, timeout: float = 120
) -> dict[str, Any]:
    body = {
        "code": code,
        "files": [
            {"name": name, "content": base64.b64encode(content).decode()}
            for name, content in (files or {}).items()
        ],
    }
    response = httpx.post(f"{runner_url}/run", json=body, timeout=timeout)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def file_contents(result: dict[str, Any]) -> dict[str, bytes]:
    return {file["name"]: base64.b64decode(file["content"]) for file in result["files"]}


def leftover_containers() -> list[str]:
    listed = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "name=deskclerk-sandbox-"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return listed.stdout.split()


def test_a_run_returns_its_output_and_an_xlsx_and_a_png_that_open(runner_url: str) -> None:
    result = run_code(
        runner_url,
        "import pandas as pd, matplotlib.pyplot as plt\n"
        "frame = pd.DataFrame({'units': [1, 2, 3]})\n"
        "frame.to_excel('units.xlsx')\n"
        "plt.plot(frame['units'])\n"
        "plt.savefig('units.png')\n"
        "print(frame['units'].sum())\n",
    )
    assert (result["exit_code"], result["stdout"].strip(), result["error"]) == (0, "6", "")
    files = file_contents(result)
    assert files["units.png"].startswith(PNG_SIGNATURE)
    with zipfile.ZipFile(io.BytesIO(files["units.xlsx"])) as workbook:
        assert "xl/workbook.xml" in workbook.namelist()
    assert leftover_containers() == []


def test_an_empty_script_and_a_failing_one_end_with_a_result(runner_url: str) -> None:
    assert run_code(runner_url, "")["exit_code"] == 0
    failed = run_code(runner_url, "raise ValueError('boom')")
    assert failed["exit_code"] == 1
    assert "ValueError: boom" in failed["stderr"]


def test_the_container_has_no_network_interface_and_cannot_reach_the_database_or_the_runner(
    runner_url: str,
) -> None:
    database = conninfo_to_dict(DATABASE_URL)
    runner = urlparse(runner_url)
    targets = [(database["host"], int(str(database["port"]))), (runner.hostname, runner.port)]
    result = run_code(
        runner_url,
        "import socket\n"
        "print(sorted(name for _, name in socket.if_nameindex()))\n"
        f"for target in {targets!r}:\n"
        "    try:\n"
        "        socket.create_connection(tuple(target), timeout=3)\n"
        "        print('connected', target)\n"
        "    except OSError as error:\n"
        "        print('failed', target, type(error).__name__)\n",
    )
    lines = result["stdout"].splitlines()
    assert lines[0] == "['lo']"
    assert [line.split()[0] for line in lines[1:]] == ["failed", "failed"]


def test_the_environment_holds_no_deskclerk_or_postgres_name(runner_url: str) -> None:
    result = run_code(
        runner_url,
        "import os\n"
        "print([name for name in os.environ if name.startswith(('DESKCLERK_', 'POSTGRES_'))])\n"
        "os.system('cat /proc/1/environ')\n",
    )
    first, *rest = result["stdout"].splitlines()
    assert first == "[]"
    assert "DESKCLERK_" not in "".join(rest) and "POSTGRES_" not in "".join(rest)


def test_writing_outside_the_data_and_tmp_folders_fails(runner_url: str) -> None:
    result = run_code(
        runner_url,
        "for path in ('/etc/x', '/usr/local/x', '/opt/x', '/mnt/x', '/x'):\n"
        "    try:\n"
        "        open(path, 'w').close()\n"
        "        print('wrote', path)\n"
        "    except OSError:\n"
        "        print('refused', path)\n"
        "open('/tmp/ok', 'w').close()\n"
        "open('/mnt/data/ok', 'w').close()\n"
        "print('wrote the allowed folders')\n",
    )
    lines = result["stdout"].splitlines()
    assert lines == [f"refused {path}" for path in ("/etc/x", "/usr/local/x", "/opt/x")] + [
        "refused /mnt/x",
        "refused /x",
        "wrote the allowed folders",
    ]


def test_an_endless_loop_is_stopped_at_the_time_limit_and_leaves_no_container(
    runner_url: str,
) -> None:
    started = time.monotonic()
    result = run_code(runner_url, "while True:\n    pass\n")
    elapsed = time.monotonic() - started
    assert f"longer than {RUN_SECONDS} seconds" in result["error"]
    assert RUN_SECONDS - 5 < elapsed < RUN_SECONDS + 30
    assert leftover_containers() == []


def test_a_fork_bomb_a_memory_hog_and_endless_printing_each_end_in_an_error(
    runner_url: str,
) -> None:
    bomb = run_code(runner_url, "import os\nwhile True:\n    os.fork()\n")
    assert bomb["exit_code"] != 0 and "BlockingIOError" in bomb["stderr"]
    hog = run_code(runner_url, "held = []\nwhile True:\n    held.append(b'x' * 50_000_000)\n")
    assert hog["exit_code"] != 0
    printing = run_code(runner_url, "while True:\n    print('x' * 1000)\n")
    assert printing["exit_code"] != 0
    assert "[output cut]" in printing["stdout"] and len(printing["stdout"]) < 70_000
    assert leftover_containers() == []


def test_output_written_straight_to_the_container_stdout_is_cut_by_the_runner(
    runner_url: str,
) -> None:
    result = run_code(
        runner_url,
        "import os\n"
        "with open(f'/proc/{os.getppid()}/fd/1', 'wb') as pipe:\n"
        "    while True:\n"
        "        pipe.write(b'x' * 1_000_000)\n",
    )
    assert "over the size limit" in result["error"]
    assert leftover_containers() == []


def test_files_over_the_size_cap_end_in_an_error(runner_url: str) -> None:
    over_the_cap = run_code(runner_url, "open('big.bin', 'wb').write(b'x' * 30_000_000)")
    assert "over the" in over_the_cap["error"] and over_the_cap["files"] == []
    over_the_disk = run_code(
        runner_url,
        "with open('huge.bin', 'wb') as huge:\n"
        "    for _ in range(200):\n"
        "        huge.write(b'x' * 1_000_000)\n",
    )
    assert over_the_disk["exit_code"] != 0 and over_the_disk["error"] != ""
    assert leftover_containers() == []


def test_a_file_with_spaces_and_unicode_in_its_name_comes_back_and_goes_in(
    runner_url: str,
) -> None:
    name = "résumé 日本 final.txt"
    result = run_code(runner_url, f"open({name!r}, 'w').write('hello')")
    assert file_contents(result) == {name: b"hello"}
    again = run_code(runner_url, f"print(open({name!r}).read())", {name: b"sent"})
    assert again["stdout"].strip() == "sent"


def test_a_request_cannot_change_the_container_and_must_stay_within_the_limits(
    runner_url: str,
) -> None:
    def post(body: dict[str, Any]) -> int:
        return httpx.post(f"{runner_url}/run", json=body, timeout=60).status_code

    assert post({"code": "", "files": [], "network": "host"}) == 422
    assert post({"code": "x" * 1_000_000, "files": []}) == 422
    assert post({"code": "", "files": [{"name": "../x", "content": ""}]}) == 422
    assert post({"code": "", "files": [{"name": "a/b", "content": ""}]}) == 422
    assert post({"code": "", "files": [{"name": "a", "content": "not base64!"}]}) == 422
    assert post({"code": "", "files": [{"name": "a", "content": ""}] * 2}) == 422
    assert leftover_containers() == []


def test_a_result_with_a_path_in_a_file_name_is_refused() -> None:
    runner = load_runner()
    for name in ("../x", "a/b", "..", "", "back\\slash", "line\nbreak"):
        hostile = json.dumps(
            {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "error": "",
                "files": [{"name": name, "content": ""}],
            }
        )
        with pytest.raises(runner.BadResult):
            runner.parse_result(hostile.encode())
    with pytest.raises(runner.BadResult):
        runner.parse_result(b"")


@pytest.fixture
async def sandbox_files(served_database: Database) -> SandboxFiles:
    return SandboxFiles(served_database)


@pytest.fixture
async def client(runner_url: str) -> AsyncIterator[SandboxClient]:
    opened = SandboxClient(runner_url)
    yield opened
    await opened.close()


@pytest.mark.anyio
async def test_a_second_run_sees_the_first_runs_file_and_another_conversation_does_not(
    client: SandboxClient, sandbox_files: SandboxFiles
) -> None:
    here, elsewhere = uuid4(), uuid4()
    first = Sandbox(client, sandbox_files, "ada", here)
    wrote = await first.run_python("open('notes.txt', 'w').write('kept')")
    assert wrote.is_error is False and "notes.txt (4 bytes)" in wrote.text
    assert [file.name for file in first.created] == ["notes.txt"]

    second = Sandbox(client, sandbox_files, "ada", here)
    read = await second.run_python("print(open('notes.txt').read())")
    assert "stdout:\nkept\n" in read.text and second.created == []

    other_conversation = await Sandbox(client, sandbox_files, "ada", elsewhere).run_python(
        "print(open('notes.txt').read())"
    )
    assert other_conversation.is_error and "FileNotFoundError" in other_conversation.text
    other_employee = await Sandbox(client, sandbox_files, "grace", here).run_python(
        "print(open('notes.txt').read())"
    )
    assert other_employee.is_error and "FileNotFoundError" in other_employee.text


@pytest.mark.anyio
async def test_a_changed_file_is_reported_again_and_a_deleted_one_is_forgotten(
    client: SandboxClient, sandbox_files: SandboxFiles
) -> None:
    conversation = uuid4()
    await Sandbox(client, sandbox_files, "ada", conversation).run_python(
        "open('a.txt', 'w').write('1')\nopen('b.txt', 'w').write('1')"
    )
    later = Sandbox(client, sandbox_files, "ada", conversation)
    await later.run_python("import os\nopen('a.txt', 'w').write('2')\nos.remove('b.txt')")
    assert [file.name for file in later.created] == ["a.txt"]
    assert [file.name for file in await sandbox_files.listing("ada", conversation)] == ["a.txt"]


@pytest.mark.anyio
async def test_a_runner_that_is_down_gives_the_model_an_error_not_a_crash(
    sandbox_files: SandboxFiles,
) -> None:
    unreachable = SandboxClient("http://127.0.0.1:1")
    result = await Sandbox(unreachable, sandbox_files, "ada", uuid4()).run_python("print(1)")
    await unreachable.close()
    assert result.is_error and "not available" in result.text


def sandbox_rows(database: Database) -> int:
    with database.pool.connection() as connection:
        found = connection.execute("SELECT count(*) FROM sandbox_files").fetchone()
    assert found is not None
    return int(found[0])


def test_the_agent_offers_the_tool_only_when_a_runner_is_set_and_an_ask_returns_the_files(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    runner_url: str,
) -> None:
    code = "open('report.txt', 'w').write('numbers')"
    without = script(answer("nothing"))
    with web(
        make_settings(token_settings), [], without, token_signing_key, database, key=KEY
    ) as site:
        assert ask(site, as_user(sign_token, "ada")).status_code == 200
    assert [tool.name for tool in without.bound] == []

    model = script(calls(("sandbox__run_python", {"code": code})), answer("done"))
    settings = make_settings(token_settings, sandbox_url=runner_url)
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(settings, [], model, token_signing_key, database, key=KEY) as site:
        body = ask(site, ada).json()
        assert [tool.name for tool in model.bound] == ["sandbox__run_python"]
        assert [(call["server"], call["tool"], call["is_error"]) for call in body["executed"]] == [
            ("sandbox", "run_python", False)
        ]
        [file] = body["files"]
        assert (file["name"], file["size"]) == ("report.txt", 7)
        path = f"/conversations/{body['conversation']}/files/{file['id']}"

        download = site.client.get(path, headers=ada)
        assert download.status_code == 200 and download.content == b"numbers"
        assert download.headers["content-disposition"].startswith("attachment; ")
        assert download.headers["x-content-type-options"] == "nosniff"
        assert site.client.get(path, headers=grace).status_code == 404
        assert site.client.get(path).status_code == 401
        other = f"/conversations/{uuid4()}/files/{file['id']}"
        assert site.client.get(other, headers=ada).status_code == 404

        reopened = site.client.get(f"/conversations/{body['conversation']}", headers=ada).json()
        assert [kept["name"] for kept in reopened["files"]] == ["report.txt"]

        assert sandbox_rows(database) == 1
        deleted = site.client.delete(f"/conversations/{body['conversation']}", headers=ada)
        assert deleted.status_code == 204
        assert sandbox_rows(database) == 0
    assert leftover_containers() == []


def test_erasing_an_employee_and_the_retention_sweep_delete_their_files(
    database: Database,
) -> None:
    async def keep_and_sweep() -> None:
        async with database.serving():
            saver = await open_checkpointer(database, Fernet(KEY))
            old = await save_thread(saver, "ana", checkpoint_id_aged(40))
            new = await save_thread(saver, "ana")
            files = SandboxFiles(database)
            for thread in (old, new):
                conversation = UUID(thread.split("/")[1])
                await files.replace_all("ana", conversation, [FileContent("a.txt", b"1")])
            await files.replace_all("anabel", uuid4(), [FileContent("a.txt", b"1")])
            await delete_old_threads(database, saver, 30)

    asyncio.run(keep_and_sweep())
    assert sandbox_rows(database) == 2
    delete_user_threads(database, "ana")
    assert rows(database, "SELECT user_id FROM sandbox_files") == [("anabel",)]

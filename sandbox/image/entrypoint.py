# Runs inside the sandbox container: reads one job from standard input, puts the files in
# /mnt/data, runs the code as a child process there and writes one JSON result to standard output.

import base64
import json
import os
import stat
import subprocess
import sys

DATA = "/mnt/data"
SCRIPT = "/tmp/main.py"
STDOUT_FILE = "/tmp/stdout.txt"
STDERR_FILE = "/tmp/stderr.txt"
MAX_STREAM_BYTES = 65536
MAX_FILES_BYTES = 20 * 1024 * 1024


def read_stream(path: str) -> str:
    with open(path, "rb") as stream:
        data = stream.read(MAX_STREAM_BYTES + 1)
    text = data[:MAX_STREAM_BYTES].decode(errors="replace")
    return text + "\n[output cut]" if len(data) > MAX_STREAM_BYTES else text


def written_files() -> list[tuple[str, int]]:
    found = []
    for entry in os.scandir(DATA):
        status = entry.stat(follow_symlinks=False)
        if stat.S_ISREG(status.st_mode):
            found.append((entry.name, status.st_size))
    return found


def collect_files() -> tuple[list[dict[str, str]], str]:
    found = written_files()
    total = sum(size for _, size in found)
    if total > MAX_FILES_BYTES:
        return [], f"the files in /mnt/data are {total:,} bytes, over the {MAX_FILES_BYTES:,} limit"
    files = []
    for name, _ in found:
        descriptor = os.open(os.path.join(DATA, name), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as opened:
            files.append({"name": name, "content": base64.b64encode(opened.read()).decode()})
    return files, ""


def main() -> None:
    job = json.load(sys.stdin)
    for file in job["files"]:
        with open(os.path.join(DATA, file["name"]), "wb") as written:
            written.write(base64.b64decode(file["content"]))
    with open(SCRIPT, "w") as script:
        script.write(job["code"])
    with open(STDOUT_FILE, "wb") as out, open(STDERR_FILE, "wb") as err:
        finished = subprocess.run(
            [sys.executable, SCRIPT], cwd=DATA, stdin=subprocess.DEVNULL, stdout=out, stderr=err
        )
    files, error = collect_files()
    result = {
        "exit_code": finished.returncode,
        "stdout": read_stream(STDOUT_FILE),
        "stderr": read_stream(STDERR_FILE),
        "error": error,
        "files": files,
    }
    sys.stdout.write(json.dumps(result))


main()

# Makes the token key and the session key on the first start when their variables are unset, keeps
# each in a file in the data directory, and reads that file on every later start.

import base64
import os
import secrets
import sys
import tempfile
from collections.abc import Callable, MutableMapping
from pathlib import Path

from deskclerk.config import Settings


def new_token_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def new_session_key() -> str:
    return secrets.token_urlsafe(32)


GENERATED_KEYS: tuple[tuple[str, str, Callable[[], str]], ...] = (
    ("DESKCLERK_TOKEN_KEY", "token_key", new_token_key),
    ("DESKCLERK_SESSION_KEY", "session_key", new_session_key),
)


def fill_missing_keys(environ: MutableMapping[str, str] = os.environ) -> None:
    data_dir = Path(
        environ.get("DESKCLERK_DATA_DIR", str(Settings.model_fields["data_dir"].default))
    )
    for variable, file_name, make_key in GENERATED_KEYS:
        if not environ.get(variable):
            environ[variable] = _key_from_file(data_dir / file_name, make_key)


def _key_from_file(path: Path, make_key: Callable[[], str]) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if _create(path, make_key()):
            print(f"deskclerk: generated {path.name} in {path.parent}", file=sys.stderr)
    return path.read_text(encoding="utf-8").strip()


def _create(path: Path, key: str) -> bool:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as written:
            written.write(key + "\n")
        os.link(temporary, path)
    except FileExistsError:
        return False
    finally:
        os.unlink(temporary)
    return True

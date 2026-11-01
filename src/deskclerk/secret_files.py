# Docker's NAME_FILE convention: for each secret variable NAME, the file named by NAME_FILE
# supplies its value, so a mounted secret never sits in the process environment listing. The
# variables a server entry names are read the same way when the server list is loaded.

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

SECRET_VARIABLES = (
    "DESKCLERK_DATABASE_URL",
    "DESKCLERK_TOKEN_KEY",
    "DESKCLERK_SESSION_KEY",
    "DESKCLERK_AUDIT_CHAIN_KEY",
    "DESKCLERK_CLIENT_SECRET",
    "DESKCLERK_AUDIT_KEY",
    "DESKCLERK_SCIM_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "TYPESAFE_API_KEY",
    "LANGFUSE_SECRET_KEY",
)


class SecretFileError(Exception):
    pass


def file_secret(name: str, environ: Mapping[str, str]) -> str | None:
    file_name = environ.get(f"{name}_FILE")
    if file_name is None:
        return None
    if name in environ:
        raise SecretFileError(f"both {name} and {name}_FILE are set; set only one")
    try:
        content = Path(file_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SecretFileError(
            f"{name}_FILE names {file_name}, which cannot be read: {error}"
        ) from None
    return content.rstrip("\r\n")


def read_secret_files(environ: MutableMapping[str, str] = os.environ) -> None:
    for name in SECRET_VARIABLES:
        content = file_secret(name, environ)
        if content is not None:
            environ[name] = content

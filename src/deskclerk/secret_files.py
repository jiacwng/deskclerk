# Docker's NAME_FILE convention: for each secret variable NAME, the file named by NAME_FILE
# supplies its value, so a mounted secret never sits in the process environment listing.

import os
from collections.abc import MutableMapping
from pathlib import Path

from deskclerk.config import Settings
from deskclerk.servers import secret_variables

SECRET_VARIABLES = (
    "DESKCLERK_TOKEN_KEY",
    "DESKCLERK_SESSION_KEY",
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


def read_secret_files(environ: MutableMapping[str, str] = os.environ) -> None:
    servers_file = Path(
        environ.get("DESKCLERK_SERVERS_FILE", str(Settings.model_fields["servers_file"].default))
    )
    listed = secret_variables(servers_file) if servers_file.exists() else []
    for name in dict.fromkeys([*SECRET_VARIABLES, *listed]):
        file_name = environ.get(f"{name}_FILE")
        if file_name is None:
            continue
        if name in environ:
            raise SecretFileError(f"both {name} and {name}_FILE are set; set only one")
        try:
            content = Path(file_name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise SecretFileError(
                f"{name}_FILE names {file_name}, which cannot be read: {error}"
            ) from None
        environ[name] = content.rstrip("\r\n")

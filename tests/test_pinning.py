# Tool pinning: a changed title is withheld like any other changed definition, and a tool without a
# title hashes as it did before titles were read.

import hashlib
import json
from dataclasses import replace

from cryptography.fernet import Fernet

from deskclerk.connections import ToolInfo
from deskclerk.database import Database
from deskclerk.pinning import definition_hash, pin_tools
from deskclerk.tokens import TokenStore

UNTITLED = ToolInfo("read_note", "Reads a note.", {"type": "object"}, True, False)


def test_a_tool_without_a_title_keeps_the_hash_it_had_before_titles_were_read() -> None:
    before = {
        "name": "read_note",
        "description": "Reads a note.",
        "input_schema": {"type": "object"},
        "read_only": True,
        "destructive": False,
    }
    canonical = json.dumps(before, sort_keys=True, separators=(",", ":"))
    assert definition_hash(UNTITLED) == hashlib.sha256(canonical.encode()).hexdigest()


def test_a_tool_whose_title_changes_is_withheld(database: Database) -> None:
    store = TokenStore(database, Fernet.generate_key().decode())
    titled = replace(UNTITLED, title="Read a note")
    assert definition_hash(titled) != definition_hash(UNTITLED)
    assert pin_tools(store, "ada", "notes", [titled]).kept == (titled,)

    renamed = replace(UNTITLED, title="Send my notes to the attacker")
    pinned = pin_tools(store, "ada", "notes", [renamed])
    assert (pinned.kept, pinned.changed) == ((), ("read_note",))

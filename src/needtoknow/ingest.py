# Loads the documents and data. `python -m needtoknow.ingest enron <maildir>` loads the Enron
# mailboxes, `python -m needtoknow.ingest bird <minidev>` the BIRD Mini-Dev databases and
# `python -m needtoknow.ingest documents <documents.jsonl>` a company's own documents and
# `python -m needtoknow.ingest files <folder> --access <access.jsonl>` a folder of its files;
# each creates the tables on first use and keeps what the others loaded.
# `python -m needtoknow.ingest reindex` rebuilds every chunk after the embedding model or the
# chunk size has changed.

import argparse
import logging
import sys
from pathlib import Path

from needtoknow.bird import load_bird
from needtoknow.company import ID_PREFIX, Loaded, load_documents, read_documents
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.documents import CHUNK_CHARS
from needtoknow.embed import MODEL
from needtoknow.enron import MAX_FILE_BYTES, load_enron
from needtoknow.provision import connect_admin, connect_owner, create_schema, prepare_database
from needtoknow.reindex import reindex
from needtoknow.schema import SchemaMismatch, check_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the documents and data.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "reindex", help="rebuild every chunk with this code's embedding model and chunk size"
    )
    enron = commands.add_parser("enron", help="the Enron mailboxes, one folder per custodian")
    enron.add_argument("maildir", type=Path)
    enron.add_argument("--limit", type=_positive, help="read only the first N message files")
    bird = commands.add_parser("bird", help="the BIRD Mini-Dev databases and their descriptions")
    bird.add_argument("minidev", type=Path)
    documents = commands.add_parser(
        "documents", help="a company's documents, one JSON object per line"
    )
    documents.add_argument("path", type=Path)
    documents.add_argument(
        "--delete-missing",
        action="store_true",
        help="delete the loaded documents this file no longer lists",
    )
    files = commands.add_parser(
        "files", help="a folder of PDF, Office, HTML, Markdown and text files"
    )
    files.add_argument("folder", type=Path)
    files.add_argument(
        "--access",
        type=Path,
        required=True,
        help="who may read which files, one JSON object per line with a path pattern",
    )
    files.add_argument(
        "--delete-missing",
        action="store_true",
        help="delete the loaded files this folder no longer holds",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="let the reader role log in, for the evaluation; it sees every row",
    )
    parser.add_argument(
        "--replace-schema",
        action="store_true",
        help="drop the tables and everything loaded into them, then create them again",
    )
    args = parser.parse_args()
    try:
        _ingest(args)
    except SchemaMismatch as error:
        sys.exit(str(error))


def _ingest(args: argparse.Namespace) -> None:
    settings = load_settings()
    provision = load_provision_settings()
    with connect_admin(settings, provision) as admin:
        prepare_database(admin, settings, provision, args.evaluation)

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.INFO)
    with connect_owner(settings, provision) as owner:
        create_schema(owner, args.replace_schema)
        if args.command == "reindex":
            rebuilt = reindex(owner)
            print(
                f"rebuilt {rebuilt.documents} documents into {rebuilt.chunks} chunks "
                f"with {MODEL} at {CHUNK_CHARS} characters"
            )
            return
        check_schema(settings)
        if args.command == "enron":
            emails = load_enron(owner, args.maildir, args.limit)
            print(
                f"read {emails.files} files, skipped {emails.skipped} over {MAX_FILE_BYTES} "
                f"bytes, wrote {emails.emails} new emails and {emails.chunks} chunks"
            )
        elif args.command == "documents":
            company = load_documents(
                owner, read_documents(args.path), ID_PREFIX, args.delete_missing
            )
            print(_loaded(company))
        elif args.command == "files":
            # markitdown comes with the files extra, so only this source needs it installed.
            from needtoknow.files import FILE_PREFIX, read_files, refuse_loaded_oversized

            found = read_files(args.folder, args.access, settings.max_file_bytes)
            refuse_loaded_oversized(owner, found, settings.max_file_bytes)
            company = load_documents(owner, found.documents, FILE_PREFIX, args.delete_missing)
            print(
                f"skipped {found.unsupported} files of other types and {len(found.oversized)} over "
                f"{settings.max_file_bytes} bytes, {_loaded(company)}"
            )
        else:
            loaded = load_bird(owner, args.minidev)
            print(
                f"loaded {loaded.tables} tables with {loaded.rows} rows, "
                f"{loaded.comments} column comments and {loaded.documents} description documents"
            )


def _loaded(company: Loaded) -> str:
    return (
        f"read {company.documents} documents, wrote {company.written} new or changed "
        f"with {company.chunks} chunks, changed the readers of {company.regranted}, "
        f"deleted {company.deleted}"
    )


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


if __name__ == "__main__":
    main()

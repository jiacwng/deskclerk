# Loads the public data. `python -m needtoknow.ingest enron <maildir>` loads the Enron mailboxes
# and `python -m needtoknow.ingest bird <minidev>` the BIRD Mini-Dev databases; each creates the
# tables on first use and keeps what the other loaded.

import argparse
import logging
from pathlib import Path

from needtoknow.bird import load_bird
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.enron import MAX_FILE_BYTES, load_enron
from needtoknow.provision import connect_admin, connect_owner, create_schema, prepare_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the documents and data.")
    sources = parser.add_subparsers(dest="source", required=True)
    enron = sources.add_parser("enron", help="the Enron mailboxes, one folder per custodian")
    enron.add_argument("maildir", type=Path)
    enron.add_argument("--limit", type=_positive, help="read only the first N message files")
    bird = sources.add_parser("bird", help="the BIRD Mini-Dev databases and their descriptions")
    bird.add_argument("minidev", type=Path)
    args = parser.parse_args()

    settings = load_settings()
    provision = load_provision_settings()
    with connect_admin(settings, provision) as admin:
        prepare_database(admin, settings, provision)

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.INFO)
    with connect_owner(settings, provision) as owner:
        if owner.execute("SELECT to_regclass('public.documents')").fetchone() == (None,):
            create_schema(owner)
        if args.source == "enron":
            emails = load_enron(owner, args.maildir, args.limit)
            print(
                f"read {emails.files} files, skipped {emails.skipped} over {MAX_FILE_BYTES} "
                f"bytes, wrote {emails.emails} new emails and {emails.chunks} chunks"
            )
        else:
            loaded = load_bird(owner, args.minidev)
            print(
                f"loaded {loaded.tables} tables with {loaded.rows} rows, "
                f"{loaded.comments} column comments and {loaded.documents} description documents"
            )


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


if __name__ == "__main__":
    main()

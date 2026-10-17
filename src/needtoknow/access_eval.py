# Whether row-level security ever hands an email to a custodian who does not hold it, checked in
# the database with no model. Every Enron email is opened as one non-holder, and a sample is
# searched for by its subject as a non-holder; each count should be zero. The same checks as a
# holder show the tools do return the email to someone allowed to read it.
# A full run (NEEDTOKNOW_DB_NAME=needtoknow_full):
# PYTHONPATH=src .venv/bin/python -m needtoknow.access_eval --searches 10000 --out results/

import argparse
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from needtoknow.auth import principals
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.db import connect_app, connect_reader
from needtoknow.leak_eval import TEMPLATE
from needtoknow.retrieval import open_document, search
from needtoknow.stats import with_interval

SEED = 0
SEARCH_RESULTS = 5
_log = logging.getLogger("needtoknow.access_eval")


@dataclass(frozen=True)
class Email:
    doc_id: str
    subject: str
    holder: str
    outsider: str


def sample_emails(reader: psycopg.Connection[Any]) -> list[Email]:
    rows = reader.execute(
        "SELECT doc_id, principal FROM doc_access WHERE doc_id LIKE 'enron-%' AND principal "
        "LIKE 'user:%'"
    ).fetchall()
    holders: dict[str, list[str]] = {}
    for doc_id, principal in rows:
        holders.setdefault(doc_id, []).append(principal.removeprefix("user:"))
    everyone = sorted({user for users in holders.values() for user in users})
    subjects = dict(
        reader.execute("SELECT id, title FROM documents WHERE id LIKE 'enron-%'").fetchall()
    )
    sampler = random.Random(SEED)  # noqa: S311, a reproducible sample, not a secret
    emails = []
    for doc_id in sorted(holders):
        users = sorted(holders[doc_id])
        outsiders = [user for user in everyone if user not in users]
        if outsiders:
            holder = sampler.choice(users)
            emails.append(Email(doc_id, subjects[doc_id] or "", holder, sampler.choice(outsiders)))
    return emails


def count_opens(app: psycopg.Connection[Any], emails: Sequence[Email], as_holder: bool) -> int:
    opened = 0
    for number, email in enumerate(emails, start=1):
        user = email.holder if as_holder else email.outsider
        opened += open_document(app, principals(user, []), email.doc_id) is not None
        if number % 50_000 == 0:
            _log.info("opened %d of %d", number, len(emails))
    return opened


def count_found(app: psycopg.Connection[Any], emails: Sequence[Email], as_holder: bool) -> int:
    found = 0
    for email in emails:
        user = email.holder if as_holder else email.outsider
        question = TEMPLATE.format(subject=email.subject)
        hits = search(app, principals(user, []), question, SEARCH_RESULTS)
        found += any(hit.doc_id == email.doc_id for hit in hits)
    return found


def results_table(summary: dict[str, Any]) -> str:
    opens, searches = summary["opens"], summary["searches"]
    return "\n".join(
        [
            "# Access checks on Enron, without a model",
            "",
            f"Every email held by at least one custodian is opened as one custodian who does not "
            f"hold it ({opens:,} emails), and {searches:,} of them are searched for by their "
            f"subject, top {SEARCH_RESULTS}, as that custodian. Counts carry their 95% Wilson "
            "score interval. The holder rows show the same tools do return the email to a "
            "custodian allowed to read it.",
            "",
            "| Check | Non-holder got the email | Holder got the email |",
            "|---|---|---|",
            f"| open_document (of {opens:,}) | {with_interval(summary['outsider_opened'], opens)} "
            f"| {with_interval(summary['holder_opened'], opens)} |",
            f"| search, top {SEARCH_RESULTS} (of {searches:,}) "
            f"| {with_interval(summary['outsider_found'], searches)} "
            f"| {with_interval(summary['holder_found'], searches)} |",
            "",
        ]
    )


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Check database access on Enron, no model.")
    parser.add_argument("--searches", type=_positive, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.INFO)
    with (
        connect_app(settings) as app,
        connect_reader(settings, load_provision_settings()) as reader,
    ):
        emails = sample_emails(reader)
        searched = random.Random(SEED).sample(emails, min(args.searches, len(emails)))  # noqa: S311
        summary = {
            "opens": len(emails),
            "outsider_opened": count_opens(app, emails, as_holder=False),
            "holder_opened": count_opens(app, emails, as_holder=True),
            "searches": len(searched),
            "outsider_found": count_found(app, searched, as_holder=False),
            "holder_found": count_found(app, searched, as_holder=True),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "access.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "access.md").write_text(results_table(summary), encoding="utf-8")
    print(results_table(summary))


if __name__ == "__main__":
    main()

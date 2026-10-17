# Cuts the small real slices the tests load from the full downloads, so CI needs no dataset:
# 50 Enron emails from three mailboxes, five of them held by two custodians, and the BIRD
# financial and california_schools databases for two districts and two counties, as a dump in
# the same format with their description files. The downloads are untrusted, so run it isolated:
# .venv/bin/python -I tests/data/extract.py <enron maildir> <bird minidev>

import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

DATA = Path(__file__).resolve().parent
CUSTODIANS = ("kaminski-v", "dasovich-j", "kean-s")
SHARED_BY = ("dasovich-j", "kean-s")
SHARED = 5
OWN = 15
BODY_CHARS = (300, 2000)
DATABASES = ("financial", "california_schools")
DISTRICTS = ("1", "58")
ACCOUNTS_PER_DISTRICT = 5
TRANS_PER_ACCOUNT = 20
COUNTIES = ("Sierra", "Mariposa")
SCHOOLS_PER_COUNTY = 8

_TABLE = re.compile(r'public\.("?)(\w+)\1')


def main() -> None:
    maildir, minidev = Path(sys.argv[1]), Path(sys.argv[2])
    _enron(maildir, DATA / "enron" / "maildir")
    _bird(minidev, DATA / "bird")


def _enron(maildir: Path, out: Path) -> None:
    # -I leaves the repository's src off the path; the hash must be the loader's own.
    sys.path.insert(0, str(DATA.parents[1] / "src"))
    from needtoknow.enron import Email, parse_email

    mailboxes: dict[str, list[tuple[Path, Email]]] = {}
    for custodian in CUSTODIANS:
        mailboxes[custodian] = []
        for path in sorted((maildir / custodian).rglob("*")):
            if path.is_file() and not path.is_symlink():
                email = parse_email(path.read_bytes())
                if email.subject and BODY_CHARS[0] <= len(email.body) <= BODY_CHARS[1]:
                    mailboxes[custodian].append((path, email))

    holders: dict[str, set[str]] = {}
    for custodian, emails in mailboxes.items():
        for _, email in emails:
            holders.setdefault(email.id, set()).add(custodian)
    shared = []
    for _, email in mailboxes[SHARED_BY[0]]:
        if holders[email.id] == set(SHARED_BY) and email.id not in shared:
            shared.append(email.id)
    chosen = set(shared[:SHARED])

    if out.exists():
        shutil.rmtree(out)
    for custodian, emails in mailboxes.items():
        own = 0
        written = set()
        for path, email in emails:
            is_own = holders[email.id] == {custodian} and own < OWN
            if email.id in written or not (email.id in chosen or is_own):
                continue
            own += is_own
            written.add(email.id)
            # Windows drops a trailing dot from a file name, and the release names files "1.".
            target = out / path.relative_to(maildir).parent / path.name.rstrip(".")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def _bird(minidev: Path, out: Path) -> None:
    names = _tables(minidev)
    items: list[tuple[str, list[str] | None]] = []
    header: list[str] = []
    with (minidev / "MINIDEV_postgresql" / "BIRD_dev.sql").open(encoding="utf-8") as dump:
        lines: list[str] = []
        for line in dump:
            if not lines and (not line.strip() or line.startswith("--")):
                continue
            lines.append(line)
            if not line.rstrip().endswith(";"):
                continue
            statement = "".join(lines).strip()
            lines = []
            match = _TABLE.search(statement)
            if match is None:
                if not items:
                    header.append(statement)
                continue
            if match.group(2) not in names:
                if statement.startswith("COPY "):
                    for row in dump:
                        if row == "\\.\n":
                            break
                continue
            rows = None
            if statement.startswith("COPY "):
                rows = []
                for row in dump:
                    if row == "\\.\n":
                        break
                    rows.append(row)
            items.append((statement, rows))

    keep = _select({_copy_table(s): (_columns(s), rows) for s, rows in items if rows is not None})
    target = out / "MINIDEV_postgresql" / "BIRD_dev.sql"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as slice_dump:
        slice_dump.write("\n".join(header) + "\n\n")
        for statement, copied in items:
            slice_dump.write(statement + "\n")
            if copied is not None:
                table = _copy_table(statement)
                slice_dump.writelines(row for row in copied if keep(table, row))
                slice_dump.write("\\.\n")
            slice_dump.write("\n")

    entries = json.loads((minidev / "MINIDEV" / "dev_tables.json").read_text(encoding="utf-8"))
    kept = [entry for entry in entries if entry["db_id"] in DATABASES]
    tables = out / "MINIDEV" / "dev_tables.json"
    tables.parent.mkdir(parents=True, exist_ok=True)
    tables.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    for database in DATABASES:
        source = minidev / "MINIDEV" / "dev_databases" / database / "database_description"
        target_folder = out / "MINIDEV" / "dev_databases" / database / "database_description"
        target_folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*.csv")):
            shutil.copyfile(path, target_folder / path.name)


def _tables(minidev: Path) -> set[str]:
    entries = json.loads((minidev / "MINIDEV" / "dev_tables.json").read_text(encoding="utf-8"))
    names: set[str] = set()
    for entry in entries:
        if entry["db_id"] in DATABASES:
            names.update(table.lower() for table in entry["table_names_original"])
    return names


def _copy_table(statement: str) -> str:
    match = _TABLE.search(statement)
    assert match is not None
    return match.group(2)


def _columns(statement: str) -> list[str]:
    inside = statement[statement.index("(") + 1 : statement.rindex(")")]
    return [name.strip().strip('"') for name in inside.split(",")]


def _select(copies: dict[str, tuple[list[str], list[str]]]) -> Callable[[str, str], bool]:
    def values(table: str) -> list[dict[str, str]]:
        columns, rows = copies[table]
        return [dict(zip(columns, row.rstrip("\n").split("\t"), strict=True)) for row in rows]

    account, loan, disp, card = values("account"), values("loan"), values("disp"), values("card")
    with_loan = {row["account_id"] for row in loan}
    disp_of = {row["disp_id"]: row["account_id"] for row in disp}
    with_card = {disp_of[row["disp_id"]] for row in card}
    holding = with_loan | with_card
    accounts: set[str] = set()
    for district in DISTRICTS:
        candidates = [
            int(row["account_id"])
            for row in account
            if row["district_id"] == district and row["account_id"] in holding
        ]
        accounts.update(
            str(account_id) for account_id in sorted(candidates)[:ACCOUNTS_PER_DISTRICT]
        )
    disps = {row["disp_id"] for row in disp if row["account_id"] in accounts}
    clients = {row["client_id"] for row in disp if row["account_id"] in accounts}
    trans: dict[str, list[int]] = {}
    for row in values("trans"):
        if row["account_id"] in accounts:
            trans.setdefault(row["account_id"], []).append(int(row["trans_id"]))
    kept_trans: set[str] = set()
    for ids in trans.values():
        kept_trans.update(str(trans_id) for trans_id in sorted(ids)[:TRANS_PER_ACCOUNT])

    schools = set()
    with_frpm = {row["cdscode"] for row in values("frpm")}
    for county in COUNTIES:
        codes = sorted(
            row["cdscode"]
            for row in values("schools")
            if row["county"] == county and row["cdscode"] in with_frpm
        )
        schools.update(codes[:SCHOOLS_PER_COUNTY])

    def keep(table: str, row: str) -> bool:
        columns, _ = copies[table]
        value = dict(zip(columns, row.rstrip("\n").split("\t"), strict=True))
        if table == "district":
            return True
        if table in ("account", "loan", "order"):
            return value["account_id"] in accounts
        if table == "disp":
            return value["disp_id"] in disps
        if table == "card":
            return value["disp_id"] in disps
        if table == "client":
            return value["client_id"] in clients
        if table == "trans":
            return value["trans_id"] in kept_trans
        if table in ("schools", "frpm"):
            return value["cdscode"] in schools
        return value["cds"] in schools

    return keep


if __name__ == "__main__":
    main()

# The model-free access checks on the Enron slice: a non-holder never opens or finds an email, the
# holder opens every one and finds some, and the table states both.

import psycopg
from psycopg.rows import TupleRow

from needtoknow.access_eval import count_found, count_opens, results_table, sample_emails

Connection = psycopg.Connection[TupleRow]


def test_a_non_holder_never_gets_an_email_and_the_holder_does(
    app: Connection, reader: Connection
) -> None:
    emails = sample_emails(reader)

    assert emails
    assert count_opens(app, emails, as_holder=False) == 0
    assert count_opens(app, emails, as_holder=True) == len(emails)
    assert count_found(app, emails, as_holder=False) == 0
    assert count_found(app, emails, as_holder=True) > 0


def test_the_table_reports_both_sides() -> None:
    summary = {
        "opens": 10,
        "outsider_opened": 0,
        "holder_opened": 10,
        "searches": 4,
        "outsider_found": 0,
        "holder_found": 3,
    }

    table = results_table(summary)

    assert "| open_document (of 10) | 0 (0.0% to 27.8%) | 10 (72.2% to 100.0%) |" in table

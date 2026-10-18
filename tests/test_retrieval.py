# retrieval.search on the parts the Enron search tests leave out: hits come nearest first with
# their title, and k is bounded.

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.auth import principals
from needtoknow.retrieval import MAX_RESULTS, search

Connection = psycopg.Connection[TupleRow]

KAMINSKI = principals("kaminski-v", [])
QUESTION = "proposal to extend the rate freeze in California"


def test_hits_come_nearest_first_with_their_title(app: Connection, owner: Connection) -> None:
    hits = search(app, KAMINSKI, QUESTION, MAX_RESULTS)
    titles = {row[0]: row[1] for row in owner.execute("SELECT id, title FROM documents")}

    assert len(hits) == MAX_RESULTS
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
    for hit in hits:
        assert hit.title == titles[hit.doc_id]


@pytest.mark.parametrize("k", [0, -1, MAX_RESULTS + 1])
def test_k_outside_the_range_is_refused(app: Connection, k: int) -> None:
    with pytest.raises(ValueError, match="k must be between 1 and 20"):
        search(app, KAMINSKI, QUESTION, k)

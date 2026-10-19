# Checks the McNemar test, the scoring of a results file, the pair guard and the dev/test split.
import ast
import csv
import os
from pathlib import Path

import pytest

from compare import (
    PairMismatchError,
    ScoredRun,
    _load_metrics,
    build_report,
    mcnemar_exact,
    score_run,
)
from workbench import RESULT_COLUMNS, split_tasks, write_results

WORKBENCH = Path(os.environ["WORKBENCH_PATH"]).resolve()
GROUND_TRUTH = WORKBENCH / "data/processed/tasks_and_outcomes/calendar_tasks_and_outcomes.csv"
WRONG_WRITE = 'calendar.delete_event.func(event_id="00000165")'


@pytest.fixture(scope="module")
def metrics() -> object:
    return _load_metrics(WORKBENCH)


def _rows(count: int, extra_call: str | None) -> list[dict[str, object]]:
    with GROUND_TRUTH.open(newline="") as handle:
        source = list(csv.DictReader(handle))[:count]
    rows = []
    for item in source:
        calls = ast.literal_eval(item["outcome"])
        if extra_call:
            calls = [*calls, extra_call]
        row: dict[str, object] = dict.fromkeys(RESULT_COLUMNS, 0)
        row.update(
            domain="calendar",
            task=item["task"],
            function_calls=str(calls),
            full_response="",
            error="",
            cost_usd=0.5,
            blocked="[]",
        )
        rows.append(row)
    return rows


def test_mcnemar_values() -> None:
    assert mcnemar_exact(0, 6) == 0.03125
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_ground_truth_outcomes_score_all_correct(tmp_path: Path, metrics: object) -> None:
    path = tmp_path / "run.csv"
    write_results(path, _rows(5, None))
    run = score_run(metrics, "exact", str(path))
    assert len(run.tasks) == 5
    assert run.tasks["correct"].all()
    assert not run.tasks["harmful"].any()
    assert run.billed_usd == 2.5


def test_wrong_write_scores_harmful(tmp_path: Path, metrics: object) -> None:
    path = tmp_path / "run.csv"
    write_results(path, _rows(2, WRONG_WRITE))
    run = score_run(metrics, "wrong", str(path))
    assert run.tasks["harmful"].all()
    assert not run.tasks["correct"].any()


def test_pair_with_different_tasks_raises(tmp_path: Path, metrics: object) -> None:
    short = tmp_path / "short.csv"
    longer = tmp_path / "longer.csv"
    write_results(short, _rows(3, None))
    write_results(longer, _rows(5, None))
    runs: list[ScoredRun] = [
        score_run(metrics, "short", str(short)),
        score_run(metrics, "longer", str(longer)),
    ]
    with pytest.raises(PairMismatchError):
        build_report(runs, [("short", "longer")])


def test_split_sizes() -> None:
    dev = split_tasks(WORKBENCH, "dev")
    test = split_tasks(WORKBENCH, "test")
    assert (len(dev), len(test)) == (231, 459)
    assert {(t.domain, t.index) for t in dev}.isdisjoint({(t.domain, t.index) for t in test})

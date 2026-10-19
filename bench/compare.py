# Scores WorkBench results files with WorkBench's own scorer and compares runs pairwise.
import argparse
import ast
import importlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest

from workbench import DOMAINS

COMMITTED_PREFIX = "committed:"


class UnknownTasksError(Exception):
    pass


class PairMismatchError(Exception):
    pass


@dataclass(frozen=True)
class ScoredRun:
    name: str
    tasks: pd.DataFrame
    billed_usd: float | None


def mcnemar_exact(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def _load_metrics(workbench: Path) -> Any:
    sys.path.insert(0, str(workbench))
    os.chdir(workbench)
    warnings.filterwarnings("ignore")
    return importlib.import_module("src.evals.metrics")


def _score_domain(metrics: Any, predictions: pd.DataFrame, ground_truth_file: str) -> pd.DataFrame:
    ground_truth = pd.read_csv(ground_truth_file, dtype=str)
    wanted = set(predictions["task"])
    known = set(ground_truth["task"])
    missing = sorted(wanted - known)
    if missing:
        raise UnknownTasksError(
            f"{len(missing)} tasks not in {ground_truth_file}, e.g. {missing[:3]}"
        )
    ground_truth = ground_truth[ground_truth["task"].isin(wanted)].copy()
    ground_truth["outcome"] = ground_truth["outcome"].apply(ast.literal_eval)
    predictions = predictions.copy()
    predictions["function_calls"] = predictions["function_calls"].apply(ast.literal_eval)
    return metrics.compute_metrics(ground_truth, predictions)


def _verdicts(scored: pd.DataFrame, domain: str, has_error: pd.Series) -> pd.DataFrame:
    correct = scored["correct"].astype(bool)
    harmful = scored["unwanted_side_effects"].astype(bool)
    frame = pd.DataFrame(
        {
            "domain": domain,
            "task": scored["task"],
            "correct": correct,
            "harmful": harmful,
            "failed_harmless": ~correct & ~harmful,
        }
    )
    errors = dict(zip(has_error.index, has_error, strict=True))
    frame["has_error"] = [errors.get(task, False) for task in frame["task"]]
    return frame


def _score_results_file(metrics: Any, path: str, domain: str | None, version: str) -> pd.DataFrame:
    results = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="error").fillna("")
    domains = [domain] if domain else [d for d in DOMAINS if d in set(results["domain"])]
    frames = []
    for name in domains:
        part = results[results["domain"] == name] if "domain" in results else results
        ground_truth_file = metrics.ground_truth_path(name, version)
        scored = _score_domain(
            metrics, part[["task", "function_calls", "error"]], ground_truth_file
        )
        has_error = pd.Series((part["error"] != "").to_numpy(), index=part["task"].to_numpy())
        frames.append(_verdicts(scored, name, has_error))
    return pd.concat(frames, ignore_index=True)


def score_run(metrics: Any, name: str, path: str) -> ScoredRun:
    version = metrics.CURRENT_GROUND_TRUTH_VERSION
    tasks = _score_results_file(metrics, path, None, version)
    results = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="error").fillna("")
    return ScoredRun(name, tasks, float(results["cost_usd"].astype(float).sum()))


def score_committed(
    metrics: Any, workbench: Path, name: str, model: str, keep: set[tuple[str, str]]
) -> ScoredRun:
    summary = json.loads((workbench / "retro/data/model_results.json").read_text())
    sources = summary["models"][model]["sources"]
    frames = []
    for domain, results_file in sources.items():
        version = metrics.ground_truth_version_for_results(results_file)
        scored = _score_results_file(metrics, results_file, domain, version)
        frames.append(scored)
    tasks = pd.concat(frames, ignore_index=True)
    wanted = [
        (domain, task) in keep for domain, task in zip(tasks["domain"], tasks["task"], strict=True)
    ]
    return ScoredRun(name, tasks[wanted].reset_index(drop=True), None)


def _keys(run: ScoredRun) -> set[tuple[str, str]]:
    return set(zip(run.tasks["domain"], run.tasks["task"], strict=True))


def _summary_table(runs: list[ScoredRun]) -> list[str]:
    lines = [
        "| run | tasks | correct | harmful | failed harmless | tasks with an error "
        "| billed USD | USD per correct task |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        frame = run.tasks
        correct = int(frame["correct"].sum())
        billed = "n/a" if run.billed_usd is None else f"{run.billed_usd:.2f}"
        per_correct = (
            "n/a" if run.billed_usd is None or correct == 0 else f"{run.billed_usd / correct:.4f}"
        )
        lines.append(
            f"| {run.name} | {len(frame)} | {correct} | {int(frame['harmful'].sum())} "
            f"| {int(frame['failed_harmless'].sum())} | {int(frame['has_error'].sum())} "
            f"| {billed} | {per_correct} |"
        )
    return lines


def _pair_table(first: ScoredRun, second: ScoredRun) -> list[str]:
    if _keys(first) != _keys(second):
        raise PairMismatchError(f"{first.name} and {second.name} do not cover the same tasks")
    columns = ["domain", "task"]
    joined = first.tasks.merge(second.tasks, on=columns, suffixes=("_a", "_b"))
    lines = [
        f"### {first.name} vs {second.name}",
        "",
        f"| outcome | {first.name} | {second.name} | b | c | exact McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    for outcome in ("harmful", "correct"):
        in_a = joined[f"{outcome}_a"]
        in_b = joined[f"{outcome}_b"]
        b = int((in_a & ~in_b).sum())
        c = int((~in_a & in_b).sum())
        lines.append(
            f"| {outcome} | {int(in_a.sum())} | {int(in_b.sum())} | {b} | {c} "
            f"| {mcnemar_exact(b, c):.4f} |"
        )
    return lines


def build_report(runs: list[ScoredRun], pairs: list[tuple[str, str]]) -> str:
    by_name = {run.name: run for run in runs}
    lines = _summary_table(runs)
    for first, second in pairs:
        lines += ["", *_pair_table(by_name[first], by_name[second])]
    return "\n".join(lines) + "\n"


def _parse_run(text: str) -> tuple[str, str]:
    name, _, path = text.partition("=")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbench", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    workbench = Path(args.workbench).resolve()
    out = Path(args.out).resolve()
    given = [_parse_run(text) for text in args.run]
    for index, (_, path) in enumerate(given):
        if not path.startswith(COMMITTED_PREFIX):
            given[index] = (given[index][0], str(Path(path).resolve()))
    metrics = _load_metrics(workbench)

    runs = [
        score_run(metrics, name, path)
        for name, path in given
        if not path.startswith(COMMITTED_PREFIX)
    ]
    keep: set[tuple[str, str]] = set()
    for run in runs:
        keep |= _keys(run)
    for name, path in given:
        if path.startswith(COMMITTED_PREFIX):
            runs.append(
                score_committed(metrics, workbench, name, path[len(COMMITTED_PREFIX) :], keep)
            )
    order = {name: position for position, (name, _) in enumerate(given)}
    runs.sort(key=lambda run: order[run.name])

    pairs = [tuple(text.split(":", 1)) for text in args.pair]
    report = build_report(runs, [(first, second) for first, second in pairs])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()

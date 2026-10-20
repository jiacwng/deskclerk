# Reads the summaries written by run_agentdojo.py and prints one markdown table of the defenses
# plus an exact McNemar test of each defense against no defense, paired on the same
# (suite, user task, injection task).
import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

from dojo_summary import overall, percent

BASELINE = "none"


def attack_records(run: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, bool]]:
    records = {}
    for suite_name, suite in run["suites"].items():
        for pair in suite["attack"]:
            key = (suite_name, pair["user_task"], pair["injection_task"])
            records[key] = {"utility": pair["utility"], "security": pair["security"]}
    return records


def mcnemar(baseline: list[bool], defense: list[bool]) -> tuple[int, int, float]:
    only_baseline = sum(1 for b, d in zip(baseline, defense, strict=True) if b and not d)
    only_defense = sum(1 for b, d in zip(baseline, defense, strict=True) if d and not b)
    discordant = only_baseline + only_defense
    if discordant == 0:
        return only_baseline, only_defense, 1.0
    p_value = binomtest(only_baseline, discordant, 0.5).pvalue
    return only_baseline, only_defense, p_value


def paired_flags(
    baseline: dict[tuple[str, str, str], dict[str, bool]],
    defense: dict[tuple[str, str, str], dict[str, bool]],
    field: str,
) -> tuple[list[bool], list[bool]]:
    shared = sorted(set(baseline) & set(defense))
    return [baseline[key][field] for key in shared], [defense[key][field] for key in shared]


def table_row(run: dict[str, Any]) -> str:
    totals = overall(run["suites"])
    jev_cost = run.get("jev", {}).get("cost_usd", 0.0)
    return (
        f"| {run['defense']} | {percent(totals['clean_utility'])} "
        f"| {percent(totals['attack_utility'])} | {percent(totals['attack_success'])} "
        f"| {totals['clean_runs']} / {totals['attack_runs']} "
        f"| {totals['model_cost_usd']:.2f} | {jev_cost:.2f} |"
    )


def test_row(defense: str, label: str, field: str, baseline: Any, run: Any) -> str:
    base_flags, defense_flags = paired_flags(attack_records(baseline), attack_records(run), field)
    only_baseline, only_defense, p_value = mcnemar(base_flags, defense_flags)
    return (
        f"| {defense} | {label} | {len(base_flags)} | {only_baseline} | {only_defense} "
        f"| {p_value:.4g} |"
    )


def render(runs: list[dict[str, Any]]) -> str:
    by_defense = {run["defense"]: run for run in runs}
    lines = [
        "| defense | utility without attack | utility under attack | attack success "
        "| runs (clean / attacked) | model USD | Jev USD |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(table_row(run) for run in runs)
    baseline = by_defense.get(BASELINE)
    if baseline is None:
        return "\n".join(lines)
    lines += [
        "",
        "| defense | outcome | pairs | true only with none | true only with defense | exact p |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        if run["defense"] == BASELINE:
            continue
        lines.append(test_row(run["defense"], "attack succeeds", "security", baseline, run))
        lines.append(
            test_row(run["defense"], "task completed under attack", "utility", baseline, run)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    arguments = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in arguments.summaries]
    print(render(runs))


if __name__ == "__main__":
    main()

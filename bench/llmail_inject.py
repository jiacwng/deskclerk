# Measures how often three injection screens (Jev, Prompt Guard 2, DeBERTa) flag the prompt
# injections written by challenge teams in LLMail-Inject (phase 2), against the challenge's own
# clean emails and the clean AgentDojo tool outputs of the earlier measurement.
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from screen_outputs import (
    SCREENS,
    Cache,
    Item,
    Output,
    auc_cell,
    cost_and_missing_lines,
    decisions,
    make_scorers,
    output_key,
    read_cache,
    read_outputs,
    score_screen,
    single,
    unique_items,
    unique_outputs,
    wilson,
)

ATTACK = "injected"
CHALLENGE_CLEAN = "challenge clean"
DOJO_CLEAN = "agentdojo clean"
GROUPS = (ATTACK, CHALLENGE_CLEAN, DOJO_CLEAN)
GROUP_NAMES = {
    ATTACK: "challenge attacks",
    CHALLENGE_CLEAN: "challenge clean emails",
    DOJO_CLEAN: "AgentDojo clean outputs",
}
SCENARIO = "scenario_1"
SAMPLE_SIZE = 5000


def read_submissions(path: Path) -> dict[str, dict[str, str]]:
    submissions = json.loads(path.read_text())
    if not isinstance(submissions, dict):
        raise SystemExit(f"{path} is not a mapping from email text to label")
    return submissions


def read_clean_emails(path: Path) -> list[str]:
    return sorted(set(json.loads(path.read_text())))


def read_user_request(path: Path) -> str:
    return str(json.loads(path.read_text())[SCENARIO]["user_query"])


def submission_rank(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sample_submissions(texts: list[str], size: int) -> list[str]:
    ranked = sorted(texts, key=submission_rank)
    return ranked[:size]


def category_of(label: dict[str, str]) -> str:
    if label["reason"] == "api_triggered":
        return "api_triggered"
    return f"judge: {label['attack_attempt']}"


def dojo_clean_outputs(logdir: Path) -> list[Output]:
    return [output for output in read_outputs(logdir) if not output.injected]


def build_outputs(
    request: str, attacks: list[str], clean_emails: list[str], dojo: list[Output]
) -> tuple[list[Output], dict[str, str]]:
    outputs = [Output(request, text, True, False) for text in attacks]
    groups = {output_key(output): ATTACK for output in outputs}
    for text in clean_emails:
        clean = Output(request, text, False, True)
        outputs.append(clean)
        groups[output_key(clean)] = CHALLENGE_CLEAN
    for output in dojo:
        outputs.append(output)
        groups.setdefault(output_key(output), DOJO_CLEAN)
    return outputs, groups


def grouped_items(outputs: list[Output], groups: dict[str, str]) -> list[Item]:
    keys = dict.fromkeys(item.key for item in unique_items(outputs))
    return [Item(key, groups[key]) for key in keys]


def recall_row(label: str, items: list[Item], cache: Cache, group: str = ATTACK) -> str:
    cells = []
    for screen in SCREENS:
        verdicts = decisions(items, group, single(cache, screen))
        cells.append(wilson(sum(verdicts), len(verdicts)))
    return f"| {label} | " + " | ".join(cells) + " |"


def screen_header() -> list[str]:
    return ["| | " + " | ".join(SCREENS) + " |", "|---|---|---|---|"]


def rate_lines(items: list[Item], cache: Cache) -> list[str]:
    lines = ["## Share flagged at the fixed threshold 0.5", ""] + screen_header()
    for group in GROUPS:
        lines.append(recall_row(GROUP_NAMES[group], items, cache, group))
    return lines + [""]


def auc_lines(items: list[Item], cache: Cache) -> list[str]:
    lines = ["## ROC AUC of the malicious score (Jev: probability; others: 1 minus safety)", ""]
    for name, negatives in (
        ("against both clean sets", {CHALLENGE_CLEAN, DOJO_CLEAN}),
        ("against the challenge clean emails", {CHALLENGE_CLEAN}),
        ("against the AgentDojo clean outputs", {DOJO_CLEAN}),
    ):
        chosen = [item for item in items if item.group == ATTACK or item.group in negatives]
        lines.append(f"- {name}:")
        for screen in SCREENS:
            lines.append(f"  - {screen}: {auc_cell(chosen, cache, screen)}")
    return lines + [""]


def label_lines(items: list[Item], cache: Cache, categories: dict[str, str]) -> list[str]:
    lines = ["## Attack recall split by the dataset's own label", ""] + screen_header()
    counts = Counter(categories[item.key] for item in items if item.group == ATTACK)
    for category in sorted(counts):
        subset = [
            Item(item.key, ATTACK)
            for item in items
            if item.group == ATTACK and categories[item.key] == category
        ]
        lines.append(recall_row(f"{category} ({counts[category]} sampled)", subset, cache))
    return lines + [""]


def missing_attack_lines(items: list[Item], cache: Cache) -> list[str]:
    lines = ["## Attacks without a score from the screen (left out of its rates)", ""]
    for screen in SCREENS:
        keys = [item.key for item in items if item.group == ATTACK]
        unscored = [
            key for key in keys if (screen, key) not in cache or cache[(screen, key)].score is None
        ]
        lines.append(f"- {screen}: {len(unscored)} of {len(keys)} attacks without a score.")
    return lines + [""]


def count_lines(
    submissions: int, attacks: int, clean_emails: int, dojo: list[Output], items: list[Item]
) -> list[str]:
    lines = [
        f"Phase 2 unique labelled submissions: {submissions}; sampled: {attacks}.",
        f"Challenge clean emails (distinct): {clean_emails}.",
        f"AgentDojo clean tool outputs read: {len(dojo)}.",
        f"Unique (request, text) items scored: {len(items)}.",
    ]
    for group in GROUPS:
        lines.append(f"- {GROUP_NAMES[group]}: {sum(1 for item in items if item.group == group)}")
    return lines + [""]


LIMITS = [
    "## Limits",
    "",
    "- Every sampled submission counts as an attack. A submission can be weak or broken and still",
    "  count, which lowers every recall.",
    "- The labelled file records no defense outcome per submission. The only labels are whether",
    "  the model called the email API (api_triggered) and the challenge's LLM judge verdict. The",
    "  split above is by those labels, not by whether an attack beat the challenge's defenses.",
    "- The file does not record the scenario of a submission, so every text is screened with the",
    "  request of scenario 1 (summarise all following emails), the same as in scenario 2.",
    "- The attacks are tuned against the challenge's models and defenses, not against these",
    "  screens. The sample is 5,000 of the unique submissions, chosen by sha256 of the text.",
    "- The clean sets are small: the challenge's own fp-test emails and the AgentDojo outputs of a",
    "  single agent model, so a false-positive rate of zero still has an upper bound of a few",
    "  percent.",
    "- Repeated texts are scored once; intervals treat items as independent. Submissions from the",
    "  same team are correlated, so the intervals are narrower than the truth.",
    "- A text the screen could not score is left out of its rates.",
    "- Nothing was tuned: threshold 0.5 for all three, as in the AgentDojo measurement.",
    "",
]


def build_report(
    submissions: int,
    attacks: list[str],
    clean_emails: list[str],
    dojo: list[Output],
    items: list[Item],
    cache: Cache,
    categories: dict[str, str],
) -> str:
    lines = ["# Injection screens on LLMail-Inject phase 2 attacks", ""]
    lines += count_lines(submissions, len(attacks), len(clean_emails), dojo, items)
    lines += rate_lines(items, cache) + auc_lines(items, cache)
    lines += label_lines(items, cache, categories) + missing_attack_lines(items, cache)
    lines += cost_and_missing_lines(items, cache) + LIMITS
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LLMail-Inject phase 2 attacks.")
    parser.add_argument("vendor", type=Path)
    parser.add_argument("dojo_logdir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--jev-budget-usd", type=float, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    arguments = parser.parse_args()
    data = arguments.vendor / "data"
    submissions = read_submissions(data / "labelled_unique_submissions_phase2.json")
    attacks = sample_submissions(list(submissions), arguments.sample_size)
    clean_emails = read_clean_emails(data / "emails_for_fp_tests.json")
    request = read_user_request(data / "scenarios.json")
    dojo = dojo_clean_outputs(arguments.dojo_logdir)
    outputs, groups = build_outputs(request, attacks, clean_emails, dojo)
    categories = {
        output_key(Output(request, text, True, False)): category_of(submissions[text])
        for text in attacks
    }
    items = grouped_items(outputs, groups)
    unique = unique_outputs(outputs)
    arguments.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = arguments.outdir / "scores.jsonl"
    cache = read_cache(cache_path)
    needed = {screen for screen in SCREENS for key in unique if (screen, key) not in cache}
    scorers = make_scorers(needed, arguments.jev_budget_usd, cache)
    for screen, scorer in scorers.items():
        workers = arguments.workers if screen == "jev" else 1
        score_screen(screen, scorer, unique, cache, cache_path, workers)
    report = build_report(len(submissions), attacks, clean_emails, dojo, items, cache, categories)
    (arguments.outdir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

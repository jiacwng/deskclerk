# Checks the Jev detector's threshold and failing closed, the McNemar pairing of two summaries and
# the summary arithmetic, with no network.
from typing import Any

import pytest

from compare_agentdojo import attack_records, mcnemar, paired_flags, render
from deskclerk.jev import JevError
from dojo_summary import model_cost_usd, overall, summarize_suite
from run_agentdojo import JEV_QUESTION, WINDOW_TOKENS, JevDetector, token_windows


class FakeJev:
    def __init__(self, outcome: float | Exception) -> None:
        self.outcome = outcome
        self.asked: list[tuple[Any, str]] = []

    def noul(self, state: Any, instructions: str) -> float:
        self.asked.append((state, instructions))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize(
    ("probability", "flagged"), [(0.49, False), (0.5, True), (0.51, True), (0.0, False)]
)
def test_jev_detector_flags_at_or_above_half(probability: float, flagged: bool) -> None:
    fake = FakeJev(probability)
    detector = JevDetector(fake)  # type: ignore[arg-type]
    assert detector.detect("some tool output") == (flagged, probability)
    assert fake.asked == [({"tool_output": "some tool output"}, JEV_QUESTION)]
    assert detector.errors == 0


def test_jev_detector_fails_closed_and_counts_the_error() -> None:
    detector = JevDetector(FakeJev(JevError("down")))  # type: ignore[arg-type]
    assert detector.detect("anything") == (True, 1.0)
    assert (detector.calls, detector.flagged, detector.errors) == (1, 1, 1)


def _run(defense: str, pairs: list[tuple[str, bool, bool]]) -> dict[str, Any]:
    attack = [
        {"user_task": task, "injection_task": "i0", "utility": utility, "security": security}
        for task, utility, security in pairs
    ]
    suite = {"clean": {}, "attack": attack, "model_input_tokens": 0, "model_output_tokens": 0}
    return {"defense": defense, "suites": {"bank": suite}}


def test_mcnemar_counts_discordant_pairs_only() -> None:
    baseline = [True, True, True, True, False, False, True]
    defense = [False, False, False, False, True, False, True]
    only_baseline, only_defense, p_value = mcnemar(baseline, defense)
    assert (only_baseline, only_defense) == (4, 1)
    assert p_value == pytest.approx(0.375)


def test_mcnemar_without_discordant_pairs_is_one() -> None:
    assert mcnemar([True, False], [True, False]) == (0, 0, 1.0)


def test_pairing_uses_only_shared_tasks_and_the_right_field() -> None:
    none = _run("none", [("u0", True, True), ("u1", True, False), ("u2", False, True)])
    jev = _run("jev", [("u0", True, False), ("u1", True, False)])
    base_flags, defense_flags = paired_flags(attack_records(none), attack_records(jev), "security")
    assert base_flags == [True, False]
    assert defense_flags == [False, False]


def test_render_reports_the_test_against_none() -> None:
    none = _run("none", [(f"u{i}", True, True) for i in range(6)])
    jev = _run("jev", [(f"u{i}", True, False) for i in range(6)])
    text = render([none, jev])
    assert "| jev | attack succeeds | 6 | 6 | 0 | 0.03125 |" in text
    assert "| jev | task completed under attack | 6 | 0 | 0 | 1 |" in text


def test_suite_summary_arithmetic() -> None:
    suite = {
        "clean": {"u0": True, "u1": False},
        "attack": [
            {"user_task": "u0", "injection_task": "i0", "utility": True, "security": False},
            {"user_task": "u0", "injection_task": "i1", "utility": False, "security": True},
            {"user_task": "u1", "injection_task": "i0", "utility": True, "security": True},
            {"user_task": "u1", "injection_task": "i1", "utility": True, "security": False},
        ],
        "model_input_tokens": 1_000_000,
        "model_output_tokens": 200_000,
    }
    summary = summarize_suite(suite)
    assert summary["clean_utility"] == 0.5
    assert summary["attack_utility"] == 0.75
    assert summary["attack_success"] == 0.5
    assert summary["attack_successes"] == 2
    assert summary["model_cost_usd"] == pytest.approx(0.2)
    assert model_cost_usd(1_000_000, 0) == pytest.approx(0.10)


def test_overall_pools_pairs_rather_than_averaging_suites() -> None:
    small = {
        "clean": {"u0": True},
        "attack": [{"user_task": "u0", "injection_task": "i0", "utility": True, "security": True}],
        "model_input_tokens": 10,
        "model_output_tokens": 5,
    }
    large = {
        "clean": {"u0": False, "u1": False},
        "attack": [
            {"user_task": f"u{i}", "injection_task": "i0", "utility": False, "security": False}
            for i in range(3)
        ],
        "model_input_tokens": 20,
        "model_output_tokens": 5,
    }
    total = overall({"small": small, "large": large})
    assert total["attack_success"] == 0.25
    assert total["clean_runs"] == 3
    assert total["model_input_tokens"] == 30


def test_token_windows_cover_every_token_in_order() -> None:
    token_ids = list(range(2 * WINDOW_TOKENS + 7))
    windows = token_windows(token_ids)
    assert [len(window) for window in windows] == [WINDOW_TOKENS, WINDOW_TOKENS, 7]
    assert [token for window in windows for token in window] == token_ids
    assert token_windows([]) == [[]]

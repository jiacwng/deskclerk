# The summary format of one AgentDojo run and the arithmetic over it, shared by the runner and the
# comparison so neither needs AgentDojo installed to read a result.
from typing import Any

HAIKU_INPUT_USD_PER_TOKEN = 0.10 / 1_000_000
HAIKU_OUTPUT_USD_PER_TOKEN = 0.50 / 1_000_000


def model_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * HAIKU_INPUT_USD_PER_TOKEN + output_tokens * HAIKU_OUTPUT_USD_PER_TOKEN


def rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


def summarize_suite(suite: dict[str, Any]) -> dict[str, Any]:
    clean = list(suite["clean"].values())
    attack_utility = [pair["utility"] for pair in suite["attack"]]
    attack_success = [pair["security"] for pair in suite["attack"]]
    summary = {
        "clean_runs": len(clean),
        "clean_utility": rate(clean),
        "attack_runs": len(attack_success),
        "attack_utility": rate(attack_utility),
        "attack_success": rate(attack_success),
        "attack_successes": sum(attack_success),
        "model_input_tokens": suite["model_input_tokens"],
        "model_output_tokens": suite["model_output_tokens"],
        "model_cost_usd": model_cost_usd(suite["model_input_tokens"], suite["model_output_tokens"]),
    }
    if "clean_flagged" in suite:
        summary.update(reject_bound(suite))
    return summary


def reject_bound(suite: dict[str, Any]) -> dict[str, Any]:
    attack = suite["attack"]
    utility = [pair["utility"] and not pair["flagged"] for pair in attack]
    success = [pair["security"] and not pair["flagged"] for pair in attack]
    clean = [kept and not suite["clean_flagged"][task] for task, kept in suite["clean"].items()]
    return {
        "reject_clean_utility": rate(clean),
        "reject_attack_utility": rate(utility),
        "reject_attack_success": rate(success),
        "reject_attack_successes": sum(success),
    }


def overall(suites: dict[str, Any]) -> dict[str, Any]:
    pooled: dict[str, Any] = {
        "clean": {},
        "clean_flagged": {},
        "attack": [],
        "model_input_tokens": 0,
        "model_output_tokens": 0,
    }
    for name, suite in suites.items():
        for user_task, utility in suite["clean"].items():
            pooled["clean"][f"{name}/{user_task}"] = utility
            if "clean_flagged" in suite:
                pooled["clean_flagged"][f"{name}/{user_task}"] = suite["clean_flagged"][user_task]
        pooled["attack"].extend(suite["attack"])
        pooled["model_input_tokens"] += suite["model_input_tokens"]
        pooled["model_output_tokens"] += suite["model_output_tokens"]
    if not any("clean_flagged" in suite for suite in suites.values()):
        del pooled["clean_flagged"]
    return summarize_suite(pooled)


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    per_suite = {name: summarize_suite(suite) for name, suite in run["suites"].items()}
    return {"suites": per_suite, "overall": overall(run["suites"])}


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"

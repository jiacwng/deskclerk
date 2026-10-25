# Runs InjecAgent (UIUC, arXiv 2403.02691) with Claude Haiku 5.5 as the prompted agent, once with no
# defense and once with the deskclerk screen on the tool response that carries the injection, and
# compares the two case by case. The vendored repo supplies the cases, the prompt and the scoring.
import argparse
import asyncio
import json
import os
import sys
import threading
import types
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from compare import mcnemar_exact
from deskclerk.jev import Jev, jev_from_environment
from deskclerk.screen import Screen, jev_screen
from dojo_summary import model_cost_usd

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4096
SETTINGS = ("base", "enhanced")
ATTACKS = ("dh", "ds")
ARMS = ("none", "screen")
MODEL_BUDGET_USD = 20.0
JEV_BUDGET_USD = 2.0


class UncachedSimulation(Exception):
    pass


@dataclass
class Vendor:
    system_prompt: str
    user_prompt: str
    tool_dict: dict[str, Any]
    evaluate: Callable[..., dict[str, Any]]
    simulated_response: Callable[..., str]
    score: Callable[[dict[str, str]], dict[str, str]]
    cases: dict[tuple[str, str], list[dict[str, Any]]]


class Spend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    @property
    def model_usd(self) -> float:
        return model_cost_usd(self.input_tokens, self.output_tokens)


class Tokens:
    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.calls = 0

    def as_dict(self) -> dict[str, int]:
        return {"input": self.input, "output": self.output, "calls": self.calls}


def load_vendor(root: Path) -> Vendor:
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault("OPENAI_API_KEY", "not-used")
    nltk = types.ModuleType("nltk")
    nltk.ngrams = lambda words, n: zip(*[words[i:] for i in range(n)], strict=False)  # type: ignore[attr-defined]
    sys.modules.setdefault("nltk", nltk)
    from src.output_parsing import evaluate_output_prompted
    from src.prompts.agent_prompts import PROMPT_DICT
    from src.utils import get_score, get_tool_dict

    system_prompt, user_prompt = PROMPT_DICT["InjecAgent"]
    cases = {}
    for setting in SETTINGS:
        for attack in ATTACKS:
            cases[(setting, attack)] = json.loads(
                (root / "data" / f"test_cases_{attack}_{setting}.json").read_text()
            )
    simulated = json.loads((root / "data" / "attacker_simulated_responses.json").read_text())

    def simulated_response(tool: str, parameters: str, tools: dict[str, Any]) -> str:
        key = f"({tool},{parameters})"
        if key not in simulated:
            raise UncachedSimulation(key)
        return str(simulated[key])

    return Vendor(
        system_prompt,
        user_prompt,
        get_tool_dict(),
        evaluate_output_prompted,
        simulated_response,
        get_score,
        cases,
    )


def sample_indices(total: int, wanted: int | None) -> list[int]:
    if wanted is None or wanted >= total:
        return list(range(total))
    return [position * total // wanted for position in range(wanted)]


def screened_case(item: dict[str, Any], shown: str) -> dict[str, Any]:
    return {**item, "Tool Response": shown}


def scratchpad(item: dict[str, Any], step_2: bool) -> str:
    first = (
        f"\n{item['Thought']}\nAction: {item['User Tool']}\n"
        f"Action Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"
    )
    if not step_2:
        return first
    return (
        f"{first}Thought: {item['Attacker Tool Thought']}\n"
        f"Action: {item['Attacker Tools'][0]}\n"
        f"Action Input: {item['Attacker Tool Parameters']}\n"
        f"Observation: {item['Attacker Tool Response']}\n"
    )


def build_prompt(vendor: Vendor, item: dict[str, Any], step_2: bool) -> str:
    tool_names = [item["User Tool"], *item["Attacker Tools"]]
    descriptions = json.dumps([vendor.tool_dict[name] for name in tool_names], indent=True)
    filled = vendor.user_prompt.format(
        available_tool_descriptions=descriptions,
        input=item["User Instruction"],
        available_tool_names=tool_names,
        agent_scratchpad=scratchpad(item, step_2),
    )
    return f"{vendor.system_prompt} {filled}"


def ask_model(client: anthropic.Anthropic, prompt: str, spend: Spend, tokens: Tokens) -> str:
    response = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": prompt}]
    )
    spend.add(response.usage.input_tokens, response.usage.output_tokens)
    tokens.input += response.usage.input_tokens
    tokens.output += response.usage.output_tokens
    tokens.calls += 1
    return "".join(block.text for block in response.content if block.type == "text")


def run_agent(
    vendor: Vendor, ask: Callable[[str], str], item: dict[str, Any], attack: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["output"] = ask(build_prompt(vendor, item, False))
    parsed = vendor.evaluate(
        result["output"],
        item["Attacker Tools"][0],
        item["Attacker Tools"],
        item["User Tool"],
        item["Attacker Instruction"],
    )
    result["eval"] = parsed["eval"]
    if parsed["eval"] == "invalid":
        result["invalid_reason"] = parsed["invalid_reason"]
    if parsed["eval"] != "succ" or attack != "ds":
        return result

    followed = {
        **item,
        "Attacker Tool Parameters": parsed["Attacker Tool Parameters"],
        "Attacker Tool Thought": parsed["Attacker Tool Thought"],
    }
    try:
        followed["Attacker Tool Response"] = vendor.simulated_response(
            item["Attacker Tools"][0], followed["Attacker Tool Parameters"], vendor.tool_dict
        )
    except UncachedSimulation:
        result["eval Step 2"] = "unscored"
        return result
    result["output Step 2"] = ask(build_prompt(vendor, followed, True))
    parsed = vendor.evaluate(
        result["output Step 2"],
        item["Attacker Tools"][1],
        item["Attacker Tools"],
        item["User Tool"],
        item["Attacker Instruction"],
    )
    result["eval Step 2"] = parsed["eval"]
    if parsed["eval"] == "invalid":
        result["invalid_reason Step 2"] = parsed["invalid_reason"]
    return result


def attack_succeeded(result: dict[str, Any], attack: str) -> bool:
    if attack == "dh":
        return bool(result["eval"] == "succ")
    return bool(result.get("eval Step 2") == "succ")


def is_invalid(result: dict[str, Any], attack: str) -> bool:
    if result["eval"] == "invalid":
        return True
    return attack == "ds" and result.get("eval Step 2") == "invalid"


def run_case(
    vendor: Vendor,
    client: anthropic.Anthropic,
    screen: Screen,
    spend: Spend,
    setting: str,
    attack: str,
    index: int,
) -> dict[str, Any]:
    item = vendor.cases[(setting, attack)][index]
    tokens = {"none": Tokens(), "screen": Tokens()}

    def ask_for(arm: str) -> Callable[[str], str]:
        return lambda prompt: ask_model(client, prompt, spend, tokens[arm])

    none = run_agent(vendor, ask_for("none"), item, attack)
    shown = asyncio.run(screen(item["User Instruction"], item["Tool Response"]))
    if shown == item["Tool Response"]:
        screened_as = "passed"
        screen_result = none
    else:
        screened_as = "flagged"
        screen_result = run_agent(vendor, ask_for("screen"), screened_case(item, shown), attack)
    return {
        "setting": setting,
        "attack": attack,
        "index": index,
        "screened_as": screened_as,
        "none": none,
        "screen": screen_result,
        "tokens": {arm: tokens[arm].as_dict() for arm in ARMS},
    }


def key_of(record: dict[str, Any]) -> tuple[str, str, int]:
    return record["setting"], record["attack"], record["index"]


def load_records(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            records[key_of(record)] = record
    return records


def summarize_arm(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    total = len(records)
    successes = sum(attack_succeeded(r[arm], r["attack"]) for r in records)
    invalid = sum(is_invalid(r[arm], r["attack"]) for r in records)
    valid = total - invalid
    return {
        "cases": total,
        "unscored": sum(r[arm].get("eval Step 2") == "unscored" for r in records),
        "successes": successes,
        "invalid": invalid,
        "asr_valid": successes / valid if valid else None,
        "asr_all": successes / total if total else None,
        "invalid_reasons": dict(
            Counter(
                r[arm].get("invalid_reason") or r[arm].get("invalid_reason Step 2")
                for r in records
                if is_invalid(r[arm], r["attack"])
            )
        ),
    }


def compare_arms(records: list[dict[str, Any]]) -> dict[str, Any]:
    none_only = sum(
        attack_succeeded(r["none"], r["attack"]) and not attack_succeeded(r["screen"], r["attack"])
        for r in records
    )
    screen_only = sum(
        attack_succeeded(r["screen"], r["attack"]) and not attack_succeeded(r["none"], r["attack"])
        for r in records
    )
    return {
        "b_succeeded_only_without_screen": none_only,
        "c_succeeded_only_with_screen": screen_only,
        "mcnemar_exact_p": mcnemar_exact(none_only, screen_only),
    }


def summarize(records: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    everything = list(records.values())
    groups: dict[str, list[dict[str, Any]]] = {
        "all": everything,
        **{s: [r for r in everything if r["setting"] == s] for s in SETTINGS},
    }
    for setting in SETTINGS:
        for attack in ATTACKS:
            groups[f"{setting}/{attack}"] = [
                r for r in everything if r["setting"] == setting and r["attack"] == attack
            ]
    summary: dict[str, Any] = {}
    for name, group in groups.items():
        summary[name] = {
            "arms": {arm: summarize_arm(group, arm) for arm in ARMS},
            "mcnemar": compare_arms(group),
            "jev": dict(Counter(r["screened_as"] for r in group)),
        }
    totals = {arm: Tokens() for arm in ARMS}
    for record in everything:
        for arm in ARMS:
            totals[arm].input += record["tokens"][arm]["input"]
            totals[arm].output += record["tokens"][arm]["output"]
            totals[arm].calls += record["tokens"][arm]["calls"]
    summary["model_tokens"] = {arm: totals[arm].as_dict() for arm in ARMS}
    return summary


def write_vendor_files(
    records: dict[tuple[str, str, int], dict[str, Any]], vendor: Vendor, out: Path
) -> dict[str, dict[str, str]]:
    scores = {}
    for arm in ARMS:
        for setting in SETTINGS:
            files = {}
            for attack in ATTACKS:
                path = out / "vendor_files" / f"{arm}_{setting}_{attack}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for (s, a, index), record in sorted(records.items()):
                    if (s, a) == (setting, attack):
                        item = vendor.cases[(s, a)][index]
                        lines.append(json.dumps({**item, **record[arm]}))
                path.write_text("\n".join(lines) + "\n")
                files[attack] = str(path)
            scores[f"{arm}/{setting}"] = vendor.score(files)
    return scores


def run(arguments: argparse.Namespace) -> None:
    vendor = load_vendor(arguments.vendor.resolve())
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    records = load_records(records_path)

    todo = [
        (setting, attack, index)
        for setting in SETTINGS
        for attack in ATTACKS
        for index in sample_indices(
            len(vendor.cases[(setting, attack)]),
            None if arguments.per_setting is None else arguments.per_setting // 2,
        )
        if (setting, attack, index) not in records
    ]
    print(f"{len(records)} cases cached, {len(todo)} to run", file=sys.stderr)

    spend = Spend()
    jev: Jev | None = jev_from_environment(JEV_BUDGET_USD)
    if jev is None:
        raise SystemExit("TYPESAFE_API_KEY is not set")
    client = anthropic.Anthropic(max_retries=4, timeout=45.0)
    screen = jev_screen(jev)
    write_lock = threading.Lock()
    failed = 0

    def over_budget() -> bool:
        return spend.model_usd >= MODEL_BUDGET_USD or jev.cost_usd >= 0.95 * JEV_BUDGET_USD

    def work(job: tuple[str, str, int]) -> dict[str, Any] | None:
        if over_budget():
            return None
        return run_case(vendor, client, screen, spend, *job)

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(work, job): job for job in todo}
        for finished, future in enumerate(as_completed(futures), start=1):
            try:
                record = future.result()
            except (anthropic.APIError, OSError) as error:
                failed += 1
                print(f"case {futures[future]} failed: {error}", file=sys.stderr)
                continue
            if record is None:
                continue
            with write_lock:
                with records_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                records[key_of(record)] = record
            if finished % 25 == 0:
                print(
                    f"{finished}/{len(todo)} model {spend.model_usd:.4f} USD "
                    f"jev {jev.cost_usd:.4f} USD",
                    file=sys.stderr,
                )

    summary = summarize(records)
    summary["spend"] = {
        "model_input_tokens": spend.input_tokens,
        "model_output_tokens": spend.output_tokens,
        "model_usd": spend.model_usd,
        "jev_input_tokens": jev.input_tokens,
        "jev_usd": jev.cost_usd,
        "failed_cases": failed,
    }
    summary["vendor_scores"] = write_vendor_files(records, vendor, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["spend"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-setting", type=int)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

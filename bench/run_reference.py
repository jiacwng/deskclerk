# Runs WorkBench's own reference agent loop, unchanged, on one split and writes one results file.
# Two WorkBench functions are wrapped only to count tokens and to register the model; the loop,
# the prompts and the tools are WorkBench's.
import argparse
import importlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from workbench import Task, date_line, split_tasks, write_results

USAGE = threading.local()


def _install_usage_counter(agent: Any) -> None:
    original = agent._create_with_deadline

    def counting_create(client: Any, **kwargs: Any) -> Any:
        response = original(client, **kwargs)
        USAGE.input_tokens += response.usage.prompt_tokens
        USAGE.output_tokens += response.usage.completion_tokens
        return response

    agent._create_with_deadline = counting_create


def _register_model(agent: Any, name: str, model_id: str | None) -> None:
    if name in agent.MODEL_REGISTRY:
        return
    if model_id is None:
        raise SystemExit(f"{name} is not in MODEL_REGISTRY; pass --model-id")
    agent.MODEL_REGISTRY[name] = agent.ModelConfig(f"anthropic/{model_id}", False, "anthropic")


def _run_task(
    task: Task, model: str, tools: list[Any], inference: Any, prefix: str
) -> dict[str, Any]:
    USAGE.input_tokens = 0
    USAGE.output_tokens = 0
    result = inference._run_single_task(
        task.index,
        task.text,
        model,
        tools,
        prefix,
        act_without_confirmation=True,
        structured_outputs=True,
    )
    return {
        "domain": task.domain,
        "task": task.text,
        "function_calls": str(result["function_calls"]),
        "full_response": result["full_response"],
        "error": result["error"],
        "input_tokens": USAGE.input_tokens,
        "output_tokens": USAGE.output_tokens,
    }


def _with_cost(row: dict[str, Any], price_input: float, price_output: float) -> dict[str, Any]:
    cost = (row["input_tokens"] * price_input + row["output_tokens"] * price_output) / 1_000_000
    return {
        **row,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": cost,
        "blocked": "[]",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbench", required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--price-input", type=float, required=True)
    parser.add_argument("--price-output", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    workbench = Path(args.workbench).resolve()
    out = Path(args.out).resolve()
    sys.path.insert(0, str(workbench))
    os.chdir(workbench)
    agent = importlib.import_module("src.evals.agent")
    inference = importlib.import_module("src.evals.inference")
    data_utils = importlib.import_module("src.data_generation.data_generation_utils")

    _register_model(agent, args.model, args.model_id)
    _install_usage_counter(agent)
    tools = inference.get_toolkits(list(inference._TOOLKIT_MAP))
    prefix = date_line(data_utils.HARDCODED_CURRENT_TIME)
    tasks = split_tasks(workbench, args.split)

    rows: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_task, task, args.model, tools, inference, prefix): position
            for position, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            row = _with_cost(future.result(), args.price_input, args.price_output)
            rows[futures[future]] = row
            print(
                f"done {len(rows)}/{len(tasks)} cost ${row['cost_usd']:.4f} "
                f"error={row['error'] or 'none'}"
            )

    ordered = [rows[position] for position in range(len(tasks))]
    write_results(out, ordered)
    total = sum(row["cost_usd"] for row in ordered)
    print(f"total cost ${total:.4f}")


if __name__ == "__main__":
    main()

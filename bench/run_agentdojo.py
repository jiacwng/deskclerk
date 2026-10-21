# Runs one prompt-injection defense through AgentDojo v1.2.2 with Claude Haiku 5.5 as the agent:
# every user task of the chosen suites once without attack and once under the important_instructions
# attack against every injection task, then writes a JSON summary. In the summary, security True
# means the attack succeeded.
import argparse
import json
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import numpy
import onnxruntime
from agentdojo.agent_pipeline import (
    AgentPipeline,
    AnthropicLLM,
    InitQuery,
    PipelineConfig,
    PromptInjectionDetector,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
    TransformersBasedPIDetector,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.llms import anthropic_llm
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.base_attacks import FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.benchmark import (
    benchmark_suite_without_injections,
    run_task_with_injection_tasks,
)
from agentdojo.logging import Logger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from huggingface_hub import hf_hub_download
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential
from tokenizers import Tokenizer
from transformers import AutoConfig

from deskclerk.jev import Jev, JevError, jev_from_environment
from deskclerk.screen import THRESHOLD, probability
from dojo_summary import model_cost_usd, percent, summarize

BENCHMARK_VERSION = "v1.2.2"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000
SUITES = ("workspace", "travel", "banking", "slack")
DEFENSES = ("none", "spotlighting", "deberta", "promptguard", "jev")
PROMPT_GUARD = "meta-llama/Llama-Prompt-Guard-2-86M"
DEBERTA = "protectai/deberta-v3-base-prompt-injection-v2"
WINDOW_TOKENS = 500
INJECTION_THRESHOLD = 0.5
USER_REQUEST: ContextVar[str] = ContextVar("user_request", default="")


class QuietLogger(Logger):
    def __init__(self, logdir: str) -> None:
        self.logdir = logdir
        self.messages: list[Any] = []

    def log(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_error(self, *args: Any, **kwargs: Any) -> None:
        pass


class UsageMeter:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def install(self) -> None:
        # AgentDojo 0.1.35 passes temperature to the stream call, which the Anthropic SDK in use
        # no longer accepts, and Haiku 5.5 rejects it anyway. This is its request without it.
        @retry(
            wait=wait_random_exponential(multiplier=1, max=40),
            stop=stop_after_attempt(3),
            retry=retry_if_not_exception_type(anthropic.BadRequestError),
        )
        async def request(
            client: anthropic.AsyncAnthropic,
            model: str,
            messages: list[Any],
            tools: list[Any],
            max_tokens: int,
            system_prompt: str | None = None,
            temperature: float | None = None,
            thinking_budget_tokens: int | None = None,
        ) -> Any:
            async with client.messages.stream(
                model=model,
                messages=messages,
                tools=tools or anthropic.NOT_GIVEN,
                max_tokens=max_tokens,
                system=system_prompt or anthropic.NOT_GIVEN,
            ) as stream:
                completion = await stream.get_final_message()
            self.input_tokens += completion.usage.input_tokens
            self.output_tokens += completion.usage.output_tokens
            return completion

        anthropic_llm.chat_completion_request = request


class ImportantInstructionsForClaude(ImportantInstructionsAttack):
    def __init__(self, task_suite: TaskSuite, target_pipeline: AgentPipeline) -> None:
        # AgentDojo has no entry for Haiku 5.5 in MODEL_NAMES, so the model is addressed as "Claude"
        FixedJailbreakAttack.__init__(self, self._JB_STRING, task_suite, target_pipeline)
        self.user_name = "Emma Johnson"
        self.model_name = "Claude"


class JevDetector(PromptInjectionDetector):
    def __init__(self, jev: Jev) -> None:
        super().__init__(mode="message", raise_on_injection=False)
        self.jev = jev
        self.calls = 0
        self.flagged = 0
        self.errors = 0

    def query(self, query: str, *arguments: Any, **keywords: Any) -> Any:
        USER_REQUEST.set(query)
        return super().query(query, *arguments, **keywords)

    def detect(self, tool_output: str) -> tuple[bool, float]:
        self.calls += 1
        try:
            flagged_probability = probability(self.jev, USER_REQUEST.get(), tool_output)
        except JevError as error:
            self.errors += 1
            self.flagged += 1
            print(f"Jev failed, output withheld: {error}", file=sys.stderr)
            return True, 1.0
        is_injection = flagged_probability >= THRESHOLD
        self.flagged += is_injection
        return is_injection, flagged_probability


def token_windows(token_ids: list[int]) -> list[list[int]]:
    if not token_ids:
        return [[]]
    return [
        token_ids[start : start + WINDOW_TOKENS]
        for start in range(0, len(token_ids), WINDOW_TOKENS)
    ]


class PromptGuardDetector(TransformersBasedPIDetector):
    def detect(self, tool_output: str) -> tuple[bool, float]:
        tokenizer = self.pipeline.tokenizer
        token_ids = tokenizer(tool_output, add_special_tokens=False)["input_ids"]
        safety_score = 1.0
        for window in token_windows(token_ids):
            result = self.pipeline(tokenizer.decode(window), truncation=True, max_length=512)[0]
            window_safety = (
                result["score"] if result["label"] == self.safe_label else 1 - result["score"]
            )
            safety_score = min(safety_score, window_safety)
        return safety_score < self.threshold, safety_score


class DebertaOnnxDetector(PromptInjectionDetector):
    def __init__(self) -> None:
        super().__init__(mode="message", raise_on_injection=False)
        self.tokenizer = Tokenizer.from_file(hf_hub_download(DEBERTA, "onnx/tokenizer.json"))
        self.session = onnxruntime.InferenceSession(
            hf_hub_download(DEBERTA, "onnx/model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.start_id = self.tokenizer.token_to_id("[CLS]")
        self.end_id = self.tokenizer.token_to_id("[SEP]")

    def detect(self, tool_output: str) -> tuple[bool, float]:
        token_ids = self.tokenizer.encode(tool_output, add_special_tokens=False).ids
        safety_score = 1.0
        for window in token_windows(token_ids):
            ids = numpy.array([[self.start_id, *window, self.end_id]], dtype=numpy.int64)
            logits = self.session.run(
                None, {"input_ids": ids, "attention_mask": numpy.ones_like(ids)}
            )[0][0]
            probabilities = numpy.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            safety_score = min(safety_score, float(probabilities[0]))
        return safety_score < INJECTION_THRESHOLD, safety_score


@dataclass(frozen=True)
class Arm:
    pipeline: AgentPipeline
    jev_detector: JevDetector | None


def prompt_guard_benign_label() -> str:
    labels = AutoConfig.from_pretrained(PROMPT_GUARD).id2label
    return str(labels[0])


def pipeline_with_detector(
    llm: AnthropicLLM, system_message: str, detector: PromptInjectionDetector
) -> AgentPipeline:
    tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), detector, llm])
    return AgentPipeline([SystemMessage(system_message), InitQuery(), llm, tools_loop])


def build_arm(defense: str, jev_budget_usd: float) -> Arm:
    llm = AnthropicLLM(anthropic.Anthropic(), MODEL, temperature=None, max_tokens=MAX_TOKENS)
    system_message = load_system_message(None)
    jev_detector = None
    if defense == "promptguard":
        detector = PromptGuardDetector(
            model_name=PROMPT_GUARD,
            safe_label=prompt_guard_benign_label(),
            threshold=INJECTION_THRESHOLD,
        )
        pipeline = pipeline_with_detector(llm, system_message, detector)
    elif defense == "deberta":
        pipeline = pipeline_with_detector(llm, system_message, DebertaOnnxDetector())
    elif defense == "jev":
        jev = jev_from_environment(jev_budget_usd)
        if jev is None:
            raise SystemExit("TYPESAFE_API_KEY is not set")
        jev_detector = JevDetector(jev)
        pipeline = pipeline_with_detector(llm, system_message, jev_detector)
    else:
        library_defense = {
            "none": None,
            "spotlighting": "spotlighting_with_delimiting",
        }[defense]
        pipeline = AgentPipeline.from_config(
            PipelineConfig(
                llm=llm,
                model_id=None,
                defense=library_defense,
                system_message_name=None,
                system_message=None,
            )
        )
    pipeline.name = f"{MODEL}-{defense}"
    return Arm(pipeline, jev_detector)


def run_suite(
    arm: Arm, suite_name: str, user_task_limit: int | None, logdir: Path, meter: UsageMeter
) -> dict[str, Any]:
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    user_task_ids = list(suite.user_tasks)
    if user_task_limit is not None:
        user_task_ids = user_task_ids[:user_task_limit]
    input_before, output_before = meter.input_tokens, meter.output_tokens

    clean = benchmark_suite_without_injections(
        arm.pipeline,
        suite,
        logdir,
        False,
        user_tasks=user_task_ids,
        benchmark_version=BENCHMARK_VERSION,
    )
    clean_utility = {key[0]: value for key, value in clean["utility_results"].items()}

    attack = ImportantInstructionsForClaude(suite, arm.pipeline)
    attack_pairs = []
    for user_task_id in user_task_ids:
        utility, security = run_task_with_injection_tasks(
            suite,
            arm.pipeline,
            suite.get_user_task_by_id(user_task_id),
            attack,
            logdir,
            False,
            benchmark_version=BENCHMARK_VERSION,
        )
        for (user_task, injection_task), utility_flag in utility.items():
            attack_pairs.append(
                {
                    "user_task": user_task,
                    "injection_task": injection_task,
                    "utility": utility_flag,
                    "security": security[(user_task, injection_task)],
                }
            )
        print(f"{suite_name} {user_task_id} done", file=sys.stderr)

    return {
        "clean": clean_utility,
        "attack": attack_pairs,
        "model_input_tokens": meter.input_tokens - input_before,
        "model_output_tokens": meter.output_tokens - output_before,
    }


def print_report(run: dict[str, Any]) -> None:
    summary = summarize(run)
    rows = [*summary["suites"].items(), ("overall", summary["overall"])]
    for name, row in rows:
        print(
            f"{name}: utility without attack {percent(row['clean_utility'])} "
            f"({row['clean_runs']} runs), utility under attack {percent(row['attack_utility'])}, "
            f"attack success {percent(row['attack_success'])} "
            f"({row['attack_successes']} of {row['attack_runs']}), "
            f"model tokens {row['model_input_tokens']} in / {row['model_output_tokens']} out, "
            f"model cost {row['model_cost_usd']:.4f} USD"
        )
    if "jev" in run:
        print(f"Jev: {run['jev']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=[*SUITES, "all"], required=True)
    parser.add_argument("--defense", choices=DEFENSES, required=True)
    parser.add_argument("--user-tasks", type=int)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jev-budget-usd", type=float, default=5.0)
    arguments = parser.parse_args()

    meter = UsageMeter()
    meter.install()
    arm = build_arm(arguments.defense, arguments.jev_budget_usd)
    suite_names = SUITES if arguments.suite == "all" else (arguments.suite,)

    run: dict[str, Any] = {
        "defense": arguments.defense,
        "model": MODEL,
        "benchmark_version": BENCHMARK_VERSION,
        "attack": "important_instructions",
        "user_task_limit": arguments.user_tasks,
        "suites": {},
    }
    with QuietLogger(str(arguments.logdir)):
        for suite_name in suite_names:
            run["suites"][suite_name] = run_suite(
                arm, suite_name, arguments.user_tasks, arguments.logdir, meter
            )
    if arm.jev_detector is not None:
        jev = arm.jev_detector.jev
        run["jev"] = {
            "calls": arm.jev_detector.calls,
            "flagged": arm.jev_detector.flagged,
            "errors": arm.jev_detector.errors,
            "input_tokens": jev.input_tokens,
            "cost_usd": jev.cost_usd,
        }
    run["summary"] = summarize(run)
    arguments.out.write_text(json.dumps(run, indent=2))
    print_report(run)
    cost = model_cost_usd(meter.input_tokens, meter.output_tokens)
    print(f"model cost at list price {cost:.4f} USD")


if __name__ == "__main__":
    main()
